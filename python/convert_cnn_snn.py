"""Rate-coded spiking version of the plain 1D CNN on matched-filter profiles.

Pipeline:
  MF range profile (512 uint8)
    -> sigma-delta rate coding (T steps)
    -> 6 spiking conv layers (BN folded, int8 weights, IF soft reset)
    -> global max of per-position spike counts (32 channels)
    -> linear count readout (32 -> 2)

The conv layers are a faithful spiking feature extractor; the final readout is a
linear classifier on the T-step spike counts.  This is the algorithm prototype
for a 1D spiking-CNN accelerator; the current FC RTL cannot run convolutions.
"""
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_cnn import CNN1D
from convert_snn import encode, percentile
from eval_snn import roc_auc

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fold_bn(conv, bn):
    w = conv.weight.detach().float()
    gamma = bn.weight.detach().float()
    beta = bn.bias.detach().float()
    mean = bn.running_mean.float()
    var = bn.running_var.float()
    scale = gamma / torch.sqrt(var + bn.eps)
    w = w * scale[:, None, None]
    b = beta - mean * scale
    return w, b


def act_percentile(v, q=99.0):
    x = np.abs(v.detach().cpu().numpy()).reshape(-1)
    p = float(np.percentile(x, q))
    if p <= 0:
        p = float(np.percentile(x, 99.9))
    if p <= 0:
        p = float(x.max()) if x.size and x.max() > 0 else 1.0
    return p


@torch.no_grad()
def ann_activations(model, x_uint8):
    x = torch.from_numpy(x_uint8.astype(np.float32) / 255.0).to(DEV).unsqueeze(1)
    a1 = F.relu(model.stem[1](model.stem[0](x)))
    a2 = F.relu(model.res1.b1(model.res1.c1(a1)))
    a3 = F.relu(model.res1.b2(model.res1.c2(a2)))
    a4 = F.relu(model.down[1](model.down[0](a3)))
    a5 = F.relu(model.res2.b1(model.res2.c1(a4)))
    a6 = F.relu(model.res2.b2(model.res2.c2(a5)))
    return [a1, a2, a3, a4, a5, a6]


def build_layers(model):
    return [
        (model.stem[0], model.stem[1], 1, 3),
        (model.res1.c1, model.res1.b1, 1, 1),
        (model.res1.c2, model.res1.b2, 1, 1),
        (model.down[0], model.down[1], 2, 1),
        (model.res2.c1, model.res2.b1, 1, 1),
        (model.res2.c2, model.res2.b2, 1, 1),
    ]


def quantize_layers(model, cal_x, q=99.0):
    acts = ann_activations(model, cal_x)
    lams = [act_percentile(a, q) for a in acts]
    layers = build_layers(model)
    out = []
    lam_prev = 1.0
    for i, ((conv, bn, stride, pad), lam) in enumerate(zip(layers, lams)):
        w, b = fold_bn(conv, bn)
        wp = w * (lam_prev / lam)
        bp = b / lam
        if i == 0:
            wp = wp * (256.0 / 255.0)
        maxw = float(wp.abs().max())
        K = max(1, int(np.floor(127.0 / max(maxw, 1e-12))))
        wq = torch.clamp(torch.round(wp * K), -127, 127).to(torch.int8)
        bq = torch.round(bp * K).to(torch.int32)
        out.append(dict(w=wq, b=bq, theta=K, stride=stride, pad=pad,
                        lam=lam, lam_prev=lam_prev))
        lam_prev = lam
    return out, lams


@torch.no_grad()
def snn_features(x_uint8, layers, T, batch=512):
    feats = []
    for i in range(0, len(x_uint8), batch):
        xb = x_uint8[i:i + batch]
        sp = encode(xb, T)                       # (B,T,512) uint8
        sp = torch.from_numpy(sp).to(DEV)
        B = sp.shape[0]
        # output lengths of the six convs
        lens = [512, 512, 512, 256, 256, 256]
        vs = [torch.zeros(B, L["w"].shape[0], ln, device=DEV)
              for L, ln in zip(layers, lens)]
        counts = torch.zeros(B, layers[-1]["w"].shape[0], lens[-1], device=DEV)
        for t in range(T):
            s = sp[:, t, :].unsqueeze(1).float()  # (B,1,512)
            for li, L in enumerate(layers):
                cur = F.conv1d(s, L["w"].float(), L["b"].float(),
                               stride=L["stride"], padding=L["pad"])
                v = vs[li] + cur
                fire = v >= L["theta"]
                vs[li] = torch.where(fire, v - L["theta"], v)
                s = fire.float()
            counts += s
        feats.append(counts.max(dim=2).values.cpu())  # global max over range
    return torch.cat(feats)


def train_readout(fcal, ycal, ftest, ytest, epochs=300):
    mu = fcal.mean(0, keepdim=True)
    sd = fcal.std(0, keepdim=True) + 1e-6
    fcal = (fcal - mu) / sd
    ftest = (ftest - mu) / sd
    clf = nn.Linear(fcal.shape[1], 2).to(DEV)
    opt = torch.optim.Adam(clf.parameters(), lr=0.02, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    yc = torch.from_numpy(ycal.astype(np.int64)).to(DEV)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(clf(fcal.to(DEV)), yc)
        loss.backward()
        opt.step()
    with torch.no_grad():
        logits = clf(ftest.to(DEV)).cpu().numpy()
    # fold the standardization into the linear head (for hardware)
    w = clf.weight.detach().cpu() / sd
    b = clf.bias.detach().cpu() - (clf.weight.detach().cpu() * mu / sd).sum(1)
    return logits, w, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/cnn_mf_max_r0.pt")
    ap.add_argument("--data", default="archive/mf_mlp_experiment/radar_mf.npz")
    ap.add_argument("--cal", type=int, default=4000)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--pct", type=float, default=99.0)
    ap.add_argument("--out", default="docs/metrics_cnn_snn.json")
    args = ap.parse_args()

    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    model = CNN1D(width=ck["width"], pool=ck["pool"],
                  residual=ck.get("residual", False))
    model.load_state_dict(ck["state_dict"])
    model.eval().to(DEV)

    d = np.load(args.data)
    xcal = d["x_train"][:args.cal]
    ycal = d["y_train"][:args.cal]
    xt = d["x_test"]
    yt = d["y_test"]
    snr = d["snr_test"]

    layers, lams = quantize_layers(model, xcal, args.pct)
    print("lambda:", [round(v, 4) for v in lams])
    print("theta/K:", [L["theta"] for L in layers])
    print("bias ranges:", [int(L["b"].abs().max()) for L in layers])

    fcal = snn_features(xcal, layers, args.T)
    ftest = snn_features(xt, layers, args.T)
    logits, w_fold, b_fold = train_readout(fcal, ycal, ftest, yt)
    pred = logits.argmax(1)
    tn = int(((pred == 0) & (yt == 0)).sum())
    fp = int(((pred == 1) & (yt == 0)).sum())
    fn = int(((pred == 0) & (yt == 1)).sum())
    tp = int(((pred == 1) & (yt == 1)).sum())
    acc = float((pred == yt).mean())
    pd = float(tp / (tp + fn))
    pfa = float(fp / (fp + tn))
    auc = roc_auc(yt, logits[:, 1] - logits[:, 0])
    print(f"SNN CNN: acc={acc*100:.2f}% PD={pd*100:.2f}% PFA={pfa*100:.2f}% AUC={auc:.4f}")
    print("confusion", tn, fp, fn, tp, "params", ck["n_params"])
    bins = []
    for lo, hi in [(-8, -5), (-5, -2), (-2, 1), (1, 4)]:
        m = (snr >= lo) & (snr < hi)
        bins.append([lo, hi, int(m.sum()), float((pred[m] == yt[m]).mean())])
        print(f"  SNR [{lo},{hi}): n={m.sum()} acc={100*(pred[m]==yt[m]).mean():.2f}%")

    out = dict(model="MF + plain 1D CNN, spiking conv + count max + linear readout",
               params=ck["n_params"], T=args.T, pct=args.pct,
               accuracy=acc, PD=pd, PFA=pfa, auc=auc,
               confusion=dict(TN=tn, FP=fp, FN=fn, TP=tp),
               thresholds=[int(L["theta"]) for L in layers],
               lambdas=[float(v) for v in lams],
               readout_w=[[float(v) for v in r] for r in w_fold.tolist()],
               readout_b=[float(v) for v in b_fold.tolist()],
               snr_bins=bins)
    json.dump(out, open(args.out, "w"), indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
