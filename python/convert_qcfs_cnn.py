"""QCFS-converted spiking 1D CNN on matched-filter range profiles.

The QCFS ANN (quantization clip-floor-shift activation, L levels) is converted
to an IF network with T=L:
  - BN folded into conv;
  - IF threshold = 1, soft reset;
  - global max of spike counts over T;
  - the ANN's linear head on max_count/T.

The SNN performs the six conv layers itself; only the matched filter / FFT is a
non-spiking radar front-end.
"""
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F

from train_cnn import CNN1D, QCFS
from convert_snn import encode
from eval_snn import roc_auc

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fold_bn(conv, bn):
    w = conv.weight.detach().float()
    gamma = bn.weight.detach().float()
    beta = bn.bias.detach().float()
    mean = bn.running_mean.float()
    var = bn.running_var.float()
    scale = gamma / torch.sqrt(var + bn.eps)
    return w * scale[:, None, None], beta - mean * scale


def build_layers(model):
    return [
        (model.stem[0], model.stem[1], 1, 3),
        (model.res1.c1, model.res1.b1, 1, 1),
        (model.res1.c2, model.res1.b2, 1, 1),
        (model.down[0], model.down[1], 2, 1),
        (model.res2.c1, model.res2.b1, 1, 1),
        (model.res2.c2, model.res2.b2, 1, 1),
    ]


@torch.no_grad()
def snn_features(x_uint8, layers, T, batch=512):
    feats = []
    for i in range(0, len(x_uint8), batch):
        xb = x_uint8[i:i + batch]
        sp = torch.from_numpy(encode(xb, T)).to(DEV)
        B = sp.shape[0]
        lens = [512, 512, 512, 256, 256, 256]
        vs = [torch.zeros(B, L["w"].shape[0], ln, device=DEV)
              for L, ln in zip(layers, lens)]
        counts = torch.zeros(B, layers[-1]["w"].shape[0], lens[-1], device=DEV)
        for t in range(T):
            s = sp[:, t, :].unsqueeze(1).float()
            for li, L in enumerate(layers):
                cur = F.conv1d(s, L["w"], L["b"], stride=L["stride"], padding=L["pad"])
                v = vs[li] + cur
                fire = v >= 1.0
                vs[li] = torch.where(fire, v - 1.0, v)
                s = fire.float()
            counts += s
        feats.append(counts.max(dim=2).values.cpu())
    return torch.cat(feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/qcfs_cnn_L8.pt")
    ap.add_argument("--data", default="archive/mf_mlp_experiment/radar_mf.npz")
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--out", default="docs/metrics_qcfs_snn.json")
    args = ap.parse_args()

    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    L = int(ck.get("L", args.T))
    model = CNN1D(width=ck["width"], pool=ck["pool"], residual=ck.get("residual", False),
                  act=QCFS(L))
    model.load_state_dict(ck["state_dict"])
    model.eval().to(DEV)

    d = np.load(args.data)
    xcal = d["x_train"][:4000]
    xt = d["x_test"]
    yt = d["y_test"]
    snr = d["snr_test"]

    # QCFS ANN reference
    with torch.no_grad():
        ann = model(torch.from_numpy(xt.astype(np.float32) / 255.0).to(DEV)).argmax(1).cpu().numpy()
    print("QCFS ANN test:", 100 * (ann == yt).mean())

    layers = []
    for i, (conv, bn, stride, pad) in enumerate(build_layers(model)):
        w, b = fold_bn(conv, bn)
        if i == 0:
            w = w * (256.0 / 255.0)  # encoder rate q/256 vs ANN q/255
        layers.append(dict(w=w, b=b, stride=stride, pad=pad))

    fcal = snn_features(xcal, layers, L)
    ftest = snn_features(xt, layers, L)
    with torch.no_grad():
        logits = model.fc(ftest.to(DEV) / L).cpu().numpy()
    pred = logits.argmax(1)
    tn = int(((pred == 0) & (yt == 0)).sum())
    fp = int(((pred == 1) & (yt == 0)).sum())
    fn = int(((pred == 0) & (yt == 1)).sum())
    tp = int(((pred == 1) & (yt == 1)).sum())
    acc = float((pred == yt).mean())
    pd = float(tp / (tp + fn))
    pfa = float(fp / (fp + tn))
    auc = roc_auc(yt, logits[:, 1] - logits[:, 0])
    print(f"QCFS SNN T={L}: acc={acc*100:.2f}% PD={pd*100:.2f}% PFA={pfa*100:.2f}% AUC={auc:.4f}")
    print("confusion", tn, fp, fn, tp, "params", ck["n_params"])
    bins = []
    for lo, hi in [(-8, -5), (-5, -2), (-2, 1), (1, 4)]:
        m = (snr >= lo) & (snr < hi)
        bins.append([lo, hi, int(m.sum()), float((pred[m] == yt[m]).mean())])
        print(f"  SNR [{lo},{hi}): n={m.sum()} acc={100*(pred[m]==yt[m]).mean():.2f}%")

    out = dict(model="MF + QCFS 1D CNN, spiking conv + global max + linear head",
               params=ck["n_params"], T=L, qcfs_L=L,
               qcfs_ann_accuracy=float((ann == yt).mean()),
               snn_accuracy=acc, PD=pd, PFA=pfa, auc=auc,
               confusion=dict(TN=tn, FP=fp, FN=fn, TP=tp), snr_bins=bins)
    json.dump(out, open(args.out, "w"), indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
