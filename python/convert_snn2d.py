"""Bit-exact integer model + hardware export for the 2D time-frequency RSN.

Pipeline:
  STFT 32x32 uint8 -> sigma-delta rate coding (T=4)
  -> six 2D conv layers (BN folded, int8 weights, 24-bit membrane)
  -> SEW-ADD residual at layers 3 and 6
  -> global max of the last membrane over 16x16
  -> int8 FC 32->2 -> decision.

The integer model is the reference for the RTL; export_snn2d writes weights,
biases, thresholds, FC and hardware test vectors.
"""
import argparse
import json
import os
import numpy as np
import torch
import torch.nn.functional as F

from train_spiking_cnn2d import SpikingCNN2D
from convert_snn import encode
from eval_snn import roc_auc

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# conv layer order: (Cin, Cout, stride, sew_source_index or -1)
LAYERS = [
    (1, 16, 1, -1),
    (16, 16, 1, -1),
    (16, 16, 1, 0),    # + s1
    (16, 32, 2, -1),
    (32, 32, 1, -1),
    (32, 32, 1, 3),    # + s4
]


def fold_bn(conv, bn):
    w = conv.weight.detach().float()
    gamma = bn.weight.detach().float()
    beta = bn.bias.detach().float()
    mean = bn.running_mean.float()
    var = bn.running_var.float()
    scale = gamma / torch.sqrt(var + bn.eps)
    return w * scale[:, None, None, None], beta - mean * scale


def quant_layer(w, b, input_rate_scale=1.0, w_bits=8):
    w = w * input_rate_scale
    maxw = float(w.abs().max())
    S = maxw / 127.0 if maxw > 0 else 1.0
    wq = torch.clamp(torch.round(w / S), -127, 127).to(torch.int8)
    bq = torch.round(b / S).to(torch.int64)
    th = max(1, int(round(1.0 / S)))
    return wq, bq, S, th, 1.0 / S


@torch.no_grad()
def integer_forward(model, x_uint8, T=4, batch=256):
    """Run the int8/24-bit reference; returns final FC scores and v6 max features."""
    B = len(x_uint8)
    layers = []
    for i in range(6):
        conv = getattr(model, f"c{i+1}")
        bn = getattr(model, f"b{i+1}")
        w, b = fold_bn(conv, bn)
        wq, bq, S, th, invS = quant_layer(w, b, 256.0 / 255.0 if i == 0 else 1.0)
        layers.append(dict(wq=wq, bq=bq, S=S, th=th, stride=LAYERS[i][2],
                           sew=LAYERS[i][3]))

    scores = []
    feats = []
    counts_all = []
    for s0 in range(0, B, batch):
        xb = x_uint8[s0:s0 + batch]
        sp = encode(xb.reshape(len(xb), -1), T)               # (B,T,1024)
        sp = torch.from_numpy(sp).to(DEV).reshape(len(xb), T, 1, 32, 32).float()
        outs = []
        counts_batch = []
        v = None
        for i, L in enumerate(layers):
            cin = LAYERS[i][0]
            cout = LAYERS[i][1]
            stride = L["stride"]
            v = torch.zeros(len(xb), cout, 32 if i < 3 else 16,
                            32 if i < 3 else 16, device=DEV)
            s_all = []
            for t in range(T):
                cur = F.conv2d(sp[:, t], L["wq"].float(), L["bq"].float(),
                               stride=stride, padding=1)
                if L["sew"] >= 0:
                    cur = cur + outs[L["sew"]][:, t].float() * L["th"]
                v = v + cur
                fire = (v >= L["th"]).float()
                v = v - fire * L["th"]
                s_all.append(fire)
            outs.append(torch.stack(s_all, dim=1))
            counts_batch.append(outs[-1].sum(dim=(1, 2, 3, 4)))
            sp = outs[-1]
        v6 = v
        feat = v6.amax(dim=(2, 3)).round().long()             # (B,32)
        wfc = model.fc.weight.detach().float()
        bfc = model.fc.bias.detach().float()
        S6 = layers[-1]["S"]
        Sfc = float((wfc * S6).abs().max()) / 127.0
        wfc_q = torch.clamp(torch.round(wfc * S6 / Sfc), -127, 127).long()
        bfc_q = torch.round(bfc / Sfc).long()
        score = feat.cpu() @ wfc_q.cpu().T + bfc_q.cpu()     # int64 (B,2)
        scores.append(score.cpu())
        feats.append(feat.cpu())
        counts_all.append(torch.stack(counts_batch, dim=1).cpu())
    return (torch.cat(scores).numpy(), torch.cat(feats).numpy(),
            torch.cat(counts_all).numpy(), layers, Sfc)


def pack_word(vals, width=32):
    word = 0
    for i, v in enumerate(vals):
        word |= (int(v) & 0xFF) << (8 * i)
    return word


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/spiking_cnn2d_tf.pt")
    ap.add_argument("--data", default="data/radar_tf.npz")
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--outdir", default="weights2d")
    ap.add_argument("--hwd", default="hw2d")
    ap.add_argument("--n-cases", type=int, default=8)
    args = ap.parse_args()

    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    model = SpikingCNN2D(width=ck["width"], T=ck["T"], sew=ck["sew"],
                         readout=ck["readout"]).to(DEV)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    d = np.load(args.data)
    x = d["x_test"].astype(np.uint8)
    y = d["y_test"].astype(int)
    snr = d["snr_test"].astype(float)
    scores, feats, counts, layers, Sfc = integer_forward(model, x, args.T)
    pred = (scores[:, 1] > scores[:, 0]).astype(int)
    acc = float((pred == y).mean())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tp = int(((pred == 1) & (y == 1)).sum())
    pd = float(tp / (tp + fn)); pfa = float(fp / (fp + tn))
    auc = roc_auc(y, (scores[:, 1] - scores[:, 0]).astype(float))
    print(f"integer SNN T={args.T}: acc={acc*100:.2f}% PD={pd*100:.2f}% "
          f"PFA={pfa*100:.2f}% AUC={auc:.4f}")
    print("confusion", tn, fp, fn, tp)
    print("score range", int(scores.min()), int(scores.max()))
    print("spike counts range", int(counts.min()), int(counts.max()))
    for i, L in enumerate(layers):
        print(f"  L{i+1}: Cin={LAYERS[i][0]} Cout={LAYERS[i][1]} stride={L['stride']} "
              f"sew={L['sew']} S={L['S']:.6f} th={L['th']} "
              f"bias=[{int(L['bq'].min())},{int(L['bq'].max())}]")

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.hwd, exist_ok=True)

    # weights: one 256-bit word per tap; byte co = weight(co, cin, ky, kx)
    wlines = []
    for li, L in enumerate(layers):
        wq = L["wq"]
        cout, cin = LAYERS[li][1], LAYERS[li][0]
        for tap in range(cin * 9):
            ci, r = tap // 9, tap % 9
            ky, kx = r // 3, r % 3
            vals = [int(wq[co, ci, ky, kx]) if co < cout else 0 for co in range(32)]
            wlines.append(f"{pack_word(vals):064x}")
    open(os.path.join(args.outdir, "w2d.mem"), "w").write("\n".join(wlines) + "\n")

    # bias: 32 entries per layer, 24-bit two's complement
    blines = []
    for L in layers:
        bq = L["bq"]
        for co in range(32):
            v = int(bq[co]) if co < len(bq) else 0
            blines.append(f"{v & 0xFFFFFF:06x}")
    open(os.path.join(args.outdir, "b2d.mem"), "w").write("\n".join(blines) + "\n")

    # per-layer threshold (also the SEW residual increment)
    thlines = [f"{L['th'] & 0xFFFFFF:06x}" for L in layers]
    open(os.path.join(args.outdir, "th2d.mem"), "w").write("\n".join(thlines) + "\n")

    # FC: 2 x 32 int8 weights + 2 x 48-bit biases
    wfc = model.fc.weight.detach().float()
    bfc = model.fc.bias.detach().float()
    S6 = layers[-1]["S"]
    Sfc = float((wfc * S6).abs().max()) / 127.0
    wfc_q = torch.clamp(torch.round(wfc * S6 / Sfc), -127, 127).long()
    bfc_q = torch.round(bfc / Sfc).long()
    fclines = [f"{pack_word([int(v) for v in wfc_q[c]]):064x}" for c in range(2)]
    open(os.path.join(args.outdir, "fc_w2d.mem"), "w").write("\n".join(fclines) + "\n")
    open(os.path.join(args.outdir, "fc_b2d.mem"), "w").write(
        "\n".join(f"{int(v) & 0xFFFFFFFFFFFF:012x}" for v in bfc_q) + "\n")

    # test cases: 4 positives + 4 negatives, spread across the test set
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    n = args.n_cases // 2
    pidx = pos[np.linspace(0, len(pos) - 1, n).astype(int)]
    nidx = neg[np.linspace(0, len(neg) - 1, n).astype(int)]
    idx = np.concatenate([pidx, nidx])
    for j, i in enumerate(idx):
        with open(os.path.join(args.hwd, f"feat_{j}.mem"), "w") as f:
            f.write("\n".join(f"{int(v):02x}" for v in x[i].reshape(-1)) + "\n")
    open(os.path.join(args.hwd, "expected_dec.mem"), "w").write(
        "\n".join(f"{int(pred[i])}" for i in idx) + "\n")
    open(os.path.join(args.hwd, "expected_score0.mem"), "w").write(
        "\n".join(f"{int(scores[i,0]) & 0xFFFFFFFFFFFFFFFF:016x}" for i in idx) + "\n")
    open(os.path.join(args.hwd, "expected_score1.mem"), "w").write(
        "\n".join(f"{int(scores[i,1]) & 0xFFFFFFFFFFFFFFFF:016x}" for i in idx) + "\n")
    for l in range(6):
        open(os.path.join(args.hwd, f"expected_counts_{l}.mem"), "w").write(
            "\n".join(f"{int(counts[i,l]) & 0x1FFFF:05x}" for i in idx) + "\n")

    meta = dict(model="STFT 32x32 + 2D residual spiking CNN, integer reference",
                T=args.T, params=int(ck["n_params"]), accuracy=acc, PD=pd, PFA=pfa,
                auc=auc, confusion=dict(TN=tn, FP=fp, FN=fn, TP=tp),
                score_min=int(scores.min()), score_max=int(scores.max()),
                Sfc=Sfc, fc_w=[[int(v) for v in row] for row in wfc_q.tolist()],
                fc_b=[int(v) for v in bfc_q.tolist()],
                layers=[dict(cin=LAYERS[i][0], cout=LAYERS[i][1],
                             stride=LAYERS[i][2], sew=LAYERS[i][3],
                             S=float(L["S"]), th=int(L["th"]),
                             bias_min=int(L["bq"].min()), bias_max=int(L["bq"].max()))
                        for i, L in enumerate(layers)],
                cases=dict(indices=[int(i) for i in idx],
                           labels=[int(y[i]) for i in idx],
                           decisions=[int(pred[i]) for i in idx],
                           snr=[float(snr[i]) for i in idx],
                           counts=[[int(counts[i,l]) for l in range(6)] for i in idx]))
    json.dump(meta, open(os.path.join(args.outdir, "meta2d.json"), "w"), indent=2)
    print("saved", args.outdir, "and", args.hwd)


if __name__ == "__main__":
    main()
