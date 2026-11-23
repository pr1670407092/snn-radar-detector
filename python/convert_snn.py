"""Convert the trained ANN to the bit-accurate integer SNN used by the RTL.

Rate coding + IF neurons with soft reset.  The RTL implements:
    V <- V - (V >>> LEAK) + bias + sum_j W[j] * spike[j]
    if V >= THRESH: spike, V <- V - THRESH
with all values integer.  This module reproduces that exactly.
"""
import argparse
import os
import numpy as np
import torch

from train_ann import MLP


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = MLP(n_in=ck.get("n_in", 256), h1=ck["h1"], h2=ck["h2"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck


@torch.no_grad()
def ann_activations(model, x, batch=1024):
    """Return layer pre-activations z1,z2,z3 and activations a1,a2."""
    z1s, a1s, z2s, a2s, z3s = [], [], [], [], []
    for i in range(0, len(x), batch):
        xb = torch.from_numpy(x[i:i + batch].astype(np.float32) / 255.0)
        z1 = model.fc1(xb)
        a1 = torch.relu(z1)
        z2 = model.fc2(a1)
        a2 = torch.relu(z2)
        z3 = model.fc3(a2)
        z1s.append(z1.numpy()); a1s.append(a1.numpy())
        z2s.append(z2.numpy()); a2s.append(a2.numpy())
        z3s.append(z3.numpy())
    return (np.concatenate(z1s), np.concatenate(a1s), np.concatenate(z2s),
            np.concatenate(a2s), np.concatenate(z3s))


def percentile(v, q):
    v = np.abs(v.reshape(-1))
    p = float(np.percentile(v, q))
    if p <= 0:
        p = float(v.max()) if v.size else 1.0
    return p if p > 0 else 1.0


def choose_output_offset(z3, margin=0.5):
    # argmax is invariant to adding the same constant to both logits
    return max(0.0, float(-z3.min()) + margin)


def quantize_layer(W, b, lam_prev, lam, input_rate_scale=1.0, w_bits=8):
    Wp = W.T.astype(np.float64) * (lam_prev / lam) * input_rate_scale  # (fanin, neurons)
    bp = b.astype(np.float64) / lam
    maxw = np.abs(Wp).max()
    qmax = (1 << (w_bits - 1)) - 1
    K = max(1, int(np.floor(qmax / max(maxw, 1e-12))))
    Wq = np.clip(np.round(Wp * K), -qmax, qmax).astype(np.int8)
    bq = np.round(bp * K).astype(np.int64)
    return Wq.T.copy(), bq, K  # (neurons, fanin)


def build_snn(model, cal_x, out_percentile=99.9, hidden_percentile=99.9,
              out_margin=0.5, input_rate_scale=1.0, w_bits=8):
    z1, a1, z2, a2, z3 = ann_activations(model, cal_x)
    lam1 = percentile(a1, hidden_percentile)
    lam2 = percentile(a2, hidden_percentile)
    C = choose_output_offset(z3, out_margin)
    a3 = np.maximum(z3 + C, 0.0)
    lam3 = percentile(a3, out_percentile)
    W1 = model.fc1.weight.detach().numpy().astype(np.float64)
    b1 = model.fc1.bias.detach().numpy().astype(np.float64)
    W2 = model.fc2.weight.detach().numpy().astype(np.float64)
    b2 = model.fc2.bias.detach().numpy().astype(np.float64)
    W3 = model.fc3.weight.detach().numpy().astype(np.float64)
    b3 = model.fc3.bias.detach().numpy().astype(np.float64) + C
    Wq1, bq1, K1 = quantize_layer(W1, b1, 1.0, lam1, input_rate_scale, w_bits)
    Wq2, bq2, K2 = quantize_layer(W2, b2, lam1, lam2, 1.0, w_bits)
    Wq3, bq3, K3 = quantize_layer(W3, b3, lam2, lam3, 1.0, w_bits)
    return dict(W1=Wq1, b1=bq1, th1=K1,
                W2=Wq2, b2=bq2, th2=K2,
                W3=Wq3, b3=bq3, th3=K3,
                lam1=lam1, lam2=lam2, lam3=lam3, C=C, w_bits=w_bits)


def encode(feat, T, init_mul=73):
    """First-order sigma-delta rate encoder, bit-exact with rate_encoder.v.

    feat: (B,256) uint8. Returns (B,T,256) uint8 spikes.
    """
    B, F = feat.shape
    acc = (np.arange(F, dtype=np.int32) * init_mul) & 255
    acc = np.tile(acc, (B, 1))
    out = np.zeros((B, T, F), dtype=np.uint8)
    f = feat.astype(np.int32)
    for t in range(T):
        s = acc + f
        fire = s >= 256
        out[:, t, :] = fire.astype(np.uint8)
        acc = np.where(fire, s - 256, s)
    return out


def lif_run(spikes, Wq, bq, thresh, leak_shift=0, v_init=None, clip24=True):
    """spikes: (B,T,F) uint8; Wq: (N,F) int8; bq: (N,) int64. Returns (B,T,N) uint8."""
    B, T, F = spikes.shape
    N = Wq.shape[0]
    v = np.zeros((B, N), dtype=np.int64) if v_init is None else v_init.copy()
    wf = Wq.T.astype(np.float32)  # (F,N)
    out = np.zeros((B, T, N), dtype=np.uint8)
    for t in range(T):
        # integer dot product is exact in float32: inputs 0/1, |sum| < 2^24
        cur = np.rint(spikes[:, t, :].astype(np.float32) @ wf).astype(np.int64) + bq[None, :]
        if leak_shift:
            v = v - (v >> leak_shift)
        v = v + cur
        if clip24:
            v = np.clip(v, -(1 << 23), (1 << 23) - 1)
        fire = v >= thresh
        v = np.where(fire, v - thresh, v)
        out[:, t, :] = fire.astype(np.uint8)
    return out, v


def snn_forward(feat, snn, T, leak_shift=0, th_scale=(1.0, 1.0, 1.0), return_counts=False):
    sp0 = encode(feat, T)
    sp1, _ = lif_run(sp0, snn["W1"], snn["b1"],
                     max(1, int(round(snn["th1"] * th_scale[0]))), leak_shift)
    sp2, _ = lif_run(sp1, snn["W2"], snn["b2"],
                     max(1, int(round(snn["th2"] * th_scale[1]))), leak_shift)
    sp3, _ = lif_run(sp2, snn["W3"], snn["b3"],
                     max(1, int(round(snn["th3"] * th_scale[2]))), leak_shift)
    counts = sp3.sum(axis=1).astype(np.int64)  # (B,2)
    pred = (counts[:, 1] > counts[:, 0]).astype(np.int64)
    if return_counts:
        return pred, counts
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/ann.pt")
    ap.add_argument("--data", default="data/radar.npz")
    ap.add_argument("--cal", type=int, default=2000)
    ap.add_argument("--test", type=int, default=2000)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--pct", type=float, default=99.9)
    ap.add_argument("--leak", type=int, default=0)
    ap.add_argument("--w-bits", type=int, default=8)
    args = ap.parse_args()

    model, ck = load_model(args.model)
    d = np.load(args.data)
    cal_x = d["x_train"][:args.cal]
    test_x = d["x_test"][:args.test]
    test_y = d["y_test"][:args.test].astype(np.int64)

    snn = build_snn(model, cal_x, out_percentile=args.pct,
                    hidden_percentile=args.pct, w_bits=args.w_bits)
    print(f"w_bits={args.w_bits}  lambda: {snn['lam1']:.4f} {snn['lam2']:.4f} {snn['lam3']:.4f}  "
          f"C={snn['C']:.3f}  thresholds: {snn['th1']} {snn['th2']} {snn['th3']}")
    print(f"weight ranges: {np.abs(snn['W1']).max()} {np.abs(snn['W2']).max()} {np.abs(snn['W3']).max()}")

    # ANN reference
    with torch.no_grad():
        ann_pred = model(torch.from_numpy(test_x.astype(np.float32) / 255.0)).argmax(1).numpy()
    print(f"ANN test acc: {(ann_pred == test_y).mean()*100:.2f}%")

    for T in [16, 32, 48, 64, 96, 128]:
        pred, counts = snn_forward(test_x, snn, T, leak_shift=args.leak, return_counts=True)
        acc = (pred == test_y).mean() * 100
        print(f"T={T:3d}: SNN acc={acc:.2f}%  mean counts={counts.mean(0)}")


if __name__ == "__main__":
    main()
