"""Full evaluation / light hyper-parameter scan for the integer SNN."""
import argparse
import json
import numpy as np
from convert_snn import load_model, build_snn, snn_forward, encode, lif_run


def confusion(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return tn, fp, fn, tp


def roc_auc(y, score):
    order = np.argsort(score)
    y = y[order]
    pos = y.sum()
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = np.arange(1, len(y) + 1)
    auc = (ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)
    return float(auc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/ann.pt")
    ap.add_argument("--data", default="data/radar.npz")
    ap.add_argument("--cal", type=int, default=2000)
    ap.add_argument("--test", type=int, default=4000)
    args = ap.parse_args()

    model, ck = load_model(args.model)
    d = np.load(args.data)
    cal_x = d["x_train"][:args.cal]
    x = d["x_test"][:args.test]
    y = d["y_test"][:args.test].astype(np.int64)
    snr = d["snr_test"][:args.test]

    best = None
    print(f"{'pct':>6} {'T':>4} {'scale':>6} {'acc':>7} {'auc':>7}  mean_counts")
    for pct in (99.0, 99.5, 99.9, 100.0):
        for scale in (1.0, 256.0 / 255.0):
            snn = build_snn(model, cal_x, out_percentile=pct,
                            hidden_percentile=pct, input_rate_scale=scale)
            for T in (32, 64):
                pred, counts = snn_forward(x, snn, T, return_counts=True)
                acc = (pred == y).mean()
                auc = roc_auc(y, counts[:, 1] - counts[:, 0])
                if best is None or acc > best[0]:
                    best = (acc, auc, pct, T, scale, counts.copy(), snn)
                print(f"{pct:6.1f} {T:4d} {scale:6.4f} {acc*100:6.2f}% {auc:7.4f}  "
                      f"{counts.mean(0).round(1)}")

    acc, auc, pct, T, scale, counts, snn = best
    pred = (counts[:, 1] > counts[:, 0]).astype(np.int64)
    tn, fp, fn, tp = confusion(y, pred)
    print("\n=== best config ===")
    print(f"pct={pct} T={T} input_scale={scale:.6f} acc={acc*100:.2f}% auc={auc:.4f}")
    print(f"confusion: TN={tn} FP={fp} FN={fn} TP={tp}  "
          f"P(detect)={tp/(tp+fn)*100:.2f}%  P(fa)={fp/(fp+tn)*100:.2f}%")
    print(f"thresholds: {snn['th1']} {snn['th2']} {snn['th3']}  "
          f"lambdas: {snn['lam1']:.4f} {snn['lam2']:.4f} {snn['lam3']:.4f}  C={snn['C']:.3f}")

    print("\naccuracy by SNR:")
    edges = [-8, -5, -2, 1, 4]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (snr >= lo) & (snr < hi)
        if m.sum():
            print(f"  SNR [{lo:4.1f},{hi:4.1f}): n={m.sum():4d} acc={(pred[m]==y[m]).mean()*100:5.2f}%")

    # threshold sweep on count margin (keep T fixed)
    best_thr = None
    for thr in np.arange(-8, 8.5, 0.5):
        p = ((counts[:, 1] - counts[:, 0]) > thr).astype(np.int64)
        a = (p == y).mean()
        if best_thr is None or a > best_thr[0]:
            best_thr = (a, thr)
    print(f"\nbest count-margin threshold: {best_thr[1]:.1f} -> {best_thr[0]*100:.2f}% "
          f"(hardware decision uses 0)")

    with open("models/snn_best.json", "w") as f:
        json.dump(dict(pct=float(pct), T=int(T), input_scale=float(scale),
                       acc=float(acc), auc=float(auc),
                       confusion=[int(v) for v in (tn, fp, fn, tp)],
                       th1=int(snn["th1"]), th2=int(snn["th2"]), th3=int(snn["th3"]),
                       lam1=float(snn["lam1"]), lam2=float(snn["lam2"]),
                       lam3=float(snn["lam3"]), C=float(snn["C"])), f, indent=2)


if __name__ == "__main__":
    main()
