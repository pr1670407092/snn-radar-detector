"""Evaluate the trained 2D time-frequency residual spiking CNN.

Reports test accuracy, PD/PFA at the argmax decision and at radar-style PFA
thresholds calibrated on validation negatives, AUC, per-SNR accuracy, measured
layer spike rates, and dense/event-driven MAC estimates.
"""
import argparse
import json
import os
import numpy as np
import torch

from train_spiking_cnn2d import SpikingCNN2D
from convert_snn import encode
from eval_snn import roc_auc

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def scores(model, loader, T):
    model.eval()
    out = []
    for x, y in loader:
        q = (x * 255.0).round().clamp(0, 255).to(torch.uint8).numpy().reshape(len(x), -1)
        sp = torch.from_numpy(encode(q, T)).reshape(len(x), T, 32, 32).to(DEV)
        logits = model(sp)
        out.append((logits[:, 1] - logits[:, 0]).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def spike_rates(model, loader, T, max_batches=8):
    """Mean firing rate of the input and each of the six spiking layers."""
    model.eval()
    sums = [0.0] * 6
    n = 0
    in_sum = in_num = 0
    for bi, (x, y) in enumerate(loader):
        q = (x * 255.0).round().clamp(0, 255).to(torch.uint8).numpy().reshape(len(x), -1)
        sp = torch.from_numpy(encode(q, T)).reshape(len(x), T, 32, 32).to(DEV)
        B = sp.shape[0]
        v1 = torch.zeros(B, model.b1.num_features, 32, 32, device=DEV)
        v2 = torch.zeros_like(v1)
        v3 = torch.zeros_like(v1)
        v4 = torch.zeros(B, model.b4.num_features, 16, 16, device=DEV)
        v5 = torch.zeros_like(v4)
        v6 = torch.zeros_like(v4)
        for t in range(T):
            s = sp[:, t, :, :].unsqueeze(1).float()
            in_sum += s.sum().item(); in_num += s.numel()
            s1, v1 = model._lif(v1, model.b1(model.c1(s)))
            s2, v2 = model._lif(v2, model.b2(model.c2(s1)))
            cur = model.b3(model.c3(s2))
            if model.sew:
                cur = cur + s1
            s3, v3 = model._lif(v3, cur)
            s4, v4 = model._lif(v4, model.b4(model.c4(s3)))
            s5, v5 = model._lif(v5, model.b5(model.c5(s4)))
            cur = model.b6(model.c6(s5))
            if model.sew:
                cur = cur + s4
            s6, v6 = model._lif(v6, cur)
            for i, si in enumerate((s1, s2, s3, s4, s5, s6)):
                sums[i] += si.sum().item()
        n += B * T
        if bi + 1 >= max_batches:
            break
    shapes = [(model.b1.num_features, 32, 32)] * 3 + [(model.b4.num_features, 16, 16)] * 3
    rates = [sums[i] / (n * c * h * w) for i, (c, h, w) in enumerate(shapes)]
    return in_sum / in_num, rates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/spiking_cnn2d_tf.pt")
    ap.add_argument("--data", default="data/radar_tf.npz")
    ap.add_argument("--out", default="docs/metrics_spiking_cnn2d.json")
    args = ap.parse_args()

    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    model = SpikingCNN2D(width=ck["width"], T=ck["T"], sew=ck["sew"],
                         readout=ck["readout"], extra=ck.get("extra", 0)).to(DEV)
    model.load_state_dict(ck["state_dict"])

    d = np.load(args.data)
    xt = torch.from_numpy(d["x_test"].astype(np.float32) / 255.0)
    yt = d["y_test"].astype(int)
    xv = torch.from_numpy(d["x_val"].astype(np.float32) / 255.0)
    yv = d["y_val"].astype(int)
    snr = d["snr_test"].astype(float)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xt, torch.from_numpy(yt)), batch_size=128, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xv, torch.from_numpy(yv)), batch_size=128, shuffle=False)

    ts = scores(model, test_loader, ck["T"])
    vs = scores(model, val_loader, ck["T"])
    pred = (ts > 0).astype(int)
    tn = int(((pred == 0) & (yt == 0)).sum()); fp = int(((pred == 1) & (yt == 0)).sum())
    fn = int(((pred == 0) & (yt == 1)).sum()); tp = int(((pred == 1) & (yt == 1)).sum())
    acc = float((pred == yt).mean()); pd = float(tp / (tp + fn)); pfa = float(fp / (fp + tn))
    auc = roc_auc(yt, ts)
    print(f"2D SNN T={ck['T']} params={ck['n_params']} acc={acc*100:.2f}% "
          f"PD={pd*100:.2f}% PFA={pfa*100:.2f}% AUC={auc:.4f}")
    print("confusion", tn, fp, fn, tp)

    pfa_tab = []
    for target in (1e-2, 5e-3, 1e-3):
        thr = float(np.quantile(vs[yv == 0], 1.0 - target))
        pr = (ts > thr).astype(int)
        tn2 = int(((pr == 0) & (yt == 0)).sum()); fp2 = int(((pr == 1) & (yt == 0)).sum())
        fn2 = int(((pr == 0) & (yt == 1)).sum()); tp2 = int(((pr == 1) & (yt == 1)).sum())
        row = dict(target_pfa=target, threshold=thr,
                   PD=float(tp2 / (tp2 + fn2)), PFA=float(fp2 / (fp2 + tn2)),
                   TN=tn2, FP=fp2, FN=fn2, TP=tp2)
        pfa_tab.append(row)
        print(f"target PFA={target:.0e}: thr={thr:.4f} test_PD={row['PD']*100:.2f}% "
              f"test_PFA={row['PFA']*100:.2f}%")

    bins = []
    for lo, hi in [(-8, -5), (-5, -2), (-2, 1), (1, 4)]:
        m = (snr >= lo) & (snr < hi)
        bins.append(dict(lo=lo, hi=hi, n=int(m.sum()), acc=float((pred[m] == yt[m]).mean())))
        print(f"  SNR [{lo},{hi}): n={m.sum()} acc={100*(pred[m]==yt[m]).mean():.2f}%")

    in_rate, layer_rates = spike_rates(model, test_loader, ck["T"])
    print("input firing rate", round(in_rate, 4))
    print("layer firing rates", [round(r, 4) for r in layer_rates])

    macs = 32*32*16*9 + 2*(32*32*16*16*9) + 16*16*32*16*9 + 2*(16*16*32*32*9)
    conv_macs = [147456, 2359296, 2359296, 1179648, 2359296, 2359296]
    # conv i consumes the spikes of its input: input, s1, s2, s3, s4, s5
    in_rates = [in_rate, layer_rates[0], layer_rates[1], layer_rates[2],
                layer_rates[3], layer_rates[4]]
    event_macs = sum(m * r for m, r in zip(conv_macs, in_rates))
    out = dict(model="STFT 32x32 + 2D residual spiking CNN (time-frequency RSN)",
               params=ck["n_params"], T=ck["T"], sew=ck["sew"], readout=ck["readout"],
               accuracy=acc, PD=pd, PFA=pfa, auc=auc,
               confusion=dict(TN=tn, FP=fp, FN=fn, TP=tp),
               pfa_table=pfa_tab, snr_bins=bins,
               input_firing_rate=in_rate, layer_firing_rates=layer_rates,
               macs_per_timestep=macs, macs_total=macs * ck["T"],
               latency_ms_64lane_100mhz=macs * ck["T"] / (64 * 100e6) * 1e3,
               latency_ms_128lane_100mhz=macs * ck["T"] / (128 * 100e6) * 1e3,
               event_driven_macs_total=event_macs * ck["T"],
               event_driven_latency_ms_64lane_100mhz=event_macs * ck["T"] / (64 * 100e6) * 1e3,
               event_driven_latency_ms_128lane_100mhz=event_macs * ck["T"] / (128 * 100e6) * 1e3,
               conv_weights=27792, neurons=73728,
               membrane_bram_estimate=48, weight_bram_estimate=7,
               bram_estimate=60,
               note="STFT/FFT front-end is non-spiking; all six conv layers are spiking. "
                    "Event-driven estimate assumes uniform spikes at the measured mean rates.")
    json.dump(out, open(args.out, "w"), indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
