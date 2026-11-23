"""Harder synthetic radar TFG dataset.

Compared with radar_data.py:
  - SNR -10..0 dB (was -7..3)
  - CNR 8..18 dB (was 8..16)  -> target/clutter ratio ~5 dB lower
  - K-distributed heavy-tailed clutter + sea spikes
  - narrowband interference (both classes)
  - random missing time segment (front-end dropout)
  - target chirp rate mismatch +/-4%
"""
import argparse
import os
import numpy as np
from scipy.signal import stft
from radar_data import FS, N, TP, K, make_chirp
from tf_data import tf_features


def gen_batch_hard(rng, n_samples, snr_lo=-10.0, snr_hi=0.0,
                   cnr_lo=8.0, cnr_hi=18.0, mismatch=0.02,
                   spike_prob=0.05, interf_prob=0.15, drop_prob=0.10):
    y = rng.integers(0, 2, size=n_samples)
    n_idx = np.arange(N)

    # --- target: mismatched chirp + delay + Doppler + random phase ---
    eps = rng.uniform(-mismatch, mismatch, size=n_samples)
    t = np.arange(N) / FS
    s = np.zeros((n_samples, N), dtype=np.complex128)
    for i in range(n_samples):
        si = np.exp(1j * np.pi * K * (1.0 + eps[i]) * t ** 2)
        si[t > TP] = 0.0
        s[i] = si
    S = np.fft.fft(s, axis=1)
    delay = rng.uniform(0.0, 160.0, size=n_samples)
    echo = np.fft.ifft(S * np.exp(-2j * np.pi * n_idx[None, :] * delay[:, None] / N), axis=1)
    snr_db = rng.uniform(snr_lo, snr_hi, size=n_samples)
    amp = 10 ** (snr_db / 20.0) * 10 ** (rng.normal(0, 2.0, n_samples) / 20.0)
    fd = rng.uniform(-80e3, 80e3, size=n_samples)
    phase = rng.uniform(0, 2 * np.pi, size=n_samples)
    echo *= (amp * np.exp(1j * phase))[:, None]
    echo *= np.exp(1j * 2 * np.pi * fd[:, None] * n_idx[None, :] / FS)
    echo *= y[:, None]

    # --- AR(1) + slow sinusoids (Gaussian part) ---
    rho = rng.uniform(0.95, 0.998, size=n_samples)
    wn = (rng.standard_normal((n_samples, N)) + 1j * rng.standard_normal((n_samples, N))) / np.sqrt(2)
    clutter = np.zeros_like(wn)
    for i in range(1, N):
        clutter[:, i] = rho * clutter[:, i - 1] + np.sqrt(1 - rho ** 2) * wn[:, i]
    n_sin = 3
    f_sin = rng.uniform(0.3e6, 8e6, size=(n_samples, n_sin))
    a_sin = 10 ** (rng.normal(0, 5, (n_samples, n_sin)) / 20.0)
    p_sin = rng.uniform(0, 2 * np.pi, size=(n_samples, n_sin))
    for j in range(n_sin):
        clutter += (a_sin[:, j] * np.exp(1j * p_sin[:, j]))[:, None] * \
                   np.exp(1j * 2 * np.pi * f_sin[:, j][:, None] * n_idx[None, :] / FS)

    # --- K-distributed compound-Gaussian heavy tail ---
    shape = 0.7
    texture = rng.gamma(shape, 1.0 / shape, size=n_samples)
    g = (rng.standard_normal((n_samples, N)) + 1j * rng.standard_normal((n_samples, N))) / np.sqrt(2)
    kclut = np.sqrt(texture)[:, None] * g
    kclut *= np.sqrt(1.0 / np.mean(np.abs(kclut) ** 2, axis=1))[:, None]
    clutter = 0.65 * clutter / np.sqrt(np.mean(np.abs(clutter) ** 2, axis=1))[:, None] + 0.35 * kclut

    cnr_db = rng.uniform(cnr_lo, cnr_hi, size=n_samples)
    clutter *= np.sqrt(10 ** (cnr_db / 10.0) / np.mean(np.abs(clutter) ** 2, axis=1))[:, None]

    # --- sea spikes: short, strong, random-frequency bursts ---
    spike_mask = rng.random(n_samples) < spike_prob
    for i in np.where(spike_mask)[0]:
        length = rng.integers(2, 8)
        start = rng.integers(0, N - length)
        f = rng.uniform(0.5e6, 12e6)
        a = 10 ** (rng.uniform(12, 24) / 20.0)
        p = rng.uniform(0, 2 * np.pi)
        clutter[i, start:start + length] += a * np.exp(1j * (2 * np.pi * f * n_idx[start:start + length] / FS + p))

    noise = (rng.standard_normal((n_samples, N)) + 1j * rng.standard_normal((n_samples, N))) / np.sqrt(2)
    rx = echo + clutter + noise

    # --- narrowband interference (both classes) ---
    interf = rng.random(n_samples) < interf_prob
    for i in np.where(interf)[0]:
        f = rng.uniform(1e6, 12e6)
        a = 10 ** (rng.uniform(4, 12) / 20.0)
        p = rng.uniform(0, 2 * np.pi)
        rx[i] += a * np.exp(1j * (2 * np.pi * f * n_idx / FS + p))

    # --- random contiguous dropout (both classes) ---
    drop = rng.random(n_samples) < drop_prob
    for i in np.where(drop)[0]:
        length = rng.integers(N // 20, N // 5)
        start = rng.integers(0, N - length)
        rx[i, start:start + length] = 0.0

    return rx, y, snr_db, cnr_db


def make_split(rng, n, **kw):
    rx, y, snr, cnr = gen_batch_hard(rng, n, **kw)
    x = np.stack([tf_features(r) for r in rx])
    return dict(x=x, y=y.astype(np.uint8), snr=snr.astype(np.float32), cnr=cnr.astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=24000)
    ap.add_argument("--val", type=int, default=4000)
    ap.add_argument("--test", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--out", default="data/radar_tf_hard.npz")
    ap.add_argument("--snr-lo", type=float, default=-10.0)
    ap.add_argument("--snr-hi", type=float, default=0.0)
    ap.add_argument("--cnr-lo", type=float, default=8.0)
    ap.add_argument("--cnr-hi", type=float, default=18.0)
    args = ap.parse_args()
    kw = dict(snr_lo=args.snr_lo, snr_hi=args.snr_hi,
              cnr_lo=args.cnr_lo, cnr_hi=args.cnr_hi)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rng = np.random.default_rng(args.seed)
    tr = make_split(rng, args.train, **kw)
    va = make_split(rng, args.val, **kw)
    te = make_split(rng, args.test, **kw)
    np.savez_compressed(args.out,
                        x_train=tr["x"], y_train=tr["y"], snr_train=tr["snr"], cnr_train=tr["cnr"],
                        x_val=va["x"], y_val=va["y"], snr_val=va["snr"], cnr_val=va["cnr"],
                        x_test=te["x"], y_test=te["y"], snr_test=te["snr"], cnr_test=te["cnr"])
    print("saved", args.out)
    for nm, d in [("train", tr), ("val", va), ("test", te)]:
        print(f"  {nm}: x={d['x'].shape} pos={d['y'].mean():.3f} "
              f"snr=[{d['snr'].min():.1f},{d['snr'].max():.1f}] "
              f"cnr=[{d['cnr'].min():.1f},{d['cnr'].max():.1f}]")


if __name__ == "__main__":
    main()
