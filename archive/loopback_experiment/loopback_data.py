"""Synthetic radar loopback detection dataset.

Detection problem (known-signal detection, not target detection):

    H0: rx[n] = noise[n]
    H1: rx[n] = a * s[n-d] * exp(j*phi) + noise[n]

s[n] is the known transmit LFM chirp, d is the loopback delay (cable / RF
chain), a is the attenuated amplitude, phi is a random phase.  There is no
target, no Doppler and no clutter: the question is simply whether the
loopback signal/path is present.

Features are the same 512-bin FFT magnitude + robust normalization used by
the SNN hardware.
"""
import argparse
import os
import numpy as np
from radar_data import make_chirp, extract_features, N


def gen_batch(rng, n_samples: int, snr_min: float, snr_max: float):
    y = rng.integers(0, 2, size=n_samples)

    s = make_chirp()
    S = np.fft.fft(s)
    k = np.arange(N)
    delay = rng.uniform(0.0, 64.0, size=n_samples)       # loopback delay (samples)
    loop = np.fft.ifft(S[None, :] * np.exp(-2j * np.pi * k[None, :] * delay[:, None] / N), axis=1)

    snr_db = rng.uniform(snr_min, snr_max, size=n_samples)
    amp = 10 ** (snr_db / 20.0) * 10 ** (rng.normal(0, 2.0, size=n_samples) / 20.0)
    phase = rng.uniform(0, 2 * np.pi, size=n_samples)
    loop *= (amp * np.exp(1j * phase))[:, None]
    loop *= y[:, None]                                   # H0: no loopback

    noise = (rng.standard_normal((n_samples, N)) +
             1j * rng.standard_normal((n_samples, N))) / np.sqrt(2)
    rx = loop + noise
    return rx, y, snr_db, delay


def make_split(rng, n, snr_min, snr_max):
    rx, y, snr, delay = gen_batch(rng, n, snr_min, snr_max)
    x = extract_features(rx, 512)
    return dict(x=x, y=y.astype(np.uint8), snr=snr.astype(np.float32),
                delay=delay.astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=24000)
    ap.add_argument("--val", type=int, default=4000)
    ap.add_argument("--test", type=int, default=8000)
    ap.add_argument("--snr-min", type=float, default=-20.0)
    ap.add_argument("--snr-max", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=3030)
    ap.add_argument("--out", default="data/loopback512.npz")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rng = np.random.default_rng(args.seed)
    tr = make_split(rng, args.train, args.snr_min, args.snr_max)
    va = make_split(rng, args.val, args.snr_min, args.snr_max)
    te = make_split(rng, args.test, args.snr_min, args.snr_max)
    np.savez_compressed(
        args.out, bins=np.array(512),
        x_train=tr["x"], y_train=tr["y"], snr_train=tr["snr"], delay_train=tr["delay"],
        x_val=va["x"], y_val=va["y"], snr_val=va["snr"], delay_val=va["delay"],
        x_test=te["x"], y_test=te["y"], snr_test=te["snr"], delay_test=te["delay"],
    )
    print(f"saved {args.out}")
    for nm, d in [("train", tr), ("val", va), ("test", te)]:
        print(f"  {nm}: x={d['x'].shape} pos={d['y'].mean():.3f} "
              f"snr=[{d['snr'].min():.1f},{d['snr'].max():.1f}] "
              f"delay=[{d['delay'].min():.1f},{d['delay'].max():.1f}]")


if __name__ == "__main__":
    main()
