"""Matched-filter range-profile features for the radar echo detector.

The teacher's standard detector is a matched filter.  We implement it in the
frequency domain (FFT multiply by conj(S), then IFFT) and use the magnitude
range profile as the SNN input.  This keeps the frequency-domain simplification
but restores coherent integration gain that the magnitude-only spectrum loses.

H0: rx = clutter + noise
H1: rx = a*s(t-tau)*exp(j2pi fd t + jphi) + clutter + noise
"""
import argparse
import os
import numpy as np
from radar_data import gen_batch, make_chirp, N


def mf_features(rx: np.ndarray) -> np.ndarray:
    s = make_chirp()
    S = np.fft.fft(s)
    mf = np.fft.ifft(np.fft.fft(rx, axis=1) * np.conj(S)[None, :], axis=1)
    P = np.abs(mf)
    P_db = 10.0 * np.log10(P ** 2 + 1e-12)
    med = np.median(P_db, axis=1, keepdims=True)
    mad = np.median(np.abs(P_db - med), axis=1, keepdims=True) + 1e-6
    z = np.clip((P_db - med) / (1.4826 * mad), -3.0, 3.0)
    q = np.clip(np.round((z + 3.0) / 6.0 * 255.0), 0, 255).astype(np.uint8)
    return q


def make_split(rng, n):
    rx, y, snr, cnr = gen_batch(rng, n)
    return dict(x=mf_features(rx), y=y.astype(np.uint8),
                snr=snr.astype(np.float32), cnr=cnr.astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=24000)
    ap.add_argument("--val", type=int, default=4000)
    ap.add_argument("--test", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--out", default="data/radar_mf.npz")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rng = np.random.default_rng(args.seed)
    tr = make_split(rng, args.train)
    va = make_split(rng, args.val)
    te = make_split(rng, args.test)
    np.savez_compressed(args.out, bins=np.array(512),
                        x_train=tr["x"], y_train=tr["y"],
                        snr_train=tr["snr"], cnr_train=tr["cnr"],
                        x_val=va["x"], y_val=va["y"],
                        snr_val=va["snr"], cnr_val=va["cnr"],
                        x_test=te["x"], y_test=te["y"],
                        snr_test=te["snr"], cnr_test=te["cnr"])
    print(f"saved {args.out}")
    for nm, d in [("train", tr), ("val", va), ("test", te)]:
        print(f"  {nm}: x={d['x'].shape} pos={d['y'].mean():.3f} "
              f"snr=[{d['snr'].min():.1f},{d['snr'].max():.1f}]")


if __name__ == "__main__":
    main()
