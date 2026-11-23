"""Synthetic LFM radar target-detection dataset with frequency-domain features.

Signal model (complex baseband):
    rx[n] = a * s[n-d] * exp(j*2*pi*fd*n/fs) * exp(j*phi)
            + clutter[n] + noise[n]
Only the positive-frequency half of the FFT (256 bins) is used.
Each sample is robustly normalized (median/MAD of the log spectrum), so the
classifier has to use spectral *shape*, not absolute energy.
"""
import argparse
import os
import numpy as np

# ---- radar / sampling parameters ----
FS = 50e6          # 50 MHz sampling
N = 512            # samples per pulse
TP = 10e-6         # chirp duration
B = 20e6           # chirp bandwidth
K = B / TP
N_BINS = 256       # positive-frequency FFT bins (0 .. fs/2)


def make_chirp():
    n = np.arange(N)
    t = n / FS
    s = np.exp(1j * np.pi * K * t ** 2)
    s[t > TP] = 0.0
    return s


def gen_batch(rng, n_samples: int):
    """Return complex received signals, shape (n_samples, N), and labels."""
    y = rng.integers(0, 2, size=n_samples)
    n_idx = np.arange(N)

    # target echo: delayed / Doppler-shifted / random-phase chirp
    s = make_chirp()
    S = np.fft.fft(s)
    k = np.arange(N)
    delay = rng.uniform(0.0, 160.0, size=n_samples)          # samples
    echo = np.fft.ifft(S[None, :] * np.exp(-2j * np.pi * k[None, :] * delay[:, None] / N), axis=1)

    snr_db = rng.uniform(-7.0, 3.0, size=n_samples)
    amp = 10 ** (snr_db / 20.0) * 10 ** (rng.normal(0, 1.5, n_samples) / 20.0)
    fd = rng.uniform(-80e3, 80e3, size=n_samples)
    phase = rng.uniform(0, 2 * np.pi, size=n_samples)
    echo *= (amp * np.exp(1j * phase))[:, None]
    echo *= np.exp(1j * 2 * np.pi * fd[:, None] * n_idx[None, :] / FS)
    echo *= y[:, None]  # only present for class 1

    # low-frequency AR(1) clutter + a few slow sinusoids
    rho = rng.uniform(0.94, 0.995, size=n_samples)
    wn = (rng.standard_normal((n_samples, N)) + 1j * rng.standard_normal((n_samples, N))) / np.sqrt(2)
    clutter = np.zeros_like(wn)
    for i in range(1, N):
        clutter[:, i] = rho * clutter[:, i - 1] + np.sqrt(1 - rho ** 2) * wn[:, i]
    n_sin = 3
    f_sin = rng.uniform(0.3e6, 6e6, size=(n_samples, n_sin))
    a_sin = 10 ** (rng.normal(0, 4, (n_samples, n_sin)) / 20.0)
    p_sin = rng.uniform(0, 2 * np.pi, (n_samples, n_sin))
    for j in range(n_sin):
        clutter += (a_sin[:, j] * np.exp(1j * p_sin[:, j]))[:, None] * \
                   np.exp(1j * 2 * np.pi * f_sin[:, j][:, None] * n_idx[None, :] / FS)

    cnr_db = rng.uniform(8.0, 16.0, size=n_samples)
    clutter *= np.sqrt(10 ** (cnr_db / 10.0) / np.mean(np.abs(clutter) ** 2, axis=1))[:, None]

    noise = (rng.standard_normal((n_samples, N)) + 1j * rng.standard_normal((n_samples, N))) / np.sqrt(2)
    rx = echo + clutter + noise
    return rx, y, snr_db, cnr_db


def extract_features(rx: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """FFT magnitude -> robust z-score -> uint8 in [0,255]."""
    X = np.fft.fft(rx, axis=1)[:, :n_bins]
    P = np.abs(X) ** 2
    P_db = 10.0 * np.log10(P + 1e-12)
    med = np.median(P_db, axis=1, keepdims=True)
    mad = np.median(np.abs(P_db - med), axis=1, keepdims=True) + 1e-6
    z = (P_db - med) / (1.4826 * mad)
    z = np.clip(z, -3.0, 3.0)
    x = (z + 3.0) / 6.0
    q = np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)
    return q


def make_split(rng, n, name, n_bins=N_BINS):
    rx, y, snr, cnr = gen_batch(rng, n)
    x = extract_features(rx, n_bins)
    return dict(x=x, y=y.astype(np.uint8), snr=snr.astype(np.float32), cnr=cnr.astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=8000)
    ap.add_argument("--val", type=int, default=2000)
    ap.add_argument("--test", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20261123)
    ap.add_argument("--bins", type=int, default=N_BINS)
    ap.add_argument("--out", default="data/radar.npz")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rng = np.random.default_rng(args.seed)
    tr = make_split(rng, args.train, "train", args.bins)
    va = make_split(rng, args.val, "val", args.bins)
    te = make_split(rng, args.test, "test", args.bins)
    np.savez_compressed(
        args.out,
        x_train=tr["x"], y_train=tr["y"], snr_train=tr["snr"], cnr_train=tr["cnr"],
        x_val=va["x"], y_val=va["y"], snr_val=va["snr"], cnr_val=va["cnr"],
        x_test=te["x"], y_test=te["y"], snr_test=te["snr"], cnr_test=te["cnr"],
    )
    print(f"saved {args.out}")
    for nm, d in [("train", tr), ("val", va), ("test", te)]:
        print(f"  {nm}: x={d['x'].shape} pos={d['y'].mean():.3f} "
              f"snr=[{d['snr'].min():.1f},{d['snr'].max():.1f}] "
              f"cnr=[{d['cnr'].min():.1f},{d['cnr'].max():.1f}]")


if __name__ == "__main__":
    main()
