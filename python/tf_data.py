"""Time-frequency (STFT spectrogram) dataset for the time-frequency RSN prototype."""
import argparse, os
import numpy as np
from scipy.signal import stft
from radar_data import gen_batch, FS


def tf_features(rx):
    # 64-point STFT, hop 16: 29 frames x 33 freqs -> pad to 32x32
    _, _, Z = stft(rx, fs=FS, nperseg=64, noverlap=48, nfft=64,
                   boundary=None, padded=False)
    P = 20.0 * np.log10(np.abs(Z[:32, :]) + 1e-12)   # 32 x 29
    P = np.pad(P, ((0, 0), (0, 3)), mode="edge")     # 32 x 32
    med = np.median(P)
    mad = np.median(np.abs(P - med)) + 1e-6
    z = np.clip((P - med) / (1.4826 * mad), -3, 3)
    q = np.clip(np.round((z + 3) / 6 * 255), 0, 255).astype(np.uint8)
    return q


def make_split(rng, n):
    rx, y, snr, cnr = gen_batch(rng, n)
    x = np.stack([tf_features(r) for r in rx])
    return dict(x=x, y=y.astype(np.uint8), snr=snr.astype(np.float32), cnr=cnr.astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=24000)
    ap.add_argument("--val", type=int, default=4000)
    ap.add_argument("--test", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--out", default="data/radar_tf.npz")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rng = np.random.default_rng(args.seed)
    tr = make_split(rng, args.train); va = make_split(rng, args.val); te = make_split(rng, args.test)
    np.savez_compressed(args.out, x_train=tr["x"], y_train=tr["y"], snr_train=tr["snr"],
                        x_val=va["x"], y_val=va["y"], snr_val=va["snr"],
                        x_test=te["x"], y_test=te["y"], snr_test=te["snr"])
    print("saved", args.out)
    for nm, d in [("train", tr), ("val", va), ("test", te)]:
        print(f"  {nm}: x={d['x'].shape} pos={d['y'].mean():.3f} "
              f"snr=[{d['snr'].min():.1f},{d['snr'].max():.1f}]")


if __name__ == "__main__":
    main()
