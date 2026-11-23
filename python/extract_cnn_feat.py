"""Extract the 32-dim global-max features of the MF+CNN front-end.

The CNN front-end (9.7k params) runs on the matched-filter range profile; its
32 global-max features are quantized to uint8 and used as the input of a tiny
SNN classifier.  This keeps the existing FC-SNN hardware usable.
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from train_cnn import CNN1D

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/cnn_mf_max_r0.pt")
    ap.add_argument("--data", default="archive/mf_mlp_experiment/radar_mf.npz")
    ap.add_argument("--out", default="data/mf_cnn_feat.npz")
    args = ap.parse_args()
    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    model = CNN1D(width=ck["width"], pool=ck["pool"], residual=ck.get("residual", False))
    model.load_state_dict(ck["state_dict"])
    model.eval().to(DEV)

    @torch.no_grad()
    def feat(x):
        out = []
        for i in range(0, len(x), 1024):
            xb = torch.from_numpy(x[i:i+1024].astype(np.float32) / 255.0).to(DEV).unsqueeze(1)
            a = model.stem(xb)
            a = model.res1(a)
            a = model.down(a)
            a = model.res2(a)
            a = F.adaptive_max_pool1d(a, 1).flatten(1)
            out.append(a.cpu())
        return torch.cat(out).numpy()

    d = np.load(args.data)
    ftr = feat(d["x_train"]); fva = feat(d["x_val"]); fte = feat(d["x_test"])
    scale = np.percentile(ftr, 99.9, axis=0)
    scale[scale <= 0] = 1.0
    qtr = np.clip(np.round(ftr / scale * 255.0), 0, 255).astype(np.uint8)
    qva = np.clip(np.round(fva / scale * 255.0), 0, 255).astype(np.uint8)
    qte = np.clip(np.round(fte / scale * 255.0), 0, 255).astype(np.uint8)
    np.savez_compressed(args.out, bins=np.array(32),
                        x_train=qtr, y_train=d["y_train"], snr_train=d["snr_train"],
                        x_val=qva, y_val=d["y_val"], snr_val=d["snr_val"],
                        x_test=qte, y_test=d["y_test"], snr_test=d["snr_test"],
                        feat_scale=scale.astype(np.float32))
    print(f"saved {args.out}: train={qtr.shape} val={qva.shape} test={qte.shape}")
    print("feature raw range:", float(ftr.min()), float(ftr.max()),
          "scale range:", float(scale.min()), float(scale.max()))


if __name__ == "__main__":
    main()
