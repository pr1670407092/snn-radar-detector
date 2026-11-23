"""Time-frequency residual spiking CNN (2D) on STFT graphs.

The SNN does the main 2D convolution; the readout is the global max of the
last-layer spike counts or membrane potential, followed by a small linear head.
Direct BPTT with a surrogate gradient; SEW-ADD residual at 16/32 channels.
"""
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_ann import load_data
from train_spiking_cnn import SpikeFn
from convert_snn import encode
from eval_snn import roc_auc

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SpikingCNN2D(nn.Module):
    def __init__(self, width=16, T=4, sew=True, readout="membrane", extra=0):
        super().__init__()
        self.T = T
        self.sew = sew
        self.readout = readout
        self.extra = extra
        self.c1 = nn.Conv2d(1, width, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(width)
        self.c2 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(width)
        self.c3 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.b3 = nn.BatchNorm2d(width)
        self.c4 = nn.Conv2d(width, width * 2, 3, padding=1, stride=2, bias=False)
        self.b4 = nn.BatchNorm2d(width * 2)
        self.c5 = nn.Conv2d(width * 2, width * 2, 3, padding=1, bias=False)
        self.b5 = nn.BatchNorm2d(width * 2)
        self.c6 = nn.Conv2d(width * 2, width * 2, 3, padding=1, bias=False)
        self.b6 = nn.BatchNorm2d(width * 2)
        if extra:
            self.c7 = nn.Conv2d(width * 2, width * 2, 3, padding=1, bias=False)
            self.b7 = nn.BatchNorm2d(width * 2)
            self.c8 = nn.Conv2d(width * 2, width * 2, 3, padding=1, bias=False)
            self.b8 = nn.BatchNorm2d(width * 2)
        self.fc = nn.Linear(width * 2, 2)

    @staticmethod
    def _lif(v, cur):
        v = v + cur
        s = SpikeFn.apply(v)
        v = v - s
        return s, v

    def forward(self, sp):
        B = sp.shape[0]
        v1 = torch.zeros(B, self.b1.num_features, 32, 32, device=sp.device)
        v2 = torch.zeros_like(v1)
        v3 = torch.zeros_like(v1)
        v4 = torch.zeros(B, self.b4.num_features, 16, 16, device=sp.device)
        v5 = torch.zeros_like(v4)
        v6 = torch.zeros_like(v4)
        if self.extra:
            v7 = torch.zeros_like(v4)
            v8 = torch.zeros_like(v4)
        counts = torch.zeros_like(v4)
        for t in range(self.T):
            s = sp[:, t, :, :].unsqueeze(1).float()
            s1, v1 = self._lif(v1, self.b1(self.c1(s)))
            s2, v2 = self._lif(v2, self.b2(self.c2(s1)))
            cur = self.b3(self.c3(s2))
            if self.sew:
                cur = cur + s1
            s3, v3 = self._lif(v3, cur)
            s4, v4 = self._lif(v4, self.b4(self.c4(s3)))
            s5, v5 = self._lif(v5, self.b5(self.c5(s4)))
            cur = self.b6(self.c6(s5))
            if self.sew:
                cur = cur + s4
            s6, v6 = self._lif(v6, cur)
            if self.extra:
                s7, v7 = self._lif(v7, self.b7(self.c7(s6)))
                cur = self.b8(self.c8(s7))
                if self.sew:
                    cur = cur + s6
                s8, v8 = self._lif(v8, cur)
                counts = counts + s8
            else:
                counts = counts + s6
        if self.extra:
            vout = v8
        else:
            vout = v6
        if self.readout == "membrane":
            feat = vout.amax(dim=(2, 3))
        else:
            feat = counts.amax(dim=(2, 3)) / self.T
        return self.fc(feat)


@torch.no_grad()
def evaluate(model, loader, device, T):
    model.eval()
    correct = total = 0
    for x, y in loader:
        q = (x * 255.0).round().clamp(0, 255).to(torch.uint8).numpy().reshape(len(x), -1)
        sp = torch.from_numpy(encode(q, T)).reshape(len(x), T, 32, 32).to(device)
        pred = model(sp).argmax(1)
        correct += (pred == y.to(device)).sum().item()
        total += y.numel()
    return correct / total


@torch.no_grad()
def scores(model, loader, device, T):
    model.eval()
    out = []
    for x, y in loader:
        q = (x * 255.0).round().clamp(0, 255).to(torch.uint8).numpy().reshape(len(x), -1)
        sp = torch.from_numpy(encode(q, T)).reshape(len(x), T, 32, 32).to(device)
        logits = model(sp)
        out.append((logits[:, 1] - logits[:, 0]).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/radar_tf.npz")
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--sew", type=int, default=1)
    ap.add_argument("--readout", choices=["count", "membrane"], default="membrane")
    ap.add_argument("--extra", type=int, default=0,
                    help="add a 32ch residual block (2 more conv layers)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--label-smooth", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--out", default="models/spiking_cnn2d_tf.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data = load_data(args.data)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*data["train"]), batch_size=args.batch,
        shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*data["val"]), batch_size=128, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*data["test"]), batch_size=128, shuffle=False)

    model = SpikingCNN2D(width=args.width, T=args.T, sew=bool(args.sew),
                         readout=args.readout, extra=args.extra).to(DEV)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SpikingCNN2D width={args.width} T={args.T} sew={args.sew} "
          f"readout={args.readout} extra={args.extra} params={n_params}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
    best_val, best_state = 0.0, None
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        run_loss = 0.0
        for x, y in train_loader:
            q = (x * 255.0).round().clamp(0, 255).to(torch.uint8).numpy().reshape(len(x), -1)
            sp = torch.from_numpy(encode(q, args.T)).reshape(len(x), args.T, 32, 32).to(DEV)
            y = y.to(DEV)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(sp), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run_loss += loss.item() * y.numel()
        sched.step()
        val = evaluate(model, val_loader, DEV, args.T)
        if val > best_val:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"epoch {ep+1:3d}: loss={run_loss/len(train_loader.dataset):.4f} "
                  f"val={val*100:.2f}% lr={sched.get_last_lr()[0]:.2e} ({time.time()-t0:.1f}s)")

    model.load_state_dict(best_state)
    test_acc = evaluate(model, test_loader, DEV, args.T)
    print(f"best val={best_val*100:.2f}%  test={test_acc*100:.2f}%  params={n_params}")

    vs = scores(model, val_loader, DEV, args.T)
    ts = scores(model, test_loader, DEV, args.T)
    yv = data["val"][1].numpy().astype(int)
    yt = data["test"][1].numpy().astype(int)
    print("score AUC test", round(roc_auc(yt, ts), 4))
    for pfa in (1e-2, 5e-3, 1e-3):
        thr = np.quantile(vs[yv == 0], 1 - pfa)
        pred = (ts > thr).astype(int)
        tn = int(((pred == 0) & (yt == 0)).sum()); fp = int(((pred == 1) & (yt == 0)).sum())
        fn = int(((pred == 0) & (yt == 1)).sum()); tp = int(((pred == 1) & (yt == 1)).sum())
        print(f"target PFA={pfa:.0e}: test_PD={tp/(tp+fn)*100:.2f}% "
              f"test_PFA={fp/(fp+tn)*100:.2f}%")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(dict(state_dict=best_state, width=args.width, T=args.T,
                    sew=bool(args.sew), readout=args.readout, extra=args.extra,
                    n_params=n_params, test_acc=test_acc), args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
