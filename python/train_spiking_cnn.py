"""Directly trained spiking 1D CNN on matched-filter range profiles.

The SNN does the main computation: six spiking conv layers + global max of
spike counts + a small linear readout.  Only the matched filter / FFT is a
non-spiking front-end (standard radar preprocessing).

Training uses BPTT with a surrogate gradient for the LIF; BN is used before the
LIF and can be folded into the conv for hardware.
"""
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_ann import load_data
from convert_snn import encode


class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v):
        ctx.save_for_backward(v)
        return (v >= 1.0).float()

    @staticmethod
    def backward(ctx, grad):
        v, = ctx.saved_tensors
        alpha = 2.0
        return grad * alpha / (1.0 + alpha * v.abs()) ** 2


class SpikingCNN1D(nn.Module):
    def __init__(self, width=16, T=4, sew=False, readout="count"):
        super().__init__()
        self.T = T
        self.sew = sew
        self.readout = readout
        self.c1 = nn.Conv1d(1, width, 7, padding=3, bias=False)
        self.b1 = nn.BatchNorm1d(width)
        self.c2 = nn.Conv1d(width, width, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm1d(width)
        self.c3 = nn.Conv1d(width, width, 3, padding=1, bias=False)
        self.b3 = nn.BatchNorm1d(width)
        self.c4 = nn.Conv1d(width, width * 2, 3, padding=1, stride=2, bias=False)
        self.b4 = nn.BatchNorm1d(width * 2)
        self.c5 = nn.Conv1d(width * 2, width * 2, 3, padding=1, bias=False)
        self.b5 = nn.BatchNorm1d(width * 2)
        self.c6 = nn.Conv1d(width * 2, width * 2, 3, padding=1, bias=False)
        self.b6 = nn.BatchNorm1d(width * 2)
        self.fc = nn.Linear(width * 2, 2)

    @staticmethod
    def _lif(v, cur):
        v = v + cur
        s = SpikeFn.apply(v)
        v = v - s
        return s, v

    def forward(self, sp):
        B = sp.shape[0]
        v1 = torch.zeros(B, self.b1.num_features, 512, device=sp.device)
        v2 = torch.zeros_like(v1)
        v3 = torch.zeros_like(v1)
        v4 = torch.zeros(B, self.b4.num_features, 256, device=sp.device)
        v5 = torch.zeros_like(v4)
        v6 = torch.zeros_like(v4)
        counts = torch.zeros(B, self.b6.num_features, 256, device=sp.device)
        for t in range(self.T):
            s = sp[:, t, :].unsqueeze(1).float()     # B,1,512
            cur = self.b1(self.c1(s))
            s1, v1 = self._lif(v1, cur)
            cur = self.b2(self.c2(s1))
            s2, v2 = self._lif(v2, cur)
            cur = self.b3(self.c3(s2))
            if self.sew:
                cur = cur + s1
            s3, v3 = self._lif(v3, cur)
            cur = self.b4(self.c4(s3))
            s4, v4 = self._lif(v4, cur)
            cur = self.b5(self.c5(s4))
            s5, v5 = self._lif(v5, cur)
            cur = self.b6(self.c6(s5))
            if self.sew:
                cur = cur + s4
            s6, v6 = self._lif(v6, cur)
            counts = counts + s6
        if self.readout == "membrane":
            feat = v6.max(dim=2).values               # membrane-voltage readout
        else:
            feat = counts.max(dim=2).values / self.T  # spike-count readout
        return self.fc(feat)


@torch.no_grad()
def evaluate(model, loader, device, T):
    model.eval()
    correct = total = 0
    for x, y in loader:
        q = (x * 255.0).round().clamp(0, 255).to(torch.uint8).numpy()
        sp = torch.from_numpy(encode(q, T)).to(device)
        pred = model(sp).argmax(1)
        correct += (pred == y.to(device)).sum().item()
        total += y.numel()
    return correct / total


@torch.no_grad()
def scores(model, loader, device, T):
    model.eval()
    out = []
    for x, y in loader:
        q = (x * 255.0).round().clamp(0, 255).to(torch.uint8).numpy()
        sp = torch.from_numpy(encode(q, T)).to(device)
        logits = model(sp)
        out.append((logits[:, 1] - logits[:, 0]).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="archive/mf_mlp_experiment/radar_mf.npz")
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--sew", type=int, default=0)
    ap.add_argument("--readout", choices=["count", "membrane"], default="count")
    ap.add_argument("--target-pfa", type=float, default=1e-2)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--label-smooth", type=float, default=0.05)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--out", default="models/spiking_cnn.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    data = load_data(args.data)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*data["train"]),
        batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*data["val"]), batch_size=256, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*data["test"]), batch_size=256, shuffle=False)

    model = SpikingCNN1D(width=args.width, T=args.T, sew=bool(args.sew),
                         readout=args.readout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SpikingCNN width={args.width} T={args.T} sew={args.sew} readout={args.readout} params={n_params} "
          f"device={device} data={args.data}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
    best_val, best_state = 0.0, None
    t0 = time.time()

    for ep in range(args.epochs):
        model.train()
        run_loss = 0.0
        for x, y in train_loader:
            q = (x * 255.0).round().clamp(0, 255).to(torch.uint8).numpy()
            sp = torch.from_numpy(encode(q, args.T)).to(device)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(sp), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run_loss += loss.item() * y.numel()
        sched.step()
        val = evaluate(model, val_loader, device, args.T)
        if val > best_val:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"epoch {ep+1:3d}: loss={run_loss/len(train_loader.dataset):.4f} "
                  f"val={val*100:.2f}% lr={sched.get_last_lr()[0]:.2e} ({time.time()-t0:.1f}s)")

    if best_state is not None:
        model.load_state_dict(best_state)
    test_acc = evaluate(model, test_loader, device, args.T)
    print(f"best val={best_val*100:.2f}%  test={test_acc*100:.2f}%  params={n_params}")

    # radar-style PFA calibration from the continuous score
    vs = scores(model, val_loader, device, args.T)
    ts = scores(model, test_loader, device, args.T)
    yv = data["val"][1].numpy().astype(int)
    yt = data["test"][1].numpy().astype(int)
    neg = vs[yv == 0]
    for pfa in (1e-2, 5e-3, 1e-3):
        thr = np.quantile(neg, 1.0 - pfa)
        pred = (ts > thr).astype(int)
        tn = int(((pred == 0) & (yt == 0)).sum()); fp = int(((pred == 1) & (yt == 0)).sum())
        fn = int(((pred == 0) & (yt == 1)).sum()); tp = int(((pred == 1) & (yt == 1)).sum())
        print(f"target PFA={pfa:.0e}: threshold={thr:.4f} test_PD={tp/(tp+fn)*100:.2f}% "
              f"test_PFA={fp/(fp+tn)*100:.2f}%")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(dict(state_dict={k: v.cpu() for k, v in model.state_dict().items()},
                    width=args.width, T=args.T, sew=bool(args.sew), readout=args.readout,
                    n_params=n_params, test_acc=test_acc, data=args.data), args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
