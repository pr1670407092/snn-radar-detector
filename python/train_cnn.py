"""Compact 1D CNN / residual CNN for the 512-bin radar feature vector.

This is the algorithm prototype for the next architecture:
  - the MLP baseline uses 164k-525k weights;
  - this CNN uses ~10k weights by sharing local kernels;
  - global max/avg pooling gives shift invariance (important for the matched-
    filter range profile, where the target is a peak at a random position).

For SNN conversion the BatchNorm layers can be folded into the preceding conv.
"""
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_ann import load_data


class QCFS(nn.Module):
    """Quantization clip-floor-shift activation (Bu et al., ICLR 2022).

    Forward is quantized to L+1 levels in [0,1]; backward is straight-through.
    """
    def __init__(self, L=8):
        super().__init__()
        self.L = L

    def forward(self, x):
        # QCFS: clip(floor(x*L)+0.5, 0, L) / L
        q = torch.clip(torch.floor(x * self.L) + 0.5, 0.0, float(self.L)) / self.L
        return x + (q - x).detach()


class ResBlock(nn.Module):
    def __init__(self, ch, residual=True, act=None):
        super().__init__()
        self.residual = residual
        self.act = act if act is not None else nn.ReLU()
        self.c1 = nn.Conv1d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm1d(ch)
        self.c2 = nn.Conv1d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm1d(ch)

    def forward(self, x):
        y = self.act(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        if self.residual:
            y = x + y
        return self.act(y)


class CNN1D(nn.Module):
    def __init__(self, width=16, pool="avg", dropout=0.0, residual=True, act=None):
        super().__init__()
        act = act if act is not None else nn.ReLU()
        self.act = act
        self.stem = nn.Sequential(
            nn.Conv1d(1, width, 7, padding=3, bias=False),
            nn.BatchNorm1d(width),
            act,
        )
        self.res1 = ResBlock(width, residual=residual, act=act)
        self.down = nn.Sequential(
            nn.Conv1d(width, width * 2, 3, padding=1, stride=2, bias=False),
            nn.BatchNorm1d(width * 2),
            act,
        )
        self.res2 = ResBlock(width * 2, residual=residual, act=act)
        self.pool_kind = pool
        self.dropout = dropout
        self.fc = nn.Linear(width * 2, 2)

    def forward(self, x):
        x = x.unsqueeze(1)                       # (B,1,512)
        x = self.stem(x)
        x = self.res1(x)
        x = self.down(x)
        x = self.res2(x)
        if self.pool_kind == "max":
            x = F.adaptive_max_pool1d(x, 1)
        else:
            x = F.adaptive_avg_pool1d(x, 1)
        x = x.flatten(1)
        x = F.dropout(x, self.dropout, self.training)
        return self.fc(x)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        pred = model(x.to(device)).argmax(1)
        correct += (pred == y.to(device)).sum().item()
        total += y.numel()
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/radar512.npz")
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--pool", choices=["avg", "max"], default="avg")
    ap.add_argument("--residual", type=int, default=1)
    ap.add_argument("--act", choices=["relu", "qcfs"], default="relu")
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--label-smooth", type=float, default=0.05)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--out", default="models/cnn1d.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    data = load_data(args.data)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*data["train"]),
        batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*data["val"]), batch_size=512, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*data["test"]), batch_size=512, shuffle=False)

    act = nn.ReLU() if args.act == "relu" else QCFS(args.L)
    model = CNN1D(width=args.width, pool=args.pool, dropout=args.dropout,
                  residual=bool(args.residual), act=act).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"CNN1D width={args.width} pool={args.pool} params={n_params} "
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
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if args.noise > 0:
                x = torch.clamp(x + args.noise * torch.randn_like(x), 0.0, 1.0)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * y.numel()
        sched.step()
        val = evaluate(model, val_loader, device)
        if val > best_val:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"epoch {ep+1:3d}: loss={run_loss/len(train_loader.dataset):.4f} "
                  f"val={val*100:.2f}% lr={sched.get_last_lr()[0]:.2e} ({time.time()-t0:.1f}s)")

    if best_state is not None:
        model.load_state_dict(best_state)
    test_acc = evaluate(model, test_loader, device)
    print(f"best val={best_val*100:.2f}%  test={test_acc*100:.2f}%  params={n_params}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(dict(state_dict={k: v.cpu() for k, v in model.state_dict().items()},
                    width=args.width, pool=args.pool, dropout=args.dropout, residual=bool(args.residual),
                    act=args.act, L=args.L,
                    n_params=n_params, test_acc=test_acc, data=args.data), args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
