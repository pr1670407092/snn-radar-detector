"""2D residual CNN on STFT time-frequency graphs (time-frequency RSN prototype).

First the ANN: if this beats the 1D MF CNN, the next step is a residual spiking
2D CNN with membrane-voltage readout / PFA control (GRSL 2025 RSN style).
"""
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_ann import load_data


class ResBlock2d(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class CNN2D(nn.Module):
    def __init__(self, width=16, pool="max", dropout=0.0):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width), nn.ReLU())
        self.res1 = ResBlock2d(width)
        self.down = nn.Sequential(
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 2), nn.ReLU())
        self.res2 = ResBlock2d(width * 2)
        self.pool = pool
        self.dropout = dropout
        self.fc = nn.Linear(width * 2, 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.res1(x)
        x = self.down(x)
        x = self.res2(x)
        if self.pool == "max":
            x = F.adaptive_max_pool2d(x, 1)
        else:
            x = F.adaptive_avg_pool2d(x, 1)
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
    ap.add_argument("--data", default="data/radar_tf.npz")
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--pool", choices=["avg", "max"], default="max")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--label-smooth", type=float, default=0.05)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--out", default="models/cnn2d_tf.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    data = load_data(args.data)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*data["train"]), batch_size=args.batch,
        shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*data["val"]), batch_size=256, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*data["test"]), batch_size=256, shuffle=False)

    model = CNN2D(width=args.width, pool=args.pool, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"CNN2D width={args.width} pool={args.pool} params={n_params} data={args.data}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
    best_val, best_state = 0.0, None
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        run_loss = 0.0
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            if args.noise > 0:
                x = torch.clamp(x + args.noise * torch.randn_like(x), 0, 1)
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

    model.load_state_dict(best_state)
    test_acc = evaluate(model, test_loader, device)
    print(f"best val={best_val*100:.2f}%  test={test_acc*100:.2f}%  params={n_params}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(dict(state_dict=best_state, width=args.width, pool=args.pool,
                    n_params=n_params, test_acc=test_acc, data=args.data), args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
