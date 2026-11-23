"""Train the ANN that will be converted to an integer SNN.

Architecture: 256 -> H1 -> H2 -> 2, ReLU hidden, linear output.
No bias in the first layer?  We keep biases: hardware stores them and adds them
as a constant input current, which maps cleanly to rate-coded IF neurons.
"""
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn


def load_data(path):
    d = np.load(path)
    out = {}
    for s in ("train", "val", "test"):
        x = d[f"x_{s}"].astype(np.float32) / 255.0
        y = d[f"y_{s}"].astype(np.int64)
        out[s] = (torch.from_numpy(x), torch.from_numpy(y))
    return out


class MLP(nn.Module):
    def __init__(self, n_in=256, h1=512, h2=704, n_out=2, dropout=0.0, bias_init=0.05):
        super().__init__()
        self.fc1 = nn.Linear(n_in, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, n_out)
        self.dropout = dropout
        # keep ReLU units alive at initialization
        for layer in (self.fc1, self.fc2, self.fc3):
            nn.init.constant_(layer.bias, bias_init)

    def forward(self, x):
        import torch.nn.functional as F
        a1 = F.relu(self.fc1(x))
        a1 = F.dropout(a1, p=self.dropout, training=self.training)
        a2 = F.relu(self.fc2(a1))
        a2 = F.dropout(a2, p=self.dropout, training=self.training)
        return self.fc3(a2)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        pred = model(x.to(device)).argmax(1)
        correct += (pred == y.to(device)).sum().item()
        total += y.numel()
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data", default="data/radar.npz")
    ap.add_argument("--h1", type=int, default=512)
    ap.add_argument("--h2", type=int, default=704)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--label-smooth", type=float, default=0.0)
    ap.add_argument("--bias-init", type=float, default=0.05)
    ap.add_argument("--out", default="models/ann.pt")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    print(f"device: {device}")
    data = load_data(args.data)
    train_ds = torch.utils.data.TensorDataset(*data["train"])
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*data["val"]), batch_size=512, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*data["test"]), batch_size=512, shuffle=False)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                                               drop_last=True)

    n_in = data["train"][0].shape[1]
    model = MLP(n_in=n_in, h1=args.h1, h2=args.h2, dropout=args.dropout,
                bias_init=args.bias_init).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_in}-{args.h1}-{args.h2}-2, params={n_params}, "
          f"dropout={args.dropout} wd={args.wd} noise={args.noise}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)

    best_val = 0.0
    best_state = None
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
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"epoch {ep+1:3d}: loss={run_loss/len(train_ds):.4f} val={val*100:.2f}% "
                  f"lr={sched.get_last_lr()[0]:.2e} ({time.time()-t0:.1f}s)")

    if best_state is not None:
        model.load_state_dict(best_state)
    test_acc = evaluate(model, test_loader, device)
    print(f"best val={best_val*100:.2f}%  test={test_acc*100:.2f}%")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(dict(state_dict=cpu_state,
                    h1=args.h1, h2=args.h2, n_in=n_in,
                    test_acc=test_acc, val_acc=best_val), args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
