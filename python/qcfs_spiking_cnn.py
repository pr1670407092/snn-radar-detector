"""QCFS 1D spiking CNN on matched-filter range profiles, following the QCFS recipe.

QCFS activation (Bu et al., ICLR 2022):
    a = lam * clip( floor(x*L/lam + 0.5) / L, 0, 1 )
with a per-layer learnable lam.  Conversion:
    W'_l = W_l * lam_{l-1} / lam_l,  b'_l = b_l / lam_l
    threshold 1, initial membrane 0.5, T = L
The SNN runs six conv layers itself; only the matched filter / FFT is non-spiking.
"""
import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_ann import load_data
from convert_snn import encode, percentile
from eval_snn import roc_auc

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QCFSFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam, L):
        q = torch.clip(torch.floor(x * L / lam + 0.5) / L, 0.0, 1.0)
        ctx.save_for_backward(q, lam)
        return lam * q

    @staticmethod
    def backward(ctx, grad):
        q, lam = ctx.saved_tensors
        # straight-through in x; lambda gradient from the outer scale
        return grad, (grad * q).sum().expand_as(lam), None


class QCFS(nn.Module):
    def __init__(self, L=8, lam_init=1.0):
        super().__init__()
        self.L = L
        # softplus(raw) = lam_init
        self.raw = nn.Parameter(torch.tensor(float(np.log(np.expm1(lam_init))), dtype=torch.float32))

    def lam(self):
        return F.softplus(self.raw) + 1e-3

    def forward(self, x):
        return QCFSFn.apply(x, self.lam(), self.L)


class QResBlock(nn.Module):
    def __init__(self, ch, q1, q2):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm1d(ch)
        self.q1 = q1
        self.c2 = nn.Conv1d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm1d(ch)
        self.q2 = q2

    def forward(self, x):
        y = self.q1(self.b1(self.c1(x)))
        y = self.q2(self.b2(self.c2(y)))
        return y


class QCFSCNN1D(nn.Module):
    def __init__(self, width=16, L=8, lam_init=1.0):
        super().__init__()
        self.L = L
        self.q = nn.ModuleList([QCFS(L, lam_init) for _ in range(6)])
        self.stem = nn.Sequential(nn.Conv1d(1, width, 7, padding=3, bias=False),
                                  nn.BatchNorm1d(width), self.q[0])
        self.res1 = QResBlock(width, self.q[1], self.q[2])
        self.down = nn.Sequential(nn.Conv1d(width, width * 2, 3, padding=1, stride=2, bias=False),
                                  nn.BatchNorm1d(width * 2), self.q[3])
        self.res2 = QResBlock(width * 2, self.q[4], self.q[5])
        self.fc = nn.Linear(width * 2, 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.res1(x)
        x = self.down(x)
        x = self.res2(x)
        x = F.adaptive_max_pool1d(x, 1).flatten(1)
        return self.fc(x)


def fold_bn(conv, bn):
    w = conv.weight.detach().float()
    gamma = bn.weight.detach().float()
    beta = bn.bias.detach().float()
    mean = bn.running_mean.float()
    var = bn.running_var.float()
    scale = gamma / torch.sqrt(var + bn.eps)
    return w * scale[:, None, None], beta - mean * scale


def qcfs_layers(model):
    return [
        (model.stem[0], model.stem[1], model.q[0], 1, 3),
        (model.res1.c1, model.res1.b1, model.res1.q1, 1, 1),
        (model.res1.c2, model.res1.b2, model.res1.q2, 1, 1),
        (model.down[0], model.down[1], model.q[3], 2, 1),
        (model.res2.c1, model.res2.b1, model.res2.q1, 1, 1),
        (model.res2.c2, model.res2.b2, model.res2.q2, 1, 1),
    ]


@torch.no_grad()
def evaluate_ann(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        pred = model(x.to(DEV)).argmax(1)
        correct += (pred == y.to(DEV)).sum().item()
        total += y.numel()
    return correct / total


@torch.no_grad()
def snn_features(x_uint8, layers, T, batch=512):
    feats = []
    for i in range(0, len(x_uint8), batch):
        xb = x_uint8[i:i + batch]
        sp = torch.from_numpy(encode(xb, T)).to(DEV)
        B = sp.shape[0]
        lens = [512, 512, 512, 256, 256, 256]
        vs = [torch.full((B, L["w"].shape[0], ln), 0.5, device=DEV)
              for L, ln in zip(layers, lens)]
        counts = torch.zeros(B, layers[-1]["w"].shape[0], lens[-1], device=DEV)
        for t in range(T):
            s = sp[:, t, :].unsqueeze(1).float()
            for li, L in enumerate(layers):
                cur = F.conv1d(s, L["w"], L["b"], stride=L["stride"], padding=L["pad"])
                v = vs[li] + cur
                fire = v >= 1.0
                vs[li] = torch.where(fire, v - 1.0, v)
                s = fire.float()
            counts += s
        feats.append(counts.max(dim=2).values.cpu())
    return torch.cat(feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="archive/mf_mlp_experiment/radar_mf.npz")
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--label-smooth", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--out", default="models/qcfs_spiking_cnn.pt")
    ap.add_argument("--metrics", default="docs/metrics_qcfs_spiking.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data = load_data(args.data)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*data["train"]), batch_size=args.batch,
        shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*data["val"]), batch_size=256, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*data["test"]), batch_size=256, shuffle=False)

    model = QCFSCNN1D(width=args.width, L=args.L).to(DEV)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"QCFS CNN width={args.width} L={args.L} params={n_params}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
    best_val, best_state = 0.0, None
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        run_loss = 0.0
        for x, y in train_loader:
            x = x.to(DEV); y = y.to(DEV)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run_loss += loss.item() * y.numel()
        sched.step()
        val = evaluate_ann(model, val_loader)
        if val > best_val:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"epoch {ep+1:3d}: loss={run_loss/len(train_loader.dataset):.4f} "
                  f"val={val*100:.2f}% lr={sched.get_last_lr()[0]:.2e} ({time.time()-t0:.1f}s)")

    model.load_state_dict(best_state)
    ann_acc = evaluate_ann(model, test_loader)
    print(f"QCFS ANN: val={best_val*100:.2f}% test={ann_acc*100:.2f}%")

    layers = []
    lams = []
    for i, (conv, bn, q, stride, pad) in enumerate(qcfs_layers(model)):
        w, b = fold_bn(conv, bn)
        lam = float(q.lam().detach().cpu())
        lams.append(lam)
        prev = 1.0 if i == 0 else lams[i - 1]
        w = w * (prev / lam)
        if i == 0:
            w = w * (256.0 / 255.0)
        b = b / lam
        layers.append(dict(w=w, b=b, stride=stride, pad=pad))
    print("QCFS lambdas:", [round(v, 4) for v in lams])

    d = np.load(args.data)
    xcal = d["x_train"][:4000]
    xt = d["x_test"]; yt = d["y_test"]; snr = d["snr_test"]
    ftest = snn_features(xt, layers, args.L)
    # last-layer activation scale: the ANN FC expects a_L = lam_L * q_L
    with torch.no_grad():
        logits = model.fc(ftest.to(DEV) * lams[-1]).cpu().numpy()
    pred = logits.argmax(1)
    tn = int(((pred == 0) & (yt == 0)).sum())
    fp = int(((pred == 1) & (yt == 0)).sum())
    fn = int(((pred == 0) & (yt == 1)).sum())
    tp = int(((pred == 1) & (yt == 1)).sum())
    acc = float((pred == yt).mean())
    pd = float(tp / (tp + fn)); pfa = float(fp / (fp + tn))
    auc = roc_auc(yt, logits[:, 1] - logits[:, 0])
    print(f"QCFS SNN T={args.L}: acc={acc*100:.2f}% PD={pd*100:.2f}% PFA={pfa*100:.2f}% AUC={auc:.4f}")
    print("confusion", tn, fp, fn, tp)
    bins = []
    for lo, hi in [(-8, -5), (-5, -2), (-2, 1), (1, 4)]:
        m = (snr >= lo) & (snr < hi)
        bins.append([lo, hi, int(m.sum()), float((pred[m] == yt[m]).mean())])
        print(f"  SNR [{lo},{hi}): n={m.sum()} acc={100*(pred[m]==yt[m]).mean():.2f}%")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(dict(state_dict=best_state, width=args.width, L=args.L,
                    n_params=n_params, ann_acc=ann_acc), args.out)
    out = dict(model="QCFS 1D CNN -> spiking conv SNN", params=n_params, T=args.L,
               qcfs_L=args.L, ann_accuracy=ann_acc, snn_accuracy=acc,
               PD=pd, PFA=pfa, auc=auc, confusion=dict(TN=tn, FP=fp, FN=fn, TP=tp),
               lambdas=lams, snr_bins=bins)
    json.dump(out, open(args.metrics, "w"), indent=2)
    print("saved", args.metrics)


if __name__ == "__main__":
    main()
