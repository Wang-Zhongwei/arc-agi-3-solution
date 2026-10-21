"""Train TinyNet on engine ground truth for one leave-5-games-out fold.

Data: bench/full/<game>.npz (every replayed step; deduplicated frames) of the 20 training games,
plus synthetic frames built only from training-game sprite crops (random backgrounds, forced contacts).
Augment: random palette permutation (p=0.5), 8 dihedral transforms.
Usage: python train_tiny.py --fold F1 [--steps 3000] [--cnn]
"""
import argparse, json, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tiny_model import TinyNet, OFFS

HERE = Path(__file__).resolve().parent
FULL = HERE / "full"
FOLDS = {"F1": "ls20,cd82,bp35,sp80,vc33".split(","), "F2": "ar25,tu93,lf52,cn04,r11l".split(",")}
ALL = sorted(p.stem for p in FULL.glob("*.npz"))


def load(games):
    frs, labs, sems, clks, crops = [], [], [], [], []
    for g in games:
        z = np.load(FULL / f"{g}.npz"); meta = json.load(open(FULL / f"{g}.meta.json"))
        seen = set()
        for i in range(len(z["frames"])):
            fr = z["frames"][i]; key = fr.tobytes()
            if key in seen:
                continue
            seen.add(key)
            inst = z["inst"][i].astype(np.int64); ui = z["ui"][i]
            lab = np.where(ui >= 0, -2 - ui.astype(np.int64), inst)  # -1 bg, <=-2 HUD interface, >=0 sprite
            sem = np.where(ui >= 0, 2, np.where(inst >= 0, 1, 0))
            click = {o[0] for o in meta[i]["objects"] if o[3]}
            clk = np.isin(inst, list(click)) & (ui < 0) if click else np.zeros_like(inst, bool)
            frs.append(fr); labs.append(lab); sems.append(sem); clks.append(clk)
            # sprite crops for synthesis (fully visible-ish, small)
            if len(crops) < 4000 and random.random() < 0.3:
                for k in np.unique(inst[ui < 0]):
                    if k < 0: continue
                    m = (inst == k) & (ui < 0)
                    ys, xs = np.nonzero(m)
                    h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
                    if 2 <= m.sum() <= 400 and h <= 24 and w <= 24:
                        sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
                        crops.append(np.where(m[sl], fr[sl], -1).astype(np.int64))
    return np.stack(frs), np.stack(labs), np.stack(sems), np.stack(clks), crops


def synth(crops, rng):
    bgc = rng.integers(16)
    fr = np.full((64, 64), bgc, np.int64); lab = np.full((64, 64), -1, np.int64)
    n = rng.integers(3, 14); k = 0
    for _ in range(n):
        c = crops[rng.integers(len(crops))]
        s = rng.integers(1, 4) if c.shape[0] * 3 <= 40 and c.shape[1] * 3 <= 40 else 1
        if s > 1:
            c = np.repeat(np.repeat(c, s, 0), s, 1)
        h, w = c.shape
        if rng.random() < 0.5 and k > 0:  # force contact: place next to an existing object
            ys, xs = np.nonzero(lab >= 0); j = rng.integers(len(ys)); y, x = ys[j] - rng.integers(h), xs[j] + 1
        else:
            y, x = rng.integers(0, 64 - h + 1), rng.integers(0, 64 - w + 1)
        y = int(np.clip(y, 0, 64 - h)); x = int(np.clip(x, 0, 64 - w))
        m = c >= 0
        fr[y:y + h, x:x + w][m] = c[m]; lab[y:y + h, x:x + w][m] = k; k += 1
        if rng.random() < 0.3:  # tile it
            for t in range(1, rng.integers(2, 5)):
                if x + (t + 1) * w <= 64:
                    fr[y:y + h, x + t * w:x + (t + 1) * w][m] = c[m]; lab[y:y + h, x + t * w:x + (t + 1) * w][m] = k; k += 1
    sem = np.where(lab >= 0, 1, 0)
    return fr, lab, sem, np.zeros((64, 64), bool)


def targets(fr, lab):
    """Affinity targets + weights for OFFS; (B,6,64,64) each."""
    T, Wt, V = [], [], []
    for dy, dx in OFFS:
        l2 = torch.roll(lab, shifts=(-dy, -dx), dims=(1, 2)); f2 = torch.roll(fr, shifts=(-dy, -dx), dims=(1, 2))
        valid = torch.ones_like(lab, dtype=torch.bool)
        if dx: valid[:, :, 64 - dx:] = False
        if dy: valid[:, 64 - dy:, :] = False
        t = (l2 == lab).float()
        w = 1 + 4 * ((t == 0) & (f2 == fr)).float()
        T.append(t); Wt.append(w * valid); V.append(valid)
    return torch.stack(T, 1), torch.stack(Wt, 1)


def d4(x, k, flip):
    x = torch.rot90(x, k, dims=(-2, -1))
    return torch.flip(x, dims=(-1,)) if flip else x


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", required=True); ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--cnn", action="store_true"); ap.add_argument("--bs", type=int, default=32)
    a = ap.parse_args()
    random.seed(0); torch.manual_seed(0); rng = np.random.default_rng(0)
    train_games = [g for g in ALL if g not in FOLDS[a.fold]]
    t0 = time.time()
    FR, LAB, SEM, CLK, crops = load(train_games)
    print(f"fold {a.fold}: {len(FR)} unique train frames from {len(train_games)} games, {len(crops)} crops, load {time.time()-t0:.0f}s", flush=True)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    net = TinyNet(transformer=not a.cnn).to(dev)
    print("params", sum(p.numel() for p in net.parameters()), flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=a.steps, pct_start=0.1)
    for step in range(a.steps):
        fr, lab, sem, clk = [], [], [], []
        for _ in range(a.bs):
            if rng.random() < 0.3 and crops:
                f, l, s, c = synth(crops, rng)
            else:
                j = rng.integers(len(FR)); f, l, s, c = FR[j].astype(np.int64), LAB[j], SEM[j], CLK[j]
            if rng.random() < 0.5:
                perm = rng.permutation(16); f = perm[f]
            fr.append(f); lab.append(l); sem.append(s); clk.append(c)
        fr = torch.as_tensor(np.stack(fr)); lab = torch.as_tensor(np.stack(lab)); sem = torch.as_tensor(np.stack(sem)); clk = torch.as_tensor(np.stack(clk)).float()
        k, fl = int(rng.integers(4)), bool(rng.integers(2))
        fr, lab, sem, clk = d4(fr, k, fl), d4(lab, k, fl), d4(sem, k, fl), d4(clk, k, fl)
        T, Wt = targets(fr, lab)
        fr, T, Wt, sem, clk = fr.to(dev), T.to(dev), Wt.to(dev), sem.to(dev), clk.to(dev)
        aff, sl, cl = net(fr)
        la = (F.binary_cross_entropy_with_logits(aff, T, reduction="none") * Wt).sum() / Wt.sum()
        ls = F.cross_entropy(sl, sem)
        lc = F.binary_cross_entropy_with_logits(cl[:, 0], clk)
        loss = la + 0.5 * ls + 0.5 * lc
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 250 == 0 or step == a.steps - 1:
            print(f"step {step} loss {loss.item():.4f} aff {la.item():.4f} sem {ls.item():.4f} clk {lc.item():.4f} t={time.time()-t0:.0f}s", flush=True)
    tag = a.fold + ("_cnn" if a.cnn else "")
    torch.save(net.state_dict(), HERE / "weights" / f"{tag}.pt")
    print("saved", tag, f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
