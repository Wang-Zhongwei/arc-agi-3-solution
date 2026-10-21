"""Candidate 4: TinyNet (engine-GT-trained hybrid CNN-ViT), CPU inference, union-find decode on 4-neighbour affinity.

setup(game) loads the fold whose held-out set contains the game (TINY_FOLD forces one; TINY_CNN=1 uses the CNN-only ablation).
TIMED = forward + decode wall time on CPU with TINY_THREADS threads (default 1).
"""
import hashlib, os, time
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from tiny_model import TinyNet
from train_tiny import FOLDS

HERE = Path(__file__).resolve().parent
torch.set_num_threads(int(os.environ.get("TINY_THREADS", "1")))
TIMED = None
_net = None; _tag = None
IDX = np.arange(4096).reshape(64, 64)


def setup(game):
    global _net, _tag
    fold = os.environ.get("TINY_FOLD") or next((f for f, gs in FOLDS.items() if game in gs), "F1")
    tag = fold + ("_cnn" if os.environ.get("TINY_CNN") == "1" else "")
    if tag != _tag:
        _net = TinyNet(transformer="_cnn" not in tag); _net.load_state_dict(torch.load(HERE / "weights" / f"{tag}.pt", map_location="cpu")); _net.eval(); _tag = tag


def canon(p):
    vs = []
    for m in (p, p[:, ::-1]):
        for r in range(4):
            q = np.rot90(m, r); vs.append(q.tobytes() + bytes(q.shape))
    return hashlib.blake2b(min(vs), digest_size=6).hexdigest()


@torch.no_grad()
def predict(fr):
    aff, sem, clk = _net(torch.as_tensor(fr, dtype=torch.long)[None])
    return torch.sigmoid(aff[0]).numpy(), sem[0].argmax(0).numpy(), torch.sigmoid(clk[0, 0]).numpy()


def decode(fr, aff, sem, thr=0.5):
    r = aff[0][:, :-1] > thr; d = aff[1][:-1, :] > thr
    src = np.concatenate([IDX[:, :-1][r], IDX[:-1, :][d]]); dst = np.concatenate([IDX[:, 1:][r], IDX[1:, :][d]])
    g = coo_matrix((np.ones(len(src)), (src, dst)), shape=(4096, 4096))
    n, lab = connected_components(g, directed=False)
    lab = lab.reshape(64, 64).astype(np.int64)
    # segments whose majority semantic class is background -> -1
    cnt = np.zeros((n, 3)); np.add.at(cnt, (lab.ravel(), sem.ravel()), 1)
    isbg = cnt.argmax(1) == 0
    lab[isbg[lab]] = -1
    return lab


def segment(frame, prev=None, state=None):
    global TIMED
    fr = np.asarray(frame, dtype=np.int64)
    if state is None:
        state = {"names": {}}
    t0 = time.perf_counter()
    aff, sem, clk = predict(fr)
    lab = decode(fr, aff, sem)
    TIMED = time.perf_counter() - t0
    names = {}
    for j in np.unique(lab):
        if j < 0:
            continue
        mm = lab == j; ys, xs = np.nonzero(mm)
        p = np.where(mm, fr, -1)[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        names[int(j)] = state["names"].setdefault(canon(p), f"o{len(state['names'])}")
    state["clk"] = clk
    return lab, names, {}, state
