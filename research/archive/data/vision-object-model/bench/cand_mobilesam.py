"""Candidate 3: MobileSAM (Apache-2.0, TinyViT-5M encoder + SAM decoder), zero-shot, point-prompted, PyTorch CPU.

Frame -> ARC palette RGB -> nearest 16x upscale to 1024x1024. One positive point per colour CC of >=2 px
(at its most interior pixel), capped at the 64 largest, decoded in one batch; best of 3 multimask outputs.
Masks majority-voted to 64x64 per 16x16 block, deduped at IoU>0.8, painted smallest-first into a label map.
TIMED = encoder + decoder wall time. Threads: SAM_THREADS (default 1).
"""
import hashlib, os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from scipy import ndimage
from mobile_sam import sam_model_registry, SamPredictor

PAL = np.array([(255, 255, 255), (204, 204, 204), (153, 153, 153), (102, 102, 102), (51, 51, 51), (0, 0, 0),
                (229, 58, 163), (255, 123, 204), (249, 60, 49), (30, 147, 255), (136, 216, 241), (255, 220, 0),
                (255, 133, 27), (146, 18, 49), (79, 204, 48), (163, 86, 214)], dtype=np.uint8)
W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sam_weights", "mobile_sam.pt")
torch.set_num_threads(int(os.environ.get("SAM_THREADS", "1")))
_model = sam_model_registry["vit_t"](checkpoint=W).eval()
_pred = SamPredictor(_model)
CROSS = ndimage.generate_binary_structure(2, 1)
TIMED = None
TIMES = {"enc": [], "dec": []}


def prompts(fr, cap=64):
    pts = []
    for c in np.unique(fr):
        l, n = ndimage.label(fr == c, structure=CROSS)
        if n == 0:
            continue
        sizes = np.bincount(l.ravel())
        for i in range(1, n + 1):
            if sizes[i] < 2:
                continue
            m = l == i
            d = ndimage.distance_transform_cdt(np.pad(m, 1), metric="taxicab")[1:-1, 1:-1]
            y, x = np.unravel_index(np.argmax(d), d.shape)
            pts.append((int(sizes[i]), int(x), int(y)))
    pts.sort(reverse=True)
    return [(x, y) for _, x, y in pts[:cap]]


def canon(p):
    vs = []
    for m in (p, p[:, ::-1]):
        for r in range(4):
            q = np.rot90(m, r); vs.append(q.tobytes() + bytes(q.shape))
    return hashlib.blake2b(min(vs), digest_size=6).hexdigest()


@torch.no_grad()
def segment(frame, prev=None, state=None):
    global TIMED
    fr = np.asarray(frame, dtype=np.int64)
    if state is None:
        state = {"names": {}}
    img = np.repeat(np.repeat(PAL[fr], 16, 0), 16, 1)
    pts = prompts(fr)
    t0 = time.perf_counter()
    _pred.set_image(img)
    t1 = time.perf_counter()
    if pts:
        coords = torch.tensor([[[x * 16 + 8, y * 16 + 8]] for x, y in pts], dtype=torch.float32)
        coords = _pred.transform.apply_coords_torch(coords, (1024, 1024))
        labels = torch.ones(coords.shape[:2], dtype=torch.int64)
        sp, de = _model.prompt_encoder(points=(coords, labels), boxes=None, masks=None)
        low, iou = _model.mask_decoder(image_embeddings=_pred.features, image_pe=_model.prompt_encoder.get_dense_pe(),
                                       sparse_prompt_embeddings=sp, dense_prompt_embeddings=de, multimask_output=True)
        best = iou.argmax(1)
        m = (low[torch.arange(len(pts)), best] > 0).numpy()  # (P,256,256) bool, 4 px per frame cell
    else:
        m = np.zeros((0, 256, 256), bool)
    t2 = time.perf_counter()
    TIMED = t2 - t0
    TIMES["enc"].append(t1 - t0); TIMES["dec"].append(t2 - t1)
    small = m.reshape(len(m), 64, 4, 64, 4).mean((2, 4)) > 0.5
    keep = []
    for s in sorted(range(len(small)), key=lambda i: small[i].sum()):
        a = small[s]
        if a.sum() == 0:
            continue
        if any((a & small[k]).sum() / (a | small[k]).sum() > 0.8 for k in keep):
            continue
        keep.append(s)
    lab = np.full(fr.shape, -1, np.int64)
    for j, s in enumerate(keep):  # smallest first claims pixels
        lab[(lab < 0) & small[s]] = j
    names = {}
    for j in np.unique(lab):
        if j < 0:
            continue
        mm = lab == j
        ys, xs = np.nonzero(mm)
        p = np.where(mm, fr, -1)[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        names[int(j)] = state["names"].setdefault(canon(p), f"o{len(state['names'])}")
    return lab, names, {}, state
