"""Candidate 1: the harness's own segment_layer (4-connected single-colour components)."""
import sys, time
import numpy as np
sys.path.insert(0, "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference")
from inference.utils.segmentation import segment_layer
from inference.utils.grid_utils import ARC_COLOR_CHARS
from scipy import ndimage

TIMED = None  # seconds of the deploy-relevant call (segment_layer), read by eval.py


def cc_labels(fr):
    """Reading-order 4-connected same-colour labels (same ids as segment_layer)."""
    lab = -np.ones(fr.shape, dtype=np.int64)
    n = 0
    tmp = np.zeros(fr.shape, dtype=np.int64); off = 0
    for c in np.unique(fr):
        l, k = ndimage.label(fr == c)
        tmp[l > 0] = l[l > 0] + off; off += k
    # renumber by first occurrence in reading order
    flat = tmp.ravel(); _, first = np.unique(flat, return_index=True)
    order = np.argsort(first)
    remap = np.empty(order.size, dtype=np.int64); remap[order] = np.arange(order.size)
    u = np.unique(flat)
    idx = np.searchsorted(u, flat)
    return remap[idx].reshape(fr.shape)


def segment(frame, prev=None, state=None):
    global TIMED
    board = frame.tolist()
    t0 = time.perf_counter()
    seg = segment_layer(board, ARC_COLOR_CHARS)  # the timed harness call
    TIMED = time.perf_counter() - t0
    lab = cc_labels(frame)
    names = {n["id"]: n["hash"] for n in seg["nodes"]}
    return lab, names, {}, state
