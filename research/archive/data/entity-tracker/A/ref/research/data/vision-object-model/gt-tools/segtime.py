import sys, json, lzma, glob, os, time, statistics as st
sys.path.insert(0, sys.argv[1])
from inference.utils.segmentation import segment_layer
from inference.utils.frame_diff import compute_frame_diff
from inference.utils.grid_utils import ARC_COLOR_CHARS
run = sys.argv[2]
boards = []
for f in sorted(glob.glob(os.path.join(run, 'artifacts', '*_events.jsonl.xz'))):
    k = 0
    for line in lzma.open(f, 'rt'):
        r = json.loads(line)
        b = r.get('board')
        if isinstance(b, str): b = json.loads(b)
        if b and r.get('type') == 'action':
            boards.append(b); k += 1
            if k >= 6: break
seg_t, nodes, diff_t = [], [], []
for b in boards:
    t = time.perf_counter(); s = segment_layer(b, ARC_COLOR_CHARS); seg_t.append(time.perf_counter() - t); nodes.append(len(s['nodes']))
for a, b in zip(boards, boards[1:]):
    if len(a) != len(b): continue
    sa = segment_layer(a, ARC_COLOR_CHARS)['nodes']; sb = segment_layer(b, ARC_COLOR_CHARS)['nodes']
    t = time.perf_counter(); compute_frame_diff(a, b, sa, sb, max_group_match=40); diff_t.append(time.perf_counter() - t)
print('frames', len(boards), 'segment_layer ms median %.1f max %.1f' % (1000*st.median(seg_t), 1000*max(seg_t)))
print('nodes per frame median %d max %d' % (st.median(nodes), max(nodes)))
print('frame_diff ms median %.1f max %.1f' % (1000*st.median(diff_t), 1000*max(diff_t)))
