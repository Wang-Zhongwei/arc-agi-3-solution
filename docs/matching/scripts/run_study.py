"""Reproduce the controlled assignment study using only the standard library."""
import csv
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = ROOT / 'snapshots'
DATA = ROOT / 'data'
sys.path.insert(0, str(SNAPSHOTS))
from hungarian_reference import _hungarian  # noqa: E402
from segmentation_snapshot import segment_layer  # noqa: E402
SEED = 20261003
OPS = ('move', 'rotate', 'mirror', 'scale', 'recolor', 'reshape')
SHAPES = (
    ((1, 0, 0), (1, 1, 1)),
    ((1, 1, 0), (0, 1, 0), (0, 1, 1)),
    ((1, 1, 1), (0, 1, 0)),
    ((1, 1, 1), (1, 0, 1), (1, 1, 1)),
)


def hungarian_candidates(candidates):
    if not candidates:
        return []
    rows = sorted({i for _, i, _, _ in candidates})
    cols = sorted({j for _, _, j, _ in candidates})
    edges = {(i, j): (rank, change) for rank, i, j, change in candidates}
    # Maximize valid cardinality, then minimize total center distance. Dummies
    # represent unmatched rows; forbidden edges always cost more than dummies.
    penalty = (len(rows) + 1) * (max(rank for rank, *_ in candidates) + 1)
    cost = [[edges.get((i, j), (2*penalty, None))[0] for j in cols]
            + [penalty]*len(rows) for i in rows]
    assignment = _hungarian(cost)
    return [(rows[r], cols[c], edges[rows[r], cols[c]][1])
            for r, c in enumerate(assignment) if c < len(cols) and (rows[r], cols[c]) in edges]


def engines():
    source = (SNAPSHOTS / 'frame_diff_snapshot.py').read_text()
    needle = 'accepted = _unique_nearest(candidates)'
    assert source.count(needle) == 1  # Exact stage only; resize policy stays fixed.
    result = {}
    for name in ('mutual', 'hungarian'):
        ns = {'_hungarian_candidates': hungarian_candidates}
        text = source if name == 'mutual' else source.replace(needle, 'accepted = _hungarian_candidates(candidates)')
        exec(compile(text, str(SNAPSHOTS / 'frame_diff_snapshot.py'), 'exec'), ns)
        result[name] = ns['compute_frame_diff']
    return result


def rotate(mask):
    return tuple(tuple(row) for row in zip(*mask[::-1]))


def reflect(mask, axis):
    if axis == '-':
        return mask[::-1]
    if axis == '|':
        return tuple(row[::-1] for row in mask)
    transposed = tuple(tuple(row) for row in zip(*mask))
    return tuple(row[::-1] for row in transposed[::-1]) if axis == '/' else transposed


def enlarge(mask, k):
    return tuple(tuple(v for v in row for _ in range(k)) for row in mask for _ in range(k))


def canonical(mask):
    variants = []
    for m in (mask, reflect(mask, '|')):
        for _ in range(4):
            variants.append((len(m), len(m[0]), tuple(itertools.chain.from_iterable(m))))
            m = rotate(m)
    return min(variants)


def reduce_mask(mask):
    h, w = len(mask), len(mask[0])
    for k in range(min(h, w), 0, -1):
        if h % k or w % k:
            continue
        if all(len({mask[r+i][c+j] for i in range(k) for j in range(k)}) == 1
               for r in range(0, h, k) for c in range(0, w, k)):
            return k, tuple(tuple(mask[r][c] for c in range(0, w, k)) for r in range(0, h, k))
    raise AssertionError('unit reduction always exists')


def obj(identity, mask, color=1, center=(22, 22), at=None):
    at = at or (center[0]-len(mask)//2, center[1]-len(mask[0])//2)
    return {'identity': identity, 'mask': mask, 'color': color, 'at': at}


def bbox(o):
    r, c = o['at']
    return r, c, r+len(o['mask'])-1, c+len(o['mask'][0])-1


def pixels(o):
    return {(r+o['at'][0], c+o['at'][1]) for r, row in enumerate(o['mask']) for c, v in enumerate(row) if v}


def changed(a, b):
    return a['color'] != b['color'] or pixels(a) != pixels(b)


def oracle(a, b):
    """Independent dense-matrix oracle for the documented single-event contract."""
    ma, mb = a['mask'], b['mask']
    ba, bb = bbox(a), bbox(b)
    same_color = a['color'] == b['color']
    if same_color and ma == mb:
        return 'moved' if ba != bb else None
    near = all(abs(ba[d]+ba[d+2]-bb[d]-bb[d+2]) <= 2 for d in (0, 1))
    if same_color and near:
        m = ma
        for _ in range(3):
            m = rotate(m)
            if m == mb:
                return 'rotated'
        if any(reflect(ma, axis) == mb for axis in ('-', '|', '/', '\\')):
            return 'mirrored'
    ka, ra = reduce_mask(ma)
    kb, rb = reduce_mask(mb)
    if same_color and near and ka != kb and ra == rb:
        return 'scaled'
    if not same_color and ma == mb and ba == bb:
        return 'changed_color'
    overlap = max(0, min(ba[2], bb[2])-max(ba[0], bb[0])+1)*max(0, min(ba[3], bb[3])-max(ba[1], bb[1])+1)
    area_a, area_b = len(ma)*len(ma[0]), len(mb)*len(mb[0])
    if same_color and canonical(ra) != canonical(rb) and overlap/min(area_a, area_b) >= 0.5:
        return 'resized'
    return None


def render(objects):
    grid = [[0]*48 for _ in range(48)]
    owners = {}
    for o in objects:
        for r, c in pixels(o):
            assert 0 <= r < 48 and 0 <= c < 48
            assert (r, c) not in owners
            grid[r][c] = o['color']
            owners[r, c] = o['identity']
    nodes = segment_layer(grid, '0123456789ABCDEF')['nodes']
    ids = {}
    for n in nodes:
        if n['color'] == '0':
            continue
        owner = owners[tuple(n['boundary'][0])]
        assert owner not in ids, 'object split by segmentation'
        o = next(o for o in objects if o['identity'] == owner)
        assert n['pixels'] == len(pixels(o)) and n['bbox'] == bbox(o), 'objects merged by segmentation'
        ids[owner] = n['id']
    assert len(ids) == len(objects)
    return grid, nodes, ids


def scenes():
    rng = random.Random(SEED)
    for count in range(1, len(OPS)+1):
        for ops in itertools.combinations(OPS, count):
            for rep in range(8):
                primitive = SHAPES[rep % len(SHAPES)]
                for _ in range(rng.randrange(4)):
                    primitive = rotate(primitive)
                ka, kb = ((1, 2), (2, 1), (2, 3), (3, 2))[rep % 4]
                before = enlarge(primitive, ka)
                after = primitive
                if 'reshape' in ops:
                    row = next(r for r in range(len(after)) if after[r][-1])
                    after = tuple(v + (int(r == row),) for r, v in enumerate(after))
                if 'mirror' in ops:
                    after = reflect(after, ('-', '|', '/', '\\')[rep % 4])
                if 'rotate' in ops:
                    for _ in range(1+rep % 3):
                        after = rotate(after)
                after = enlarge(after, kb if 'scale' in ops else ka)
                a = obj('target', before)
                b = obj('target', after, color=2 if 'recolor' in ops else 1,
                        center=(22, 29) if 'move' in ops else (22, 22))
                distractors = [obj('static0', before, center=(6, 6)),
                               obj('static1', SHAPES[(rep+1) % 4], color=3, center=(40, 6))]
                yield {'name': '+'.join(ops)+f'/{rep}', 'family': '+'.join(ops),
                       'suite': 'single' if count == 1 else 'combination',
                       'operations': ops, 'before': [a, *distractors], 'after': [b, *distractors]}
    for shape_index, mask in enumerate(SHAPES):
        variations = []
        for axis in ('-', '|', '/', '\\'):
            variations.append(('mirror_axis_'+axis, ('mirror',), mask, reflect(mask, axis)))
        for angle in (90, 180, 270):
            m = mask
            for _ in range(angle//90):
                m = rotate(m)
            variations.append(('rotation_'+str(angle), ('rotate',), mask, m))
        for ka, kb in ((1, 2), (2, 1), (2, 3), (3, 2)):
            variations.append((f'scale_{ka}_to_{kb}', ('scale',), enlarge(mask, ka), enlarge(mask, kb)))
        for family, ops, before, after in variations:
            yield {'name': family+f'/{shape_index}', 'family': family, 'suite': 'parameters',
                   'operations': ops, 'before': [obj('target', before)], 'after': [obj('target', after)]}
    for kind, after in (('rotate', rotate(SHAPES[0])), ('mirror', reflect(SHAPES[0], '|')),
                        ('scale', enlarge(SHAPES[0], 2))):
        for shift in (0, 1, 2, 3):
            yield {'name': f'center_tolerance_{kind}/{shift}', 'family': 'center_tolerance_'+kind,
                   'suite': 'parameters', 'operations': (kind, 'move') if shift else (kind,),
                   'before': [obj('target', SHAPES[0])],
                   'after': [obj('target', after, center=(22, 22+shift))]}
    for rep in range(4):
        mask = SHAPES[0]
        for _ in range(rep):
            mask = rotate(mask)
        decoys = (
            ('moved', mask, 1, (22, 29)),
            ('rotated', rotate(mask), 1, (22, 22)),
            ('mirrored', reflect(mask, '|'), 1, (22, 22)),
            ('scaled', enlarge(mask, 2), 1, (22, 22)),
            ('changed_color', mask, 2, (22, 22)),
            ('resized', SHAPES[2], 1, (22, 22)),
            ('different_shape_far_away', SHAPES[2], 1, (22, 35)),
        )
        for bucket, after, color, center in decoys:
            yield {'name': f'unrelated_{bucket}/{rep}', 'family': 'unrelated_'+bucket,
                   'suite': 'negative_controls', 'operations': (),
                   'before': [obj('old', mask)],
                   'after': [obj('new', after, color=color, center=center)]}
    # Identical point sprites isolate assignment from shape recognition. These
    # include observationally indistinguishable identity histories deliberately.
    layouts = {
        'local_competition_ordered': ([('A', 8), ('B', 12)], [('A', 11), ('B', 16)]),
        'local_competition_crossed': ([('A', 8), ('B', 12)], [('A', 16), ('B', 11)]),
        'local_tie_global_unique': ([('A', 8), ('B', 12)], [('A', 10), ('B', 16)]),
        'global_tie': ([('A', 8), ('B', 12)], [('A', 16), ('B', 20)]),
        'crossing': ([('A', 8), ('B', 16)], [('A', 12), ('B', 10)]),
        'stationary_duplicate': ([('A', 8), ('B', 12)], [('A', 8), ('B', 16)]),
        'identical_replacement': ([('A', 8)], [('B', 12)]),
        'survivor_plus_death': ([('A', 8), ('B', 12)], [('B', 16)]),
        'survivor_plus_birth': ([('A', 8)], [('A', 12), ('B', 16)]),
        'all_unchanged': ([('A', 8), ('B', 12)], [('A', 8), ('B', 12)]),
        'birth_only': ([], [('B', 12)]),
        'death_only': ([('A', 8)], []),
    }
    for family, (before, after) in layouts.items():
        for rep in range(8):
            def objects(entries):
                result = []
                for name, col in entries:
                    col = 32-col if rep % 2 else col
                    at = (col, 20+rep//4) if rep % 4 >= 2 else (20+rep//4, col)
                    result.append(obj(name, ((1,),), at=at, color=1+rep//4))
                return result
            yield {'name': family+f'/{rep}', 'family': family, 'suite': 'assignment',
                   'operations': (), 'before': objects(before), 'after': objects(after)}


def metrics(scene, result, ids_a, ids_b):
    inverse_a, inverse_b = {v: k for k, v in ids_a.items()}, {v: k for k, v in ids_b.items()}
    predicted = {}
    background_pairs = 0
    for bucket, entries in result.items():
        if not isinstance(entries, list) or bucket in ('appeared', 'disappeared'):
            continue
        for entry in entries:
            a, b = inverse_a.get(entry['before_id']), inverse_b.get(entry['after_id'])
            if a is None or b is None:
                background_pairs += 1
            else:
                assert (a, b) not in predicted
                predicted[a, b] = bucket
    counts = dict(TP=0, FN=0, FP=0, TN=0)
    supported = dict(TP=0, FN=0, FP=0, TN=0)
    truth, expected = set(), {}
    wrong_bucket, supported_total, unsupported_total, aliases = 0, 0, 0, 0
    for a in scene['before']:
        for b in scene['after']:
            pair = a['identity'], b['identity']
            positive = pair[0] == pair[1] and changed(a, b)
            expected_bucket = oracle(a, b) if positive else None
            captured = pair in predicted
            counts[('TP' if captured else 'FN') if positive else ('FP' if captured else 'TN')] += 1
            if positive:
                truth.add(pair)
                expected[pair] = expected_bucket
                if expected_bucket is not None:
                    supported_total += 1
                else:
                    unsupported_total += 1
                if captured and predicted[pair] != expected_bucket:
                    wrong_bucket += 1
                if len(scene['operations']) > 1 and expected_bucket is not None:
                    aliases += 1
            # Contract matrix excludes out-of-contract changed true pairs,
            # retaining all negative pairs and supported positive pairs.
            if not positive or expected_bucket is not None:
                supported[('TP' if captured else 'FN') if positive else ('FP' if captured else 'TN')] += 1
    disappeared = {inverse_a[e['id']] for e in result['disappeared'] if e['id'] in inverse_a}
    appeared = {inverse_b[e['id']] for e in result['appeared'] if e['id'] in inverse_b}
    return {**counts, 'contract': supported, 'supported_positives': supported_total,
            'unsupported_positives': unsupported_total, 'combination_aliases': aliases,
            'wrong_bucket': wrong_bucket, 'background_pairs_excluded': background_pairs,
            'predicted': [{'before': a, 'after': b, 'bucket': bucket} for (a, b), bucket in sorted(predicted.items())],
            'missed': [{'before': a, 'after': b, 'expected_bucket': expected[a, b]} for a, b in sorted(truth-predicted.keys())],
            'false_pairs': [{'before': a, 'after': b, 'bucket': predicted[a, b]} for a, b in sorted(predicted.keys()-truth)],
            'true_births_captured': len((set(ids_b)-set(ids_a)) & appeared),
            'true_deaths_captured': len((set(ids_a)-set(ids_b)) & disappeared)}


def totals(records, contract=False):
    result = {k: sum((r['contract'] if contract else r)[k] for r in records) for k in ('TP','FN','FP','TN')}
    tp, fn, fp = result['TP'], result['FN'], result['FP']
    result['precision'] = tp/(tp+fp) if tp+fp else None
    result['recall'] = tp/(tp+fn) if tp+fn else None
    return result


def self_test():
    # Independently check the adapter against brute-force maximum-cardinality,
    # minimum-cost matchings, including incomplete graphs and distance ties.
    rng = random.Random(7)
    for _ in range(100):
        n, m = rng.randint(1, 4), rng.randint(1, 4)
        edges = {(i, j): rng.randrange(8) for i in range(n) for j in range(m) if rng.random() < .65}
        cs = [(d, i, j, {}) for (i, j), d in edges.items()]
        got = hungarian_candidates(cs)
        scores = []
        for choices in itertools.product(range(-1, m), repeat=n):
            pairs = [(i, j) for i, j in enumerate(choices) if j >= 0]
            if len({j for _, j in pairs}) != len(pairs) or any(p not in edges for p in pairs):
                continue
            scores.append((-len(pairs), sum(edges[p] for p in pairs)))
        assert (-len(got), sum(edges[i,j] for i,j,_ in got)) == min(scores)
    mask = SHAPES[0]
    assert oracle(obj('A', mask), obj('A', rotate(mask))) == 'rotated'
    assert oracle(obj('A', mask), obj('A', reflect(mask, '|'))) == 'mirrored'
    assert oracle(obj('A', mask), obj('A', enlarge(mask, 2))) == 'scaled'
    assert oracle(obj('A', mask), obj('A', mask, color=2)) == 'changed_color'
    assert oracle(obj('A', mask), obj('A', rotate(mask), color=2)) is None


def main(check_production=False):
    self_test()
    methods = engines()
    production_source = None
    if check_production:
        production_path = ROOT.parents[1] / 'ARC3-Inference/inference/utils/frame_diff.py'
        production_source = production_path.read_text()
        production = {}
        exec(compile(production_source, str(production_path), 'exec'), production)
    results = []
    for index, scene in enumerate(scenes()):
        ga, na, ia = render(scene['before'])
        gb, nb, ib = render(scene['after'])
        measurements = {}
        for method, engine in methods.items():
            diff = engine(ga, gb, na, nb)  # Manual API: no group limit.
            measurements[method] = metrics(scene, diff, ia, ib)
            if check_production and method == 'hungarian':
                actual = production['compute_frame_diff'](ga, gb, na, nb)
                assert actual == diff, scene['name']
        results.append({**scene, 'measurements': measurements})
        if (index+1) % 100 == 0:
            print(f'{index+1} scenes complete', flush=True)
    with (DATA/'cases.jsonl').open('w') as f:
        for record in results:
            f.write(json.dumps(record, separators=(',', ':'))+'\n')
    by_family = {}
    for r in results:
        by_family.setdefault(r['family'], []).append(r)
    summary = {'seed': SEED, 'scenes': len(results), 'sha256': {
        name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ('scripts/run_study.py','snapshots/hungarian_reference.py','snapshots/frame_diff_snapshot.py','snapshots/segmentation_snapshot.py')},
        'overall': {}, 'suites': {}, 'families': {}}
    for method in methods:
        ms = [r['measurements'][method] for r in results]
        summary['overall'][method] = {'identity': totals(ms), 'contract': totals(ms, True),
                                     'wrong_bucket': sum(r['wrong_bucket'] for r in ms)}
    for suite in ('single','combination','parameters','negative_controls','assignment'):
        summary['suites'][suite] = {method: totals([r['measurements'][method] for r in results if r['suite']==suite])
                                     for method in methods}
    for family, rows in by_family.items():
        summary['families'][family] = {'scenes': len(rows), 'suite': rows[0]['suite'],
            'supported_positives': sum(r['measurements']['mutual']['supported_positives'] for r in rows),
            'unsupported_positives': sum(r['measurements']['mutual']['unsupported_positives'] for r in rows),
            **{method: totals([r['measurements'][method] for r in rows]) for method in methods}}
    if check_production:
        summary['production_replay'] = {
            'scenes': len(results),
            'matched_method': 'hungarian',
            'frame_diff_sha256': hashlib.sha256(production_source.encode()).hexdigest(),
        }
        print(f'Production Hungarian matches the frozen study on all {len(results)} scenes.')
    (DATA/'results.json').write_text(json.dumps(summary, indent=2)+'\n')
    with (DATA/'per_family.csv').open('w') as f:
        writer=csv.writer(f)
        writer.writerow(['family','suite','scenes','supported_positives','unsupported_positives','method','TP','FN','FP','TN','precision','recall'])
        for family, row in summary['families'].items():
            for method in methods:
                writer.writerow([family,row['suite'],row['scenes'],row['supported_positives'],row['unsupported_positives'],method,
                                 *[row[method][k] for k in ('TP','FN','FP','TN','precision','recall')]])
    print(json.dumps(summary['overall'], indent=2))


if __name__ == '__main__':
    if '--self-test' in sys.argv:
        self_test()
        print('100 brute-force assignment checks and oracle smoke checks passed')
    else:
        main(check_production='--check-production' in sys.argv)
