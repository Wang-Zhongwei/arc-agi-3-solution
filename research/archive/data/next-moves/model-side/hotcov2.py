"""Out-of-sample: build a map from run A's generated tokens (union with the current map's head), test coverage on run B."""
import json, lzma, glob, pickle, hashlib, sys, collections
from tokenizers import Tokenizer
tok = Tokenizer.from_file(sys.argv[1]); hot = pickle.load(open(sys.argv[2], 'rb')); A, B = sys.argv[3], sys.argv[4]
def counts(run):
    seen = set(); cnt = collections.Counter()
    for p in sorted(glob.glob(run + '/*_requests.jsonl.xz')):
        with lzma.open(p, 'rt') as f:
            for line in f:
                r = json.loads(line)
                if r.get('event') != 'request': continue
                for m in r.get('messages') or []:
                    if m.get('role') != 'assistant': continue
                    texts = [m.get('reasoning_content') or ''] + [tc.get('function', {}).get('arguments', '') for tc in m.get('tool_calls') or []]
                    for t in texts:
                        if not t: continue
                        h = hashlib.md5(t.encode()).digest()
                        if h in seen: continue
                        seen.add(h); cnt.update(tok.encode(t, add_special_tokens=False).ids)
    return cnt
ca, cb = counts(A), counts(B)
tb = sum(cb.values())
def cov(S): return 100 * sum(c for i, c in cb.items() if i in S) / tb
hs = set(hot)
print(f'run B tokens {tb:,}; current 64k map: {cov(hs):.3f}%')
for k in (16384, 32768):
    own = {i for i, _ in ca.most_common(k)}
    print(f'map = run A top-{k}: {cov(own):.3f}%;  current map with its {len(hs) - 0} entries topped up by run A tokens it lacks: {cov(hs | own):.3f}%')
# keep size 64k: replace current-map entries unused in A by A's missing tokens
missing = [i for i, _ in ca.most_common() if i not in hs]
unused = [i for i in hot if ca[i] == 0]
swap = set(hot) - set(unused[:len(missing)]) | set(missing[:len(unused)])
print(f'size-preserving swap ({min(len(missing), len(unused))} swapped, size {len(swap)}): {cov(swap):.3f}%')
