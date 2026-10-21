"""Coverage of the FR-Spec hot-token map over the model's own generated text (reasoning + tool-call code)."""
import json, lzma, glob, pickle, hashlib, sys, collections
from tokenizers import Tokenizer
TOK = sys.argv[1]; HOT = sys.argv[2]; RUN = sys.argv[3]
tok = Tokenizer.from_file(TOK)
hot = set(pickle.load(open(HOT, 'rb')))  # plain list of ints (checked with pickletools: no globals)
print('hot set size', len(hot))
seen = set(); cnt = collections.Counter(); n_tok = 0; n_in = 0; per_kind = collections.Counter(); per_kind_in = collections.Counter()
for p in sorted(glob.glob(RUN + '/*_requests.jsonl.xz')):
    with lzma.open(p, 'rt') as f:
        for line in f:
            if '"event": "request"' not in line[-400:] and '"event": "request"' not in line: continue
            r = json.loads(line)
            if r.get('event') != 'request': continue
            for m in r.get('messages') or []:
                if m.get('role') != 'assistant': continue
                parts = [('reason', m.get('reasoning_content') or '')]
                for tc in m.get('tool_calls') or []:
                    parts.append(('code', tc.get('function', {}).get('arguments', '')))
                for kind, text in parts:
                    if not text: continue
                    h = hashlib.md5(text.encode()).digest()
                    if h in seen: continue
                    seen.add(h)
                    ids = tok.encode(text, add_special_tokens=False).ids
                    for i in ids:
                        cnt[i] += 1
                    k_in = sum(1 for i in ids if i in hot)
                    n_tok += len(ids); n_in += k_in; per_kind[kind] += len(ids); per_kind_in[kind] += k_in
print(f'generated tokens seen {n_tok:,}; in hot map {100*n_in/n_tok:.3f}%')
for k in per_kind: print(f'  {k}: {per_kind[k]:,} tokens, in map {100*per_kind_in[k]/per_kind[k]:.3f}%')
miss = [(c, i) for i, c in cnt.items() if i not in hot]; miss.sort(reverse=True)
print('distinct tokens used', len(cnt), 'distinct outside map', len(miss))
print('top missing:', [(tok.id_to_token(i), c) for c, i in miss[:25]])
# a map built from these transcripts: coverage of top-k by frequency (in-sample)
tot = sum(cnt.values())
for k in (16384, 32768, 65536):
    top = sum(c for _, c in collections.Counter(cnt).most_common(k))
    print(f'in-sample coverage of own top-{k}: {100*top/tot:.3f}%')
