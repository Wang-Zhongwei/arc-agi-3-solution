"""Token counts (Qwen3.8-Flash-Next tokenizer) of the tracker's tail line and result-dict fields per update."""
import json, sys
from tokenizers import Tokenizer
TOK = "/Users/advaith/.cache/huggingface/hub/models--Intel--Qwen3.8-Flash-Next-W4A16-AutoRound/snapshots/4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897/tokenizer.json"
t = Tokenizer.from_file(TOK)
tail, facts, both = [], [], []
for line in open(sys.argv[1]):
    r = json.loads(line)
    a = len(t.encode(r["tail"], add_special_tokens=False).ids) if r["tail"] else 0
    b = len(t.encode(r["facts"], add_special_tokens=False).ids) if r["facts"] != "{}" else 0
    tail.append(a); facts.append(b); both.append(a + b)
def pct(v, q):
    v = sorted(v); return v[min(len(v) - 1, int(round(q / 100 * (len(v) - 1))))]
for name, v in (("tail_line", tail), ("result_fields", facts), ("tail+fields", both)):
    print(f"{name}: n={len(v)} mean={sum(v)/len(v):.1f} p50={pct(v,50)} p95={pct(v,95)} max={max(v)} share>40={sum(x>40 for x in v)/len(v):.3f} share>100={sum(x>100 for x in v)/len(v):.3f}")
