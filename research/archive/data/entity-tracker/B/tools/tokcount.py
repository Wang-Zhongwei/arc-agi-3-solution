"""Token counts of the delivered tracker text (json fields + tail line) with the Qwen3.8-Flash-Next tokenizer."""
import json, sys
from tokenizers import Tokenizer
tok = Tokenizer.from_file("/Users/advaith/.cache/huggingface/hub/models--Intel--Qwen3.8-Flash-Next-W4A16-AutoRound/snapshots/4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897/tokenizer.json")


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))] if xs else 0


f, tl, tot = [], [], []
for line in open(sys.argv[1]):
    r = json.loads(line)
    fs = json.dumps({"entities": r["fields"]}, separators=(", ", ": ")) if r["fields"] else ""
    nf = len(tok.encode(fs, add_special_tokens=False).ids) if fs else 0
    nt = len(tok.encode(r["tail"], add_special_tokens=False).ids) if r["tail"] else 0
    f.append(nf); tot.append(nf + nt)
    if r["tail"]:
        tl.append(nt)
print(f"steps {len(f)}; fields tokens p50 {pct(f,50)} p95 {pct(f,95)} max {max(f)}; tail lines {len(tl)} ({len(tl)/len(f):.1%} of steps) "
      f"tokens p50 {pct(tl,50)} max {max(tl) if tl else 0}; fields+tail p50 {pct(tot,50)} p95 {pct(tot,95)} max {max(tot)}; "
      f"over 100: {sum(1 for x in tot if x > 100)}; tail over 40: {sum(1 for x in tl if x > 40)}; mean {sum(tot)/len(tot):.1f}")
