"""Tables from results_<model>.json: mean KL per tensor x format, by layer region, and whole-model layouts."""
import json, sys, statistics as st
from collections import defaultdict
path = sys.argv[1]; R = json.load(open(path))
per = [r for r in R if isinstance(r["layer"], int) and ":" in r["tag"] and "coh" not in r["tag"] and "ind" not in r["tag"]] \
    + [r for r in R if isinstance(r["layer"], int) and r["tag"].startswith("kv")]
L = max(r["layer"] for r in per) + 1
region = lambda l: "first3" if l < 3 else ("last3" if l >= L - 3 else "middle")
def table(rows, keyf, title):
    g = defaultdict(list)
    for r in rows: g[keyf(r)].append(r["kl_mean"])
    print(f"\n## {title}\n\n| perturbation | mean KL | max KL | n |\n|---|---|---|---|")
    for k, v in sorted(g.items(), key=lambda kv: -st.mean(kv[1])):
        print(f"| {k} | {st.mean(v):.5f} | {max(v):.5f} | {len(v)} |")
table(per, lambda r: r["tag"], "A. one tensor of one layer, mean over layers")
table(per, lambda r: f'{r["tag"]} / {region(r["layer"])}', "A. by layer region")
grow = defaultdict(list)
for r in per: grow[r["layer"]].append(r["growth"])
print("\n## Error growth to the last block, by injected layer\n\n| layer | mean growth |\n|---|---|")
for l in sorted(grow): print(f"| {l} | {st.mean(grow[l]):.2f} |")
print("\n## B. matched relative error: coherent weight noise vs per-token noise\n\n| tensor | rel | layer | KL coherent | KL independent | ratio |\n|---|---|---|---|---|---|")
B = {(r["tag"], r["layer"]): r["kl_mean"] for r in R if "coh" in r["tag"] or "ind" in r["tag"]}
for (tag, l), kl in sorted(B.items()):
    if "coh" in tag:
        name, rel = tag.split(":coh"); ind = B[(f"{name}:ind{rel}", l)]
        print(f"| {name} | {rel} | {l} | {kl:.5f} | {ind:.5f} | {kl/ind if ind else float('nan'):.1f} |")
print("\n## C. whole-model layouts\n\n| layout | mean KL | p95 KL | top-1 disagreement |\n|---|---|---|---|")
for r in R:
    if r["layer"] == "all": print(f"| {r['tag']} | {r['kl_mean']:.5f} | {r['kl_p95']:.4f} | {r['top1_dis']:.4f} |")
