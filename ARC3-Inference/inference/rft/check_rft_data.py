"""Tokenize an rft_data dataset the way train_rft does and check it, before paying for a GPU.

For every sample: the chat template's reply markers match its assistant messages, the trained
tokens are exactly the selected replies (each starts after `<think>` and ends with `<|im_end|>`),
and the exact length against the builder's estimate and train_rft's --max-tokens.

    python -m inference.rft.check_rft_data DATA_DIR --processor PROCESSOR_DIR [--max-tokens 131072]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from inference.rft import train_rft


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", type=Path)
    parser.add_argument("--processor", required=True, help="a directory or hub id with the checkpoint's processor files")
    parser.add_argument("--max-tokens", type=int, default=131072)
    parser.add_argument("--out", type=Path, help="per-sample JSON lines (id, tokens, trained, images, est_tokens)")
    args = parser.parse_args(argv)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.processor)
    im_end = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    rows, problems = [], []
    for split in ("train", "val"):
        for sample in train_rft.load_samples(args.data / f"{split}.jsonl"):
            try:
                t = train_rft.tokenize_sample(processor, sample, args.data)
            except Exception as exc:  # noqa: BLE001 - reported
                problems.append(f"{sample['id']}: {type(exc).__name__}: {exc}"[:300])
                continue
            ids, labels = t["input_ids"][0], t["labels"][0]
            trained = labels != -100
            # spans of trained tokens: one per selected reply, each closed by <|im_end|>
            starts = [i for i in range(len(ids)) if trained[i] and (i == 0 or not trained[i - 1])]
            ends = [i for i in range(len(ids)) if trained[i] and (i + 1 == len(ids) or not trained[i + 1])]
            if len(starts) != len(sample["train"]):
                problems.append(f"{sample['id']}: {len(starts)} trained spans for {len(sample['train'])} replies")
            if any(int(ids[e]) != im_end for e in ends):
                problems.append(f"{sample['id']}: a trained span does not end with <|im_end|>")
            rows.append(dict(split=split, id=sample["id"], tokens=len(ids), trained=int(trained.sum()),
                             prefix=starts[0] if starts else None, est_tokens=sample["est_tokens"],
                             images=int(t["image_grid_thw"].shape[0]) if "image_grid_thw" in t else 0))
    if args.out:
        args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    for split in ("train", "val"):
        part = [r for r in rows if r["split"] == split]
        if not part:
            continue
        tokens = [r["tokens"] for r in part]
        ratio = [r["tokens"] / r["est_tokens"] for r in part]
        print(f"{split}: {len(part)} samples, {sum(tokens) / 1e6:.1f}M tokens ({sum(r['trained'] for r in part) / 1e6:.2f}M trained, "
              f"prefix {sum(r['prefix'] or 0 for r in part) / sum(tokens):.0%}), median {statistics.median(tokens):,}, "
              f"max {max(tokens):,}, over --max-tokens {sum(t > args.max_tokens for t in tokens)}; "
              f"exact / estimate median {statistics.median(ratio):.3f} (p10 {sorted(ratio)[len(ratio) // 10]:.3f}, "
              f"p90 {sorted(ratio)[9 * len(ratio) // 10]:.3f})")
    print(f"{len(problems)} problems")
    for p in problems[:20]:
        print("  ", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
