"""Install opt-in source hooks into Daniel's pinned, already patched wheel.

No wheel rebuild and no change to the five baseline patches. Hooks consult
runtime flags so reusing a venv with a feature disabled restores its baseline.
"""

import argparse
import ast
import hashlib
import importlib.metadata
from pathlib import Path

VERSION = "0.5.19+gd00d88efc8d6"
BUILD_ANCHOR = """    def _build_embed_tokens(self, config: Qwen4ExpTextConfig) -> nn.Module:
        return VocabParallelEmbedding("""
BUILD_HOOK = """    def _build_embed_tokens(self, config: Qwen4ExpTextConfig) -> nn.Module:
        # ARC3_BF16_INPUT_EMBEDDING_V1
        from sglang.srt.models.arc3_bf16_embedding import enabled, build_embedding
        if enabled():
            return build_embedding(config, is_draft=self._arc3_embedding_is_draft,
                                   use_attn_tp_group=is_dp_attention_enabled())
        return VocabParallelEmbedding("""
INIT_ANCHOR = "        super().__init__(config, quant_config, prefix, is_nextn)\n"
INIT_HOOK = "        self._arc3_embedding_is_draft = is_nextn\n" + INIT_ANCHOR
WIDE_HOOK = '''

# ARC3_WIDE_BF16_GEMM_V1: leave the original M <= 32 path intact.
import os as _arc3_os
if _arc3_os.environ.get("ARC3_WIDE_BF16_GEMM", "0") == "1":
    from sglang.kernels.ops.gemm.arc3_wide_bf16_gemm import eligible as _arc3_eligible, gemm as _arc3_gemm
    _arc3_lowm_use = use_sm120_lowm_bf16_gemm
    _arc3_lowm_gemm = sm120_lowm_bf16_gemm

    def use_sm120_lowm_bf16_gemm(m, n, k):
        return _arc3_lowm_use(m, n, k) or _arc3_eligible(m, n, k)

    def sm120_lowm_bf16_gemm(x, weight):
        if _arc3_eligible(x.numel() // x.shape[-1], weight.shape[0], weight.shape[1]):
            return _arc3_gemm(x, weight)
        return _arc3_lowm_gemm(x, weight)
'''


def replace_once(source, old, new):
    if source.count(new) == 1:
        return source
    if source.count(old) != 1:
        raise ValueError("Pennyroyal source hook does not match the pinned revision; refusing to patch")
    return source.replace(old, new, 1)


def install(root: Path, *, embedding=False, wide_gemm=False):
    sources = Path(__file__).parent
    edits = {}
    if embedding:
        path = root / "srt/models/qwen4_exp.py"
        source = replace_once(path.read_text(), BUILD_ANCHOR, BUILD_HOOK)
        edits[path] = replace_once(source, INIT_ANCHOR, INIT_HOOK)
        edits[root / "srt/models/arc3_bf16_embedding.py"] = (sources / "pennyroyal_bf16_embedding.py").read_text()
    if wide_gemm:
        path = root / "kernels/ops/gemm/sm120_lowm_bf16_gemm.py"
        source = path.read_text()
        # Refuse an unexpected low-M implementation or an edited prior hook.
        if "_MAX_M = 32" not in source or "def use_sm120_lowm_bf16_gemm(" not in source:
            raise ValueError("Daniel's M <= 32 BF16 GEMM patch is required")
        if "# ARC3_WIDE_BF16_GEMM_V1" in source and not source.endswith(WIDE_HOOK):
            raise ValueError("Previously installed wide GEMM hook differs")
        edits[path] = source if source.endswith(WIDE_HOOK) else source + WIDE_HOOK
        edits[root / "kernels/ops/gemm/arc3_wide_bf16_gemm.py"] = (sources / "pennyroyal_wide_gemm.py").read_text()
    # Validate every hook before changing any installed file.
    for path, source in edits.items():
        ast.parse(source, filename=str(path))
    for path, source in edits.items():
        if not path.exists() or path.read_text() != source:
            temp = path.with_suffix(".arc3.tmp")
            temp.write_text(source)
            temp.replace(path)
        print(f"ARC3 optional patch: {path.name} sha256={hashlib.sha256(source.encode()).hexdigest()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding", action="store_true")
    parser.add_argument("--wide-gemm", action="store_true")
    args = parser.parse_args()
    dist = importlib.metadata.distribution("sglang")
    if dist.version != VERSION:
        raise RuntimeError(f"Optional patches require sglang=={VERSION}, found {dist.version}")
    install(Path(dist.locate_file("sglang")), embedding=args.embedding, wide_gemm=args.wide_gemm)
