"""Source patches that tune Daniel's Pennyroyal SGLang for lordhansolo's model.

Applied by pennyroyal_launch.py to the installed `sglang` package (the
dfranzen/pennyroyal-v253 wheel: Pennyroyal d00d88e plus Daniel's five patches)
after the offline install and before the server starts. Each edit replaces one
exact anchor that must occur once; an edit whose replacement is already present
is skipped, so reapplying is a no-op. Files under pennyroyal_overlay/ are copied
in as new modules. Behavior changes stay off unless their environment flag is
set (see docs/serving/LORD_ON_SGLANG_PLAN.md):

- SGLANG_QWEN4_EXP_HOST_EMBED_TOKENS=1: `embed_tokens` lives in pinned host
  memory and is gathered by the PLE kernel (frees ~1.2 GiB of GPU for KV).
- SGLANG_CT_NVFP4_MOE_MARLIN=1: compressed-tensors NVFP4 experts run on Marlin
  W4A16 instead of FlashInfer CUTLASS; the first layer is checked against a
  dequantized reference at load.
- SGLANG_SM120_LOWM_FP8_GEMM=1: FP8 channelwise linears with M <= the low-M
  limit use a weight-only split-K Triton GEMM (pennyroyal_overlay).
- SGLANG_SM120_LOWM_MAX_M: low-M limit for that path and Daniel's BF16 one
  (default 32, his value).
- SGLANG_QWEN4_EXP_MTP_SHARED_HEAD=1: the MTP draft builds its LM head on the
  meta device. It never loads that head (only mtp.* tensors) and uses the
  target's, so this frees 1.18 GiB before the KV pool is sized.
- SGLANG_QSA_INDEX_FP8=1: the QSA compressed index keys are stored in plain e4m3,
  like vLLM's indexer_kv_dtype=fp8: 384 instead of 768 B per KV token over the
  12 full-attention layers. A startup probe checks the FP8 decode kernel
  against the torch reference first; if it fails, the cache stays BF16.

Always on once applied: an O(1) expert-name lookup in Qwen4-Exp load_weights,
which otherwise scans all 3 x num_experts mapping entries per tensor.
"""
from __future__ import annotations

import shutil
from pathlib import Path

OVERLAY_DIR = Path(__file__).resolve().parent / "pennyroyal_overlay"

BF16_GEMM = "sglang/kernels/ops/gemm/sm120_lowm_bf16_gemm.py"
CT_FP8 = "sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py"
CT_NVFP4_MOE = "sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py"
MARLIN_FP4 = "sglang/srt/layers/quantization/marlin_utils_fp4.py"
QWEN4_EXP = "sglang/srt/models/qwen4_exp.py"
QWEN4_EXP_MTP = "sglang/srt/models/qwen4_exp_mtp.py"
QSA_POOL = "sglang/srt/mem_cache/qsa_kv_pool.py"
QSA_INDEXER = "sglang/srt/layers/attention/qsa/qsa_indexer.py"
QSA_MQA = "sglang/srt/layers/attention/qsa/mqa.py"

# (file relative to site-packages, exact anchor, replacement). The replacement
# keeps the anchor text where code is only inserted, so the "already applied"
# test is: replacement present, and anchor present only inside it.
EDITS: list[tuple[str, str, str]] = [
    # ---- Daniel's low-M BF16 GEMM: configurable M limit, wide tiles above 32, self-check ----
    (
        BF16_GEMM,
        "from typing import Optional\n",
        "import os\nfrom typing import Optional\n",
    ),
    (
        BF16_GEMM,
        "_MAX_M = 32\n",
        "# 14 streams x 4 verify tokens reach 56; SGLANG_SM120_LOWM_MAX_M raises it.\n"
        '_MAX_M = int(os.environ.get("SGLANG_SM120_LOWM_MAX_M", "32"))\n',
    ),
    (
        BF16_GEMM,
        "    n_ctas = triton.cdiv(n, 16)\n"
        "    split_k = 1\n",
        "    if m > 32:\n"
        "        # 56-token verify batches: 64-wide N tiles so the 64-row activation\n"
        "        # tile is not re-read for every 16 output columns; BLOCK_K 64 keeps\n"
        "        # three stages within SM120 shared memory. Chosen by rule, not timed.\n"
        "        n_ctas = triton.cdiv(n, 64)\n"
        "        split_k = 1\n"
        "        while split_k < 16 and n_ctas * split_k < 376 and (k // (split_k * 2)) >= 256:\n"
        "            split_k *= 2\n"
        "        return (split_k, 64, 64, 4, 3)\n"
        "    n_ctas = triton.cdiv(n, 16)\n"
        "    split_k = 1\n",
    ),
    (
        BF16_GEMM,
        "def sm120_lowm_bf16_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:\n",
        "def sm120_lowm_bf16_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:\n"
        '    """Checked once per shape bucket against F.linear; a failed bucket uses it."""\n'
        "    from sglang.kernels.ops.gemm.sm120_lowm_fp8_gemm import maybe_selfcheck\n"
        "\n"
        "    if not maybe_selfcheck(\n"
        '        "bf16",\n'
        "        x.numel() // x.shape[-1],\n"
        "        weight,\n"
        "        lambda xs: _sm120_lowm_bf16_gemm_raw(xs, weight),\n"
        "        lambda xs: torch.nn.functional.linear(xs, weight),\n"
        "    ):\n"
        "        return torch.nn.functional.linear(x, weight)\n"
        "    return _sm120_lowm_bf16_gemm_raw(x, weight)\n"
        "\n"
        "\n"
        "def _sm120_lowm_bf16_gemm_raw(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:\n",
    ),
    # ---- compressed-tensors FP8 linears: small-M weight-only path ----
    (
        CT_FP8,
        'strategy_to_parameter_type = {\n',
        "# Pennyroyal tuning: on exact SM120 with SGLANG_SM120_LOWM_FP8_GEMM=1, small-M\n"
        "# channelwise FP8 linears skip per-token activation quant + CUTLASS and use a\n"
        "# weight-only split-K Triton GEMM (sglang.kernels.ops.gemm.sm120_lowm_fp8_gemm).\n"
        "_sm120_lowm_fp8 = None\n"
        "\n"
        "\n"
        "def _sm120_lowm_fp8_module():\n"
        "    global _sm120_lowm_fp8\n"
        "    if _sm120_lowm_fp8 is None:\n"
        "        _sm120_lowm_fp8 = False\n"
        "        if (\n"
        '            get_bool_env_var("SGLANG_SM120_LOWM_FP8_GEMM")\n'
        "            and torch.cuda.is_available()\n"
        "            and torch.cuda.get_device_capability() == (12, 0)\n"
        "        ):\n"
        "            from sglang.kernels.ops.gemm import sm120_lowm_fp8_gemm\n"
        "\n"
        "            _sm120_lowm_fp8 = sm120_lowm_fp8_gemm\n"
        "    return _sm120_lowm_fp8\n"
        "\n"
        "\n"
        'strategy_to_parameter_type = {\n',
    ),
    (
        CT_FP8,
        "        if _use_aiter and self.strategy == QuantizationStrategy.CHANNEL:\n"
        "            return apply_fp8_ptpc_linear(\n",
        "        if (\n"
        "            bias is None\n"
        "            and self.strategy == QuantizationStrategy.CHANNEL\n"
        "            and x.dtype == torch.bfloat16\n"
        "            and x.is_cuda\n"
        "            and not _use_aiter\n"
        "        ):\n"
        "            lowm = _sm120_lowm_fp8_module()\n"
        "            if lowm:\n"
        "                weight_nk = layer.weight.t()\n"
        "                m = x.numel() // x.shape[-1]\n"
        "                if lowm.use_sm120_lowm_fp8_gemm(m, *weight_nk.shape):\n"
        "                    out = lowm.sm120_lowm_fp8_gemm(x, weight_nk, layer.weight_scale)\n"
        "                    if out is not None:\n"
        "                        return out\n"
        "\n"
        "        if _use_aiter and self.strategy == QuantizationStrategy.CHANNEL:\n"
        "            return apply_fp8_ptpc_linear(\n",
    ),
    # ---- compressed-tensors NVFP4 MoE: Marlin W4A16 runner ----
    (
        CT_NVFP4_MOE,
        "from sglang.srt.utils import set_weight_attrs\n",
        "from sglang.srt.utils import get_bool_env_var, set_weight_attrs\n",
    ),
    (
        CT_NVFP4_MOE,
        "        self.use_flashinfer_trtllm = get_moe_runner_backend().is_flashinfer_trtllm()\n",
        "        self.use_flashinfer_trtllm = get_moe_runner_backend().is_flashinfer_trtllm()\n"
        "        # Pennyroyal tuning: Marlin W4A16 experts (SGLANG_CT_NVFP4_MOE_MARLIN=1).\n"
        "        self.use_marlin = (\n"
        '            get_bool_env_var("SGLANG_CT_NVFP4_MOE_MARLIN")\n'
        "            and not self.use_flashinfer_trtllm\n"
        "        )\n",
    ),
    (
        CT_NVFP4_MOE,
        "        return not self.use_flashinfer_trtllm\n",
        "        return not self.use_flashinfer_trtllm and not self.use_marlin\n",
    ),
    (
        CT_NVFP4_MOE,
        "        if self.use_flashinfer_trtllm:\n"
        "            w, s = reorder_w1w3_to_w3w1(\n",
        "        if self.use_marlin:\n"
        "            _process_nvfp4_moe_for_marlin(layer)\n"
        "            return\n"
        "\n"
        "        if self.use_flashinfer_trtllm:\n"
        "            w, s = reorder_w1w3_to_w3w1(\n",
    ),
    (
        CT_NVFP4_MOE,
        "        self.moe_runner_config = moe_runner_config\n"
        "        if self.use_flashinfer_trtllm:\n",
        "        self.moe_runner_config = moe_runner_config\n"
        "        if self.use_marlin:\n"
        "            self.runner = MoeRunner(MoeRunnerBackend.MARLIN, moe_runner_config)\n"
        "            return\n"
        "        if self.use_flashinfer_trtllm:\n",
    ),
    (
        CT_NVFP4_MOE,
        "        x = dispatch_output.hidden_states\n"
        "\n"
        "        if self.use_flashinfer_trtllm:\n",
        "        x = dispatch_output.hidden_states\n"
        "\n"
        "        if self.use_marlin:\n"
        "            return self.runner.run(dispatch_output, self.get_marlin_quant_info(layer))\n"
        "\n"
        "        if self.use_flashinfer_trtllm:\n",
    ),
    (
        CT_NVFP4_MOE,
        "    def create_moe_runner(\n",
        "    def get_marlin_quant_info(self, layer: torch.nn.Module):\n"
        "        from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo\n"
        "\n"
        "        expert_map = None\n"
        "        global_num_experts = -1\n"
        '        if hasattr(layer, "dispatcher") and hasattr(\n'
        '            layer.dispatcher, "local_expert_mapping"\n'
        "        ):\n"
        "            expert_map = layer.dispatcher.local_expert_mapping\n"
        "            if expert_map is not None:\n"
        "                global_num_experts = self.moe_runner_config.num_experts\n"
        "        return MarlinMoeQuantInfo(\n"
        "            w13_qweight=layer.w13_weight,\n"
        "            w2_qweight=layer.w2_weight,\n"
        "            w13_scales=layer.w13_weight_scale,\n"
        "            w2_scales=layer.w2_weight_scale,\n"
        "            w13_g_idx_sort_indices=None,\n"
        "            w2_g_idx_sort_indices=None,\n"
        "            weight_bits=4,\n"
        "            w13_global_scale=layer.w13_weight_scale_2,\n"
        "            w2_global_scale=layer.w2_weight_scale_2,\n"
        "            expert_map=expert_map,\n"
        "            global_num_experts=global_num_experts,\n"
        "        )\n"
        "\n"
        "    def create_moe_runner(\n",
    ),
    (
        CT_NVFP4_MOE,
        "class CompressedTensorsW4A4Nvfp4MoE(CompressedTensorsMoEScheme):\n",
        "_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)\n"
        "_marlin_checked = False\n"
        "\n"
        "\n"
        "def _dequant_nvfp4(packed, block_scale, global_scale):\n"
        '    """compressed-tensors NVFP4 -> fp32: low nibble first, fp8 block scale\n'
        '    per 16 values, divided by the stored (reciprocal) global scale."""\n'
        "    lut = torch.tensor(\n"
        "        _E2M1_VALUES + tuple(-v for v in _E2M1_VALUES),\n"
        "        dtype=torch.float32,\n"
        "        device=packed.device,\n"
        "    )\n"
        "    nibbles = torch.stack((packed & 0xF, packed >> 4), dim=-1).reshape(\n"
        "        packed.shape[0], -1\n"
        "    )\n"
        "    values = lut[nibbles.long()]\n"
        "    scales = block_scale.float().repeat_interleave(16, dim=-1)\n"
        "    return values * scales / global_scale.float()\n"
        "\n"
        "\n"
        "def _nvfp4_expert_reference(layer, expert, x):\n"
        '    """SwiGLU expert output from the checkpoint tensors, before any repack.\n'
        "    Gate and up use their own global scales, so a checkpoint that does not\n"
        '    share them (Marlin assumes it does) fails the check."""\n'
        "    inter = layer.intermediate_size_per_partition\n"
        "    w13 = torch.cat(\n"
        "        [\n"
        "            _dequant_nvfp4(\n"
        "                layer.w13_weight.data[expert, half * inter : (half + 1) * inter],\n"
        "                layer.w13_weight_scale.data[expert, half * inter : (half + 1) * inter],\n"
        "                layer.w13_weight_global_scale.data[expert, half],\n"
        "            )\n"
        "            for half in (0, 1)\n"
        "        ]\n"
        "    )\n"
        "    w2 = _dequant_nvfp4(\n"
        "        layer.w2_weight.data[expert],\n"
        "        layer.w2_weight_scale.data[expert],\n"
        "        layer.w2_weight_global_scale.data[expert],\n"
        "    )\n"
        "    gate_up = x.float() @ w13.T\n"
        "    hidden = torch.nn.functional.silu(gate_up[:, :inter]) * gate_up[:, inter:]\n"
        "    return hidden @ w2.T\n"
        "\n"
        "\n"
        "def _process_nvfp4_moe_for_marlin(layer) -> None:\n"
        '    """Repack compressed-tensors NVFP4 experts for Marlin W4A16.\n'
        "\n"
        "    Marlin consumes ModelOpt's weight_scale_2, the reciprocal of the\n"
        "    compressed-tensors weight_global_scale, and one scale shared by gate and up.\n"
        "    The first layer processed in this process is checked against a\n"
        '    dequantized reference; a mismatch stops the server."""\n'
        "    global _marlin_checked\n"
        "    from sglang.srt.layers.quantization.marlin_utils_fp4 import (\n"
        "        prepare_moe_nvfp4_layer_for_marlin,\n"
        "    )\n"
        "\n"
        "    if not torch.allclose(\n"
        "        layer.w13_weight_global_scale[:, 0], layer.w13_weight_global_scale[:, 1]\n"
        "    ):\n"
        "        logger.warning_once(\n"
        '            "w1_weight_global_scale must match w3_weight_global_scale. "\n'
        '            "Accuracy may be affected."\n'
        "        )\n"
        "    check = None\n"
        "    if not _marlin_checked:\n"
        "        _marlin_checked = True\n"
        "        gen = torch.Generator(device=layer.w13_weight.device).manual_seed(7)\n"
        "        hidden = layer.w2_weight.shape[1]\n"
        "        x = torch.randn(\n"
        "            (8, hidden), generator=gen, device=layer.w13_weight.device,\n"
        "            dtype=torch.float32,\n"
        "        ).to(torch.bfloat16)\n"
        "        experts = (0, layer.w13_weight.shape[0] - 1)\n"
        "        check = (x, experts, [_nvfp4_expert_reference(layer, e, x) for e in experts])\n"
        "    layer.w13_weight_scale_2 = torch.nn.Parameter(\n"
        "        (1 / layer.w13_weight_global_scale.data[:, 0]).contiguous(),\n"
        "        requires_grad=False,\n"
        "    )\n"
        "    layer.w2_weight_scale_2 = torch.nn.Parameter(\n"
        "        (1 / layer.w2_weight_global_scale.data).contiguous(), requires_grad=False\n"
        "    )\n"
        "    prepare_moe_nvfp4_layer_for_marlin(layer)\n"
        "    for name in (\n"
        '        "w13_weight_global_scale",\n'
        '        "w2_weight_global_scale",\n'
        '        "w13_input_global_scale",\n'
        '        "w2_input_global_scale",\n'
        "    ):\n"
        "        if hasattr(layer, name):\n"
        "            delattr(layer, name)\n"
        "    if check is None:\n"
        "        return\n"
        "    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (\n"
        "        fused_marlin_moe,\n"
        "    )\n"
        "\n"
        "    x, experts, references = check\n"
        "    for expert, want in zip(experts, references):\n"
        "        ids = torch.full((x.shape[0], 1), expert, dtype=torch.int32, device=x.device)\n"
        "        got = fused_marlin_moe(\n"
        "            hidden_states=x,\n"
        "            w1=layer.w13_weight,\n"
        "            w2=layer.w2_weight,\n"
        "            w1_scale=layer.w13_weight_scale,\n"
        "            w2_scale=layer.w2_weight_scale,\n"
        "            gating_output=torch.zeros(\n"
        "                (x.shape[0], layer.w13_weight.shape[0]),\n"
        "                dtype=torch.float32,\n"
        "                device=x.device,\n"
        "            ),\n"
        "            # Explicit float32: this runs inside the loader's BF16 default dtype,\n"
        "            # and Marlin reads top-k weights as float32.\n"
        "            topk_weights=torch.ones(\n"
        "                (x.shape[0], 1), dtype=torch.float32, device=x.device\n"
        "            ),\n"
        "            topk_ids=ids,\n"
        "            w1_global_scale=layer.w13_weight_scale_2,\n"
        "            w2_global_scale=layer.w2_weight_scale_2,\n"
        "            num_bits=4,\n"
        "        ).float()\n"
        "        err = (got - want).abs().max().item()\n"
        "        peak = want.abs().max().item()\n"
        "        rel = err / max(peak, 1e-6)\n"
        "        if not rel <= 5e-2:\n"
        "            raise RuntimeError(\n"
        '                f"NVFP4 Marlin MoE self-check failed (expert {expert}): max error "\n'
        '                f"{err:.4g} vs max |ref| {peak:.4g} (rel {rel:.3g})"\n'
        "            )\n"
        '        logger.info("NVFP4 Marlin MoE self-check ok: expert %d rel_err=%.2e", expert, rel)\n'
        "\n"
        "\n"
        "class CompressedTensorsW4A4Nvfp4MoE(CompressedTensorsMoEScheme):\n",
    ),
    (
        MARLIN_FP4,
        "    if layer.quant_config.group_size != 16:\n",
        "    # compressed-tensors configs carry no group_size; NVFP4 is always 16.\n"
        '    if getattr(layer.quant_config, "group_size", 16) != 16:\n',
    ),
    # ---- Qwen4-Exp: embed_tokens in pinned host memory; O(1) expert lookup ----
    (
        QWEN4_EXP,
        "    def _build_embed_tokens(self, config: Qwen4ExpTextConfig) -> nn.Module:\n"
        "        return VocabParallelEmbedding(\n",
        "    def _build_embed_tokens(self, config: Qwen4ExpTextConfig) -> nn.Module:\n"
        "        # Pennyroyal tuning (lordhansolo's trick): keep the input embedding in\n"
        "        # pinned host memory and gather rows with the PLE kernel, freeing its\n"
        "        # GPU copy for KV. The MTP draft shares the target's table.\n"
        "        from sglang.srt.utils import get_bool_env_var\n"
        "\n"
        "        if get_bool_env_var(\n"
        '            "SGLANG_QWEN4_EXP_HOST_EMBED_TOKENS"\n'
        "        ) and not is_dp_attention_enabled():\n"
        '            with torch.device("meta"):\n'
        "                embedding = VocabParallelEmbedding(\n"
        "                    config.vocab_size,\n"
        "                    config.hidden_size,\n"
        "                    org_num_embeddings=config.vocab_size,\n"
        "                )\n"
        "            embedding.register_buffer(\n"
        '                "weight_scale", torch.ones(1, dtype=torch.bfloat16), persistent=False\n'
        "            )\n"
        "            return Qwen4ExpPinnedHostEmbedding(embedding)\n"
        "        return VocabParallelEmbedding(\n",
    ),
    (
        QWEN4_EXP,
        "class Qwen4ExpModel(Qwen3_5ForCausalLM):\n",
        "# experts.<id>.<proj>. -- the weight_name key of FusedMoE.make_expert_params_mapping.\n"
        "import re as _re\n"
        "\n"
        '_EXPERT_WEIGHT_KEY = _re.compile(r"experts\\.\\d+\\.(?:gate_proj|up_proj|down_proj)\\.")\n'
        "\n"
        "\n"
        "class Qwen4ExpModel(Qwen3_5ForCausalLM):\n",
    ),
    (
        QWEN4_EXP,
        "        fused_expert_params_mapping = [\n"
        '            ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),\n'
        '            ("experts.w2_weight", "experts.down_proj", 0, "w2"),\n'
        "        ]\n",
        "        fused_expert_params_mapping = [\n"
        '            ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),\n'
        '            ("experts.w2_weight", "experts.down_proj", 0, "w2"),\n'
        "        ]\n"
        "        # The linear scan below costs 3 x num_experts substring tests per tensor\n"
        "        # (512 experts x 12 tensors each per layer); index it by weight_name.\n"
        "        expert_params_by_weight_name = {}\n"
        "        for mapping in expert_params_mapping:\n"
        "            expert_params_by_weight_name.setdefault(mapping[1], []).append(mapping)\n",
    ),
    (
        QWEN4_EXP,
        "                current_expert_params_mapping = (\n"
        "                    fused_expert_params_mapping\n"
        "                    if is_fused_expert\n"
        "                    else expert_params_mapping\n"
        "                )\n",
        "                if is_fused_expert:\n"
        "                    current_expert_params_mapping = fused_expert_params_mapping\n"
        "                else:\n"
        "                    expert_key = _EXPERT_WEIGHT_KEY.search(name)\n"
        "                    current_expert_params_mapping = (\n"
        "                        expert_params_by_weight_name.get(expert_key.group(0), ())\n"
        "                        if expert_key is not None\n"
        "                        else ()\n"
        "                    )\n",
    ),
]


EDITS += [
    # ---- Qwen4-Exp MTP draft: no private LM head (it is never loaded) ----
    (
        QWEN4_EXP_MTP,
        "        self.lm_head = ParallelLMHead(\n"
        "            config.vocab_size,\n"
        "            config.hidden_size,\n"
        "            quant_config=quant_config,\n"
        '            prefix=add_prefix("model.shared_head.head", prefix),\n'
        "            use_attn_tp_group=get_parallel().config.enable_dp_lm_head,\n"
        "        )\n",
        "        # Pennyroyal tuning (SGLANG_QWEN4_EXP_MTP_SHARED_HEAD=1): load_weights keeps\n"
        "        # only mtp.* tensors, so this head is never loaded; the speculative worker\n"
        "        # swaps in the target's head (or its hot-vocab rows) after the KV pool is\n"
        "        # sized. On meta it stops costing that pool 1.18 GiB of dead weight.\n"
        "        from contextlib import nullcontext\n"
        "\n"
        "        from sglang.srt.utils import get_bool_env_var\n"
        "\n"
        "        with (\n"
        '            torch.device("meta")\n'
        '            if get_bool_env_var("SGLANG_QWEN4_EXP_MTP_SHARED_HEAD")\n'
        "            else nullcontext()\n"
        "        ):\n"
        "            self.lm_head = ParallelLMHead(\n"
        "                config.vocab_size,\n"
        "                config.hidden_size,\n"
        "                quant_config=quant_config,\n"
        '                prefix=add_prefix("model.shared_head.head", prefix),\n'
        "                use_attn_tp_group=get_parallel().config.enable_dp_lm_head,\n"
        "            )\n"
        '        if get_bool_env_var("SGLANG_QWEN4_EXP_MTP_SHARED_HEAD"):\n'
        "            # Post-load staging refuses meta tensors: give the head an empty real\n"
        "            # weight (no rows, no memory) that the worker later replaces.\n"
        "            placeholder = nn.Parameter(\n"
        "                torch.empty((0, config.hidden_size), dtype=self.lm_head.weight.dtype),\n"
        "                requires_grad=False,\n"
        "            )\n"
        "            for key, value in vars(self.lm_head.weight).items():\n"
        "                setattr(placeholder, key, value)\n"
        "            self.lm_head.weight = placeholder\n",
    ),
]


EDITS += [
    # ---- QSA index keys in plain e4m3, as vLLM's indexer_kv_dtype=fp8 ----
    (
        QSA_POOL,
        "def _index_k_bytes(*, kv_heads: int, head_dim: int, dtype: torch.dtype) -> int:\n"
        "    return kv_heads * head_dim * dtype.itemsize\n",
        "def _index_k_bytes(*, kv_heads: int, head_dim: int, dtype: torch.dtype) -> int:\n"
        "    return kv_heads * head_dim * dtype.itemsize\n"
        "\n"
        "\n"
        "def _qsa_compressed_dtype() -> torch.dtype:\n"
        '    """Pennyroyal tuning (SGLANG_QSA_INDEX_FP8=1): the compressed index keys in\n'
        "    plain e4m3 without scales, as vLLM's indexer_kv_dtype=fp8 (the keys are\n"
        "    RMSNormed). Only when the FP8 decode kernel passes its startup probe;\n"
        '    BF16 otherwise. The pending ring stays BF16, as in vLLM."""\n'
        "    from sglang.srt.utils import get_bool_env_var\n"
        "\n"
        '    if not get_bool_env_var("SGLANG_QSA_INDEX_FP8"):\n'
        "        return torch.bfloat16\n"
        "    from sglang.srt.layers.attention.qsa.mqa import qsa_fp8_decode_dtype\n"
        "\n"
        "    return torch.float8_e4m3fn if qsa_fp8_decode_dtype() else torch.bfloat16\n",
    ),
    (
        QSA_POOL,
        "        index_k_bytes = _index_k_bytes(\n"
        "            kv_heads=kv_heads, head_dim=head_dim, dtype=cls.index_state_dtype\n"
        "        )\n"
        "        return index_k_bytes // compress_ratio * num_layers\n",
        "        index_k_bytes = _index_k_bytes(\n"
        "            kv_heads=kv_heads, head_dim=head_dim, dtype=_qsa_compressed_dtype()\n"
        "        )\n"
        "        return index_k_bytes // compress_ratio * num_layers\n",
    ),
    (
        QSA_POOL,
        "                * self.qsa_index_head_dim,\n"
        "            ),\n"
        "            dtype=self.index_state_dtype,\n",
        "                * self.qsa_index_head_dim,\n"
        "            ),\n"
        "            dtype=_qsa_compressed_dtype(),\n",
    ),
    (
        QSA_POOL,
        "        buffer = self.get_qsa_compressed_k_buffer(layer_id)\n"
        "        buffer[loc.long()] = compressed_k.to(buffer.dtype)\n",
        "        buffer = self.get_qsa_compressed_k_buffer(layer_id)\n"
        "        if buffer.dtype == torch.float8_e4m3fn:\n"
        "            # Saturate instead of overflowing to NaN.\n"
        "            compressed_k = compressed_k.float().clamp(-448.0, 448.0)\n"
        "        buffer[loc.long()] = compressed_k.to(buffer.dtype)\n",
    ),
    (
        QSA_INDEXER,
        "        compressed_buffer = pool.get_qsa_compressed_k_buffer(self.layer_id)\n"
        "        qsa_index_k_compress_store(\n"
        "            source_keys.reshape(source_keys.shape[0], -1)\n"
        "            .contiguous()\n"
        "            .to(pool.index_state_dtype),\n"
        "            group_locs.to(torch.int32),\n"
        "            source_rope,\n"
        "            self.rotary_emb.cos_sin_cache,\n"
        "            self._rope_axis_map(source_keys.device),\n"
        "            self.k_layernorm.weight.data,\n"
        "            write_locs.to(torch.int32),\n"
        "            compressed_buffer.view(compressed_buffer.shape[0], -1),\n"
        "            self.compress_ratio,\n"
        "            self.rotary_emb.rotary_dim,\n"
        "            self.k_layernorm.variance_epsilon,\n"
        "            self.rotary_emb.is_neox_style,\n"
        "        )\n",
        "        compressed_buffer = pool.get_qsa_compressed_k_buffer(self.layer_id)\n"
        "        target = compressed_buffer.view(compressed_buffer.shape[0], -1)\n"
        "        fp8_cache = target.dtype == torch.float8_e4m3fn\n"
        "        if fp8_cache:\n"
        "            # The fused kernel writes its source dtype (BF16): compress into one\n"
        "            # scratch row per write, then cast (saturating) into the e4m3 cache.\n"
        "            scratch = torch.empty(\n"
        "                (write_locs.shape[0], target.shape[1]),\n"
        "                dtype=pool.index_state_dtype,\n"
        "                device=target.device,\n"
        "            )\n"
        "            kernel_locs = torch.arange(\n"
        "                write_locs.shape[0], dtype=torch.int32, device=target.device\n"
        "            )\n"
        "        qsa_index_k_compress_store(\n"
        "            source_keys.reshape(source_keys.shape[0], -1)\n"
        "            .contiguous()\n"
        "            .to(pool.index_state_dtype),\n"
        "            group_locs.to(torch.int32),\n"
        "            source_rope,\n"
        "            self.rotary_emb.cos_sin_cache,\n"
        "            self._rope_axis_map(source_keys.device),\n"
        "            self.k_layernorm.weight.data,\n"
        "            kernel_locs if fp8_cache else write_locs.to(torch.int32),\n"
        "            scratch if fp8_cache else target,\n"
        "            self.compress_ratio,\n"
        "            self.rotary_emb.rotary_dim,\n"
        "            self.k_layernorm.variance_epsilon,\n"
        "            self.rotary_emb.is_neox_style,\n"
        "        )\n"
        "        if fp8_cache:\n"
        "            target[write_locs.long()] = (\n"
        "                scratch.float().clamp(-448.0, 448.0).to(target.dtype)\n"
        "            )\n",
    ),
    (
        QSA_MQA,
        "                                Logits[bx, position] = reduced[token] / Scale\n"
        "\n"
        "        return kernel\n"
        "\n"
        "\n"
        "def tilelang_qsa_mqa_prefill(\n",
        "                                Logits[bx, position] = reduced[token] / Scale\n"
        "\n"
        "        return kernel\n"
        "\n"
        "    @tilelang.jit(\n"
        "        pass_configs={\n"
        "            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,\n"
        "            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,\n"
        "        }\n"
        "    )\n"
        "    def _tilelang_qsa_mqa_decode_fp8_kernel(\n"
        "        heads: int,\n"
        "        head_dim: int,\n"
        "        k_dtype: str,\n"
        "        page_size: int = 64,\n"
        "        groups_per_cta: int = 1,\n"
        "        num_stages: int = 3,\n"
        "        threads: int = 128,\n"
        "    ):\n"
        '        """The decode kernel above over an FP8 (e4m3) compressed cache: T.copy\n'
        "        widens each page to BF16 in shared memory (e4m3 values are exact in\n"
        '        BF16), so the GEMM, ReLU and head sum are unchanged."""\n'
        "        GROUP = 64\n"
        "        assert GROUP % page_size == 0, page_size\n"
        "        sub_pages = GROUP // page_size\n"
        '        batch = T.dynamic("batch")\n'
        '        pages = T.dynamic("pages")\n'
        '        max_pages = T.dynamic("max_pages")\n'
        '        max_model_len = T.dynamic("max_model_len")\n'
        "\n"
        "        @T.prim_func\n"
        "        def kernel(\n"
        "            Q: T.Tensor([batch, 1, heads, head_dim], T.bfloat16),  # type: ignore\n"
        "            KCache: T.Tensor([pages, page_size, 1, head_dim], k_dtype),  # type: ignore\n"
        "            PageTable: T.Tensor([batch, max_pages], T.int32),  # type: ignore\n"
        "            ContextLens: T.Tensor([batch], T.int32),  # type: ignore\n"
        "            Logits: T.Tensor([batch, max_model_len], T.float32),  # type: ignore\n"
        "            Scale: T.float32,\n"
        "        ):\n"
        "            with T.Kernel(\n"
        "                batch,\n"
        "                T.ceildiv(T.ceildiv(max_pages, sub_pages), groups_per_cta),\n"
        "                threads=threads,\n"
        "            ) as (bx, group_block):\n"
        "                q_shared = T.alloc_shared([heads, head_dim], T.bfloat16)\n"
        "                k_shared = T.alloc_shared([GROUP, head_dim], T.bfloat16)\n"
        "                scores = T.alloc_fragment([GROUP, heads], T.float32)\n"
        "                reduced = T.alloc_fragment([GROUP], T.float32)\n"
        "                T.copy(Q[bx, 0, :, :], q_shared)\n"
        "                context_len = ContextLens[bx]\n"
        "\n"
        "                for gi in T.Pipelined(groups_per_cta, num_stages=num_stages):\n"
        "                    group = group_block * groups_per_cta + gi\n"
        "                    if group * GROUP < context_len:\n"
        "                        for sp in range(sub_pages):\n"
        "                            if (group * sub_pages + sp) * page_size < context_len:\n"
        "                                T.copy(\n"
        "                                    KCache[\n"
        "                                        PageTable[bx, group * sub_pages + sp],\n"
        "                                        :,\n"
        "                                        0,\n"
        "                                        :,\n"
        "                                    ],\n"
        "                                    k_shared[sp * page_size : (sp + 1) * page_size, :],\n"
        "                                )\n"
        "                        T.gemm(\n"
        "                            k_shared,\n"
        "                            q_shared,\n"
        "                            scores,\n"
        "                            transpose_B=True,\n"
        "                            clear_accum=True,\n"
        "                            policy=T.GemmWarpPolicy.FullCol,\n"
        "                        )\n"
        "                        for token, head in T.Parallel(GROUP, heads):\n"
        "                            scores[token, head] = T.max(scores[token, head], 0.0)\n"
        "                        T.reduce_sum(scores, reduced, dim=1, clear=True)\n"
        "                        for token in T.Parallel(GROUP):\n"
        "                            position = group * GROUP + token\n"
        "                            if position < context_len:\n"
        "                                Logits[bx, position] = reduced[token] / Scale\n"
        "\n"
        "        return kernel\n"
        "\n"
        "\n"
        "def tilelang_qsa_mqa_prefill(\n",
    ),
    (
        QSA_MQA,
        "    _tilelang_qsa_mqa_decode_kernel(\n"
        "        heads=kernel_heads, head_dim=head_dim, page_size=page_size\n"
        "    )(\n"
        "        q_kernel.unsqueeze(1).contiguous(),\n"
        "        k_cache.to(torch.bfloat16).contiguous(),\n",
        "    if k_cache.dtype == torch.float8_e4m3fn:\n"
        "        # Read the e4m3 cache in place; converting it whole would copy every\n"
        "        # compressed key of the layer on each decode step.\n"
        "        decode_kernel = _tilelang_qsa_mqa_decode_fp8_kernel(\n"
        "            heads=kernel_heads,\n"
        "            head_dim=head_dim,\n"
        "            k_dtype=qsa_fp8_decode_dtype(),\n"
        "            page_size=page_size,\n"
        "        )\n"
        "        k_kernel = k_cache\n"
        "    else:\n"
        "        decode_kernel = _tilelang_qsa_mqa_decode_kernel(\n"
        "            heads=kernel_heads, head_dim=head_dim, page_size=page_size\n"
        "        )\n"
        "        k_kernel = k_cache.to(torch.bfloat16).contiguous()\n"
        "    decode_kernel(\n"
        "        q_kernel.unsqueeze(1).contiguous(),\n"
        "        k_kernel,\n",
    ),
    (
        QSA_MQA,
        "def qsa_mqa_prefill(\n",
        "_QSA_FP8_DECODE_DTYPE = None\n"
        "\n"
        "\n"
        "def qsa_fp8_decode_dtype() -> str:\n"
        '    """TileLang dtype name under which the FP8 decode kernel compiled and\n'
        "    matched the torch reference on this GPU, or '' (keep the BF16 cache).\n"
        '    Probed once per process, before the KV pool is sized."""\n'
        "    global _QSA_FP8_DECODE_DTYPE\n"
        "    if _QSA_FP8_DECODE_DTYPE is not None:\n"
        "        return _QSA_FP8_DECODE_DTYPE\n"
        "    import logging\n"
        "\n"
        "    log = logging.getLogger(__name__)\n"
        '    _QSA_FP8_DECODE_DTYPE = ""\n'
        "    if not (HAS_TILELANG and torch.cuda.is_available()):\n"
        '        log.warning("QSA FP8 index: TileLang/CUDA unavailable; keeping BF16")\n'
        "        return _QSA_FP8_DECODE_DTYPE\n"
        '    gen = torch.Generator(device="cuda").manual_seed(3)\n'
        "    heads, head_dim, page_size, max_len = 8, 128, 16, 64\n"
        "    q = torch.randn(\n"
        '        (3, heads, head_dim), generator=gen, device="cuda", dtype=torch.float32\n'
        "    ).to(torch.bfloat16)\n"
        "    k_cache = torch.randn(\n"
        '        (6, page_size, 1, head_dim), generator=gen, device="cuda", dtype=torch.float32\n'
        "    ).to(torch.float8_e4m3fn)\n"
        "    table = torch.tensor(\n"
        '        [[0, 1, 2, 3], [4, 5, 0, 0], [3, 1, 0, 0]], dtype=torch.int32, device="cuda"\n'
        "    )\n"
        '    lengths = torch.tensor([64, 20, 33], dtype=torch.int32, device="cuda")\n'
        "    want = torch_qsa_mqa_decode(q, k_cache, table, lengths, max_len)\n"
        '    for name in ("float8_e4m3", "float8_e4m3fn"):\n'
        "        try:\n"
        "            got = torch.full(\n"
        "                (3, max_len), -float(\"inf\"), dtype=torch.float32, device=\"cuda\"\n"
        "            )\n"
        "            _tilelang_qsa_mqa_decode_fp8_kernel(\n"
        "                heads=heads, head_dim=head_dim, k_dtype=name, page_size=page_size\n"
        "            )(\n"
        "                q.unsqueeze(1).contiguous(),\n"
        "                k_cache,\n"
        "                table,\n"
        "                lengths,\n"
        "                got,\n"
        "                float(math.sqrt(head_dim)),\n"
        "            )\n"
        "            torch.cuda.synchronize()\n"
        "            if torch.allclose(got, want, rtol=2e-3, atol=2e-3):\n"
        "                _QSA_FP8_DECODE_DTYPE = name\n"
        '                log.info("QSA FP8 index: decode kernel ok with %s", name)\n'
        "                break\n"
        "            log.warning(\n"
        '                "QSA FP8 index: %s decode kernel mismatch (max err %.3g)",\n'
        "                name,\n"
        "                (got - want).nan_to_num(0.0, 0.0, 0.0).abs().max().item(),\n"
        "            )\n"
        "        except Exception as exc:  # compile or launch failure: try the next\n"
        '            log.warning("QSA FP8 index: %s decode kernel failed: %s", name, exc)\n'
        "    if not _QSA_FP8_DECODE_DTYPE:\n"
        '        log.warning("QSA FP8 index: probe failed; keeping the BF16 cache")\n'
        "    return _QSA_FP8_DECODE_DTYPE\n"
        "\n"
        "\n"
        "def qsa_mqa_prefill(\n",
    ),
]


def overlay_files() -> dict[str, Path]:
    """site-packages-relative path -> overlay source, for new modules."""
    return {
        str(path.relative_to(OVERLAY_DIR)): path
        for path in sorted(OVERLAY_DIR.rglob("*.py"))
    }


def _apply_edit(text: str, anchor: str, replacement: str, where: str) -> str:
    if replacement in text and text.count(anchor) == replacement.count(anchor):
        return text
    hits = text.count(anchor)
    if hits != 1:
        raise RuntimeError(f"{where}: expected the anchor once, found {hits}:\n{anchor}")
    return text.replace(anchor, replacement)


def apply(site_packages: str | Path) -> list[str]:
    """Patch the installed sglang under `site_packages`; return the files changed.

    All edits are computed before any file is written, so a missing anchor
    leaves the installation untouched.
    """
    root = Path(site_packages)
    pending: dict[Path, str] = {}
    for rel, anchor, replacement in EDITS:
        path = root / rel
        if not path.is_file():
            raise RuntimeError(f"{rel}: not found under {root}")
        text = pending.get(path) or path.read_text()
        pending[path] = _apply_edit(text, anchor, replacement, rel)
    changed = []
    for path, text in pending.items():
        if path.read_text() != text:
            tmp = path.with_name(path.name + ".tuneup.tmp")
            tmp.write_text(text)
            tmp.replace(path)
            changed.append(str(path.relative_to(root)))
    for rel, source in overlay_files().items():
        dest = root / rel
        if not dest.is_file() or dest.read_bytes() != source.read_bytes():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
            changed.append(rel)
    for path in pending:
        cache = path.parent / "__pycache__"
        for stale in cache.glob(path.stem + ".*.pyc") if cache.is_dir() else ():
            stale.unlink()
    return changed
