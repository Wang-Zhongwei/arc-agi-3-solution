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
        "            )\n",
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
