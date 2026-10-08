## Changes relative to Daniel's baseline

[v3.json](v3.json) is v2 plus `kaggle.sglang.OFFLOAD_EMBED_TOKENS = true`: the input embedding table moves to pinned host memory, freeing about 1.2 GiB of GPU memory for the KV pool (about 100K more tokens on top of 1,011,264). Daniel's runs fill that pool to 93–98%. The lookup reuses the PLE host-gather kernel and costs microseconds per decode step. v3 also raises `server_startup_timeout_s` from 720 to 1200 so the benchmark waits for the server; v2 released the games about 3 minutes before its server was ready after a 621 s weight load.

Use [v2.json](v2.json) for the resubmission after rebasing onto `main` at `7a64a88`.
It includes the Pennyroyal handoff fix so the analyzer and multimodal settings reach the agent; v01 lost those settings, including board images and the 131072-token context window.

1. `thinking_token_budget = 8192`. [Evidence](../../../docs/reasoning/feature_report.md).
2. `tool_output_tokens = 1024`. [Evidence](../../../docs/tools/tool-output-tokens.md).
3. Use one private wave (the knob is removed) and simplify the priority allocator. [Evidence](../../../docs/allocation/priority-replay.md).
4. Deduplicate game-over instructions and repeated tool instructions in the user prompt. Principle: keep the prompt as simple as possible.
5. Remove `shape_hash` and `pose_hash`, and add `bbox`. [Evidence](../../../docs/harness/segmentation-node-fields.md).
6. Expose previously forbidden modules such as `sys`, `os`, and `inspect`, which the agent may try to use anyway. Principle: reduce avoidable errors.
7. `multimodal.upscale = 8`. The visual difference between 8 and 10 is small, but 10 uses about 56% more image tokens because image area scales quadratically. Image size also affects multimodal prefill cost. See the [image comparison](../../../docs/multimodal/README.md) and the recorded Lord Hansolo runs [v11](../../runs/kaggle/jonathanwang2022/bp35-lordhansolo/v11/) and [v12](../../runs/kaggle/jonathanwang2022/bp35-lordhansolo/v12/).
