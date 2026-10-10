# Animation images with exact-detail guidance

Independent arm on `animations-in-context`. Config: `configs/animations-in-context/v1.json`; destination: `jonathanwang2022/animations-in-context`, private source dataset `jonathanwang2022/animations-in-context-src`.

Enable sampled animation images, set `multimodal.upscale=4` for all board/animation images, and add this sentence to the system prompt and user prompts that actually carry an animation image:

> Animation images are for intuitive understanding and may omit frames or fine details. Verify details in Python using `last_animation_frames` via `.segmentation` and targeted `.ascii` crops; `current_frame` is the settled board.

Sampled frames retain their existing indexing and eight-frame limit; full frames remain inspectable in Python. This tests the requested combination of images, guidance, and scale together, so a result alone cannot isolate which component caused it. Recovery is off and yield stays at 2,048.

All overnight arms start directly from `allocation-vc-one-wave` at `b96b113b4af948e55369d5739f56bcf6aa126ccb`, fetched from origin on 2026-10-10. No arm includes another arm's treatment. The two initial recovery runs must finish before this queue starts; thereafter no more than two notebooks may be RUNNING/QUEUED/PENDING on `jonathanwang2022` at once. Submission is already authorized and uses the normal `make kaggle-duck` target directly.

Settings held fixed: the same 15 exact game IDs and order as `early-level-recovery-baseline` v1; one public pass; priority one-wave scheduling; 14 active model streams and vLLM sequences; Lord vLLM `e975732`, Intel W4A16 target with Lord NVFP4 MTP draft (3 tokens); 131-minute public soft end plus 10-minute notebook buffer; context 131,072, server max length 147,072, thinking budget 8,192, max response 12,288 and tool returns 3,072. Intermediate action boards are off. Current/diff/death board images remain enabled. No generated-token cap per game. The two-job queue limit is Kaggle notebook concurrency, separate from each notebook's 14 inference streams.

Games: `tn36`, `lf52`, `cn04`, `bp35`, `wa30`, `sp80`, `m0r0`, `ka59`, `sk48`, `dc22`, `cd82`, `g50t`, `ls20`, `s5i5`, `su15`. Config files retain their exact IDs.

The allocation tip includes newer bounded startup prefetch than the already running baseline v1. All four new arms inherit it equally. When comparing to that earlier baseline, record actual startup/play time and generated tokens rather than attributing a startup-time difference to prompt behavior. One pass is exploratory; compare per-game scores, levels, tokens, request failures, and intervention-adjacent transcripts before claiming a gain.

Validation on 2026-10-10: Ruff passed; full suite **388 passed, 64 existing branch-divergence skips**; real Make-to-serving handoff and rendered ordinary/death prompts passed. Animation validation decoded every attached PNG and confirmed 256×256 for a 64×64 board at scale 4. Recovery validation confirms unused types in all opener modes and correct execution/level attribution. Simpler-prompt validation confirms the system prompt is byte-identical to the allocation baseline and the representative death opener shrinks from 2,912 to 1,654 characters. Normalized configs are identical after removing only the arm's named treatment and notebook/source metadata. No gameplay gain has been measured yet.
