# ARC-AGI-3 Kaggle: models used by strong teams (as of 2026-10-17)

## Current public LB top (kaggle CLI, 2026-10-17)
Tufa Labs 68.91 (10-17) | Yi-Chia Chen 67.49 (10-16) | mtg 50.37 | the last dance 49.85 | Son & Mark & Ronen 48.92 | 45.11 | 45.02 | 44.84 | NVARC3 43.94 | 41.74 | 41.19 | 40.09 | 39.69 | 39.11 | 38.98 | 38.84 | "speed, my mom is kinda homless" (us) 38.22 | fshindo 38.20 | SparseTech 37.98 | mikelou1 37.27. (Public LB = ~50% of test data; final on other 50%.)
History: M2 (9-30) winners 27.9 / 23.8 / 22.5; Tufa 27.29 -> 45.33 (9-29) -> 55.89 (10-04/05) -> 68.91 now. Yi-Chia Chen 48.59 -> 59.17 (~10-09) -> 67.49.

## Table (VERIFIED unless marked)
| Team / notebook | Model + quant | Fine-tuned | Harness | LB (public) | Source |
|---|---|---|---|---|---|
| Tufa Labs, M1 (6-30) | Qwen3.6-27B FP8 | No (stated: pretrained, "additional post-training" listed as future work) | Duck (vLLM) | 1.21 | Kaggle discussion 717133 (write-up) |
| Tufa Labs, now (68.91) | UNKNOWN (not published; closed source until competition end) | UNKNOWN. Guess only: Scott Le Grand (rank ~38) wrote "my guess is part of that is some fine tuning" (discussion 742801). Tufa refused to open-source for M2 | Duck successor, unknown | 68.91 (10-17) | CLI leaderboard; discussion 742801 |
| Yi-Chia Chen | UNKNOWN (no public notebook found; no article names model) | UNKNOWN | UNKNOWN | 67.49 (10-16) | CLI leaderboard; metirai/techtimes say model not named |
| dfranzen M2 solution (1st M2) | Qwen3.8-Flash-Next, Intel W4A16 AutoRound + albucino MTP drafter | No ("I did not train or fine-tune the models") | Duck + SGLang patches | 27.9 (M2); our reruns of fork 26.9-38.2 | local competitors/tong-m2-comparison/WRITEUP.md; kernel metadata |
| Lord Han Solo (M2 2nd) | Qwen3.8-Flash-Next mixed NVFP4/FP8 + MTP | No indication | Duck, vLLM | 23.8 | built-on-tufa notebook; postcutoff |
| rellik13 / sirikilohit (M2 3rd) | Qwen3.8-Flash-Next W4A16 + FP8 PLE, NVFP4 MTP head | No | Duck/TAAF + Wang port, SGLang | 22.53 | competitors/rellik13-milestone2/README.md |
| skarin "27.80 LB 100-trajectory audit" | Franzen's unchanged notebook (Flash-Next W4A16) | No | Franzen M2 | 27.80 (one draw; shows noise) | local notebook |
| Nick Pellegrin (disc. comment) | M2 notebooks derived (Flash-Next) | no | Franzen-derived | ~23 private, ~40 local | discussion 732854 |
| Many forks (affectify/Osaka 31.54 single sub, sovereign-agent, dprime, kinetic, df-wm, win-v2, ...) | Flash-Next W4A16 (all share dfranzen model_sources) | No (df-wm and kinetic mention LoRA in text, unverified) | Franzen fork | e.g. affectify 31.54 | kernel-metadata.json; Kaggle titles |
| keithtyser / wuliao0 Duck anim base etc. | Flash-Next NVFP4 + MTP | No | Duck | ~2-6 early Sept | notebooks |
| arc3-b32-reap (sukirman) | Flash-Next NVFP4 REAP-pruned 448 experts | No | Duck | unknown | local notebook |
| Reki (M1 2nd), Md Boktiar Murad (M1 3rd, "forge") | Gemma-4-31B (VLM policy) | unknown | own VLM, JSON actions | ~1 (M1) | metirai, postcutoff |
| Greg Kamradt sample | GPT-OSS-120B | no | baseline | low | Kaggle notebook list |
| Scott Le Grand (rank ~38 in Sept) | Qwen3.8-Flash-Next NVFP4, "general LLM agent descended from duck"; tried Claude-built game solvers (impractical offline) | No | Duck-descended | 2.7-5.2 (Sept) | discussion 732854, 742801 |

## Verified takeaways
- Every public notebook with a >=20 LB score that we found uses Qwen3.8-Flash-Next (W4A16 AutoRound or NVFP4/mixed). Qwen3.6-27B FP8 and 3.8-27B were earlier steps (Franzen: "each switch brought a substantial improvement"). Franzen/Han Solo/Siriki scored 22-28 at M2 with it; ours 26.9-38.2 (mean ~31), consistent with other forks (31.54 etc.).
- No Qwen3.6-27B / Gemma / GPT-OSS notebook is shown to reach 30. Base LB >=30 evidence exists ONLY for Qwen3.8-Flash-Next (W4A16 and likely NVFP4 variants; the 38+ entries' models are not public).
- Nobody has publicly stated fine-tuning. The only public fine-tune mention is a rules question (disc. 742801): Tong Hui Kang asked whether a fine-tuned model must publish dataset+script for milestone eligibility; Van-Phuc Huynh asked about training on proprietary-model traces; no host answer seen.
- Top-2 (68.91, 67.49) are 20+ points above the next team (50.37). Their models/harnesses are undisclosed: Tufa said they will release the final solution when the competition ends (ends ~16 days from 10-17, i.e. ~Nov 2; papers Nov 8).
- Tufa says compute is a big lever (they have lots of GPUs) and the harness is what they iterate on; Le Grand's fine-tuning remark is speculation.

## Rules on models (Kaggle rules page, read 2026-10-17)
- Code competition, 1 submission/day, 2 final picks, 9h on one RTX PRO 6000 96GB, offline (from Tufa write-up).
- External data and pretrained models allowed if public and "reasonably accessible to all / minimal cost" (Reasonableness Standard).
- Prize winners (final top 5 and milestones) must license submission + code under CC-BY 4.0 and "open source system, open source model, and open source weights/parameters" per OSI Open Source AI definition; pretrained models with incompatible license are exempted from the license grant. Must deliver training code, inference code, documentation. Open code must use OSI-approved licenses allowing commercial use. Fine-tuned weights therefore must be released to win prizes (milestone: notebook + dependencies public before deadline).
- Public code sharing must be on Kaggle forum/notebooks; no private sharing outside teams. Max team 8.
- Data: Apache 2.0. Top score prize $150k (final 40/15/10/5/5k), grand prize $700k at 100%.

## Unverified / gaps
- Models and fine-tuning status of Tufa (68.91), Yi-Chia Chen (67.49), mtg, "the last dance", NVARC3, etc. Could not pull their notebooks (private). Code tab per-notebook best scores not scraped (JS page; only the leaderboard and discussion read).
- NVARC3 (name echoes NVIDIA's ARC-AGI-2 team, which fine-tuned models) at 43.94: speculation only. CPMP (rank 9, NVIDIA-linked per his "as usual for NVIDIA teams" comment) said they won't share until the end.
- Frontier numbers (GPT-6 Astra 99.9%, etc.) are API models with provider harnesses, irrelevant to offline track.
