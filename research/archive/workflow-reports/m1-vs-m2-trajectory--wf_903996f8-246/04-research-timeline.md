# m1-vs-m2-trajectory: research:timeline

Workflow run `wf_903996f8-246`, phase Research. Research what changed from ARC-AGI-3 Milestone 1 (Tufa Duck) to Milestone 2 solutions, then verify the claims

---

# ARC-AGI-3 timeline: MS1 to MS2 to now

Sources are cited inline. The Kaggle discussion threads I read from a local JSON copy (`scratchpad/d717133.json`, `scratchpad/disc/<id>.json`) and cite by thread URL, `https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/<id>`. Ranks and scores come from the live Kaggle LB, pulled with the CLI on 2026-10-05, file `scratchpad/lb/...2026-10-05T01:38:36.csv`.

## 1. Dates
- **Competition:** starts 2026-03-25. MS1 is 2026-06-30, MS2 is 2026-09-30, submissions are due 2026-11-02, and results are announced 2026-12-04 (https://arcprize.org/competitions/2026). Each milestone pays $25K / $10K / $2.5K (https://arcprize.org/competitions/2026/arc-agi-3).
- **MS1 announcement:** the ARC Prize post says the prize "ran through June 30th" (https://arcprize.org/blog/arc-prize-2026-milestone-1). Tufa open-sourced its code on 2026-06-30 (discussion/716696) and published its write-up on 2026-07-01 (discussion/717133).
- **MS2 announcement:** there is no ARC Prize blog post yet; `arcprize.org/blog/arc-prize-2026-milestone-2` returns 404. The winners are announced in an ARC Prize tweet (https://x.com/arcprize/status/2105737436450201734), which I could not fetch. It is linked from discussion/744792, posted 2026-10-01 00:36 UTC.

## 2. Milestone 1 (Jun 30): winners and LB scores
| Place | Team | LB | Model | Source |
|---|---|---|---|---|
| 1 | Tufa Labs, "The Duck" | **1.21** | Qwen3.6-27B FP8, vLLM | discussion/717133; model per the ARC post |
| 2 | Reki | unknown (a search summary said 0.867, not verified) | Gemma-4-31B | ARC post |
| 3 | Md Boktiar Mahbub Murad, "forge" | 0.86 | Gemma-4-31B | discussion/716719 |

- Tufa's 1.21 is the official score. Kaggle retracted a 1.30 run, and the same submission scored as low as 0.77 on rerun (discussion/717133).
- An AlphaSignal article says 1.03 (https://alphasignal.ai/news/tufa-labs-wins-25k-beating-frontier-ai-on-the-world-s-hardest-benchmark). I trust Tufa's own 1.21, because it is the primary source.
- Tufa scored 1.6002 ± 0.4475 on the 25 public games (20 tries each) (discussion/717133).
- The ARC post gives no scores. It says only that Reki and Murad used "pick a JSON action" agents, while Tufa used "agent-writes-code".

## 3. Milestone 2 (Sep 30): top placers
| Place | Team | LB on 2026-09-30 | Model and quantization |
|---|---|---|---|
| 1 | Daniel Franzen | **27.89** | Qwen3.8-Flash-Next, Intel W4A16 AutoRound, INT4 MTP draft, SGLang Pennyroyal v2.5.3 |
| 2 | lordhansolo | 23.84 | Qwen3.8-Flash-Next, mixed NVFP4+FP8, vLLM nightly |
| 3 | sirikilohit (rellik13) | 22.53 | Qwen3.8-Flash-Next, Intel W4A16, NVFP4 draft, SGLang |

- The table is from discussion/744792 and `competitors/tong-m2-comparison/kaggle/COMPARISON.md:12`. All three run the Duck harness on TAAF and serve Flash-Next.
- Franzen's notebook is a fork of Duck. His WRITEUP credits Tufa and Cottaar (`daniel-franzen-sol/WRITEUP.md:7`).
- Luck mattered. Unchanged copies of Franzen's notebook scored mean 25.77, SD 3.93, median 26.19 (discussion/745062). A copy in this repo's notes scored 27.80, then 21.01 on rerun (`research-sweep.md:73`).
- Tufa and Yi-Chia Chen did not take MS2. Tufa said it feared "the best of 4000 copies" and will release its solution after the competition ends (discussion/742801; also `research-sweep.md:64`).

## 4. Current public LB (verified)
- Kaggle CLI, 2026-10-05: **1. Tufa Labs 55.89** (154 submissions, last 10-03 07:17), **2. Yi-Chia Chen 48.59**, **3. mtg 37.54**, 4. gng 36.78, 5. Andrew Reed 36.14.
- This confirms `research-sweep.md:64`, which says Tufa leads at 55.89, Chen 48.59, and the third place 37.54.
- Franzen's 27.89 is now rank 333 of 3,720 teams, because 332 teams score higher, mostly copies.
- lordhansolo's best is now 28.32 (rank 288). rellik13 is at 22.53 (rank 684).

## 5. Model path
| Stage | Model | Evidence |
|---|---|---|
| MS1 | Qwen3.6-27B FP8 on vLLM | discussion/717133; ARC post ("Runs Qwen 3.6 27B FP8 locally") |
| Aug, after MS1 | Duck with Qwen3.8-27B, about 1.4 LB (public notebook 2.2) | discussion/736578 (Aug 21); discussion/743060 (1.43) |
| Late Aug to Sep | Qwen3.8-Flash-Next, released 2026-08-26 per https://gigazine.net/gsc_news/en/20260827-qwen3-8-flash-next/ | rellik13 first Flash-Next score 0.99 → 3.81 on Sep 2 (`competitors/rellik13-milestone2/WRITEUP.md:28-29`) |
| Franzen | 3.6-27B → 3.8-27B → Flash-Next, each "a substantial improvement" (per-switch deltas unknown). First on the LB "for a short moment" on 2026-09-06, then passed by Tufa. | `daniel-franzen-sol/WRITEUP.md:15,29` |
| Flash-Next serving | Franzen switched RadixArk NVFP4 to Intel W4A16 AutoRound shortly before the deadline. The PLE table is offloaded to host RAM. | `WRITEUP.md:31` |

- Qwen3.8-27B is said to have been released 2026-08-14. That comes from a search summary only and is unverified.

## 6. Compute and runtime rules
- **Hardware and time:** one RTX Pro 6000 (96 GB) with 9 hours of runtime to play 110 games in any order (discussion/717133). The notebook budget is 32,400 s (`COMPARISON.md:75`; the Franzen notebook asserts it at `dfranzen/arc-agi-3-milestone-2-solution.ipynb:11148`).
- **Same limits at both milestones:** MS2 is still a single RTX PRO 6000 (`WRITEUP.md:49`), so the gain came from using the same compute better.
- **Franzen's per-game cap:** a shared pool of 532×60 = 31,920 s across all 110 games (`ipynb:11143`; `COMPARISON.md:74`). This repo scales that to 121 min for the 25 public games (`daniel-franzen-sol/kaggle/README.md:27`).
- **No internet during evaluation, and open source required** under a permissive license (https://arcprize.org/competitions/2026, https://arcprize.org/competitions/2026/arc-agi-3).
- **No 5× human-action cap on Kaggle.** The compute limit is the only cap (discussion/713921; the cached copy does not show the replier's name).
- **Documentation mismatch:** one thread says the docs promise "at least 8 hours" (discussion/714128). I use the 9 hours from Tufa and the notebook assert.
- **Franzen's own compute:** mostly Kaggle, plus about 150 rented RTX PRO 6000 GPU-hours before the deadline. No training or fine-tuning (`WRITEUP.md:318`).

## 7. What changed between the stages (for the learning-trajectory question)
| | MS1 Tufa | MS2 Franzen |
|---|---|---|
| Context | cap 64k, keep ~32k, "do not optimally use prefix caching" (discussion/717133) | harness 131,072, server 139,264, trim to ~57k (`WRITEUP.md:51-54,97`) |
| Serving | vLLM | SGLang, FP8 KV cache, 10 streams, speculative decoding (`WRITEUP.md:47-59`) |
| Image | 4× upscale | 10× upscale, plus diff images (`WRITEUP.md:17,203`) |
| UNDO | withheld because the model "fails to use it efficiently" | exposed as `UNDO` (`WRITEUP.md:262`) |
| World model | text notes on | turned off (`WRITEUP.md:20,295`) |
| Scheduling | equal share per game | priority scheduler with calibrated levels (`WRITEUP.md:119-174`) |

- rellik13's dated climb shows serving and context dominating (`rellik13-milestone2/WRITEUP.md:28-35`): 0.99 (Aug 11) → 3.81 (Flash-Next) → 6.42 (SGLang) → 7.14 (bigger context) → 9.96 (solved-level memory) → 13.40 (Wang's harness) → 14.49 → 22.53 (FP8 KV and longer history, no prompt change). The last jump matches lordhansolo's forum comment on discussion/744792.
- Franzen reports that extra prompt text and summaries did not clearly help (`WRITEUP.md:291-305`). rellik13 saw added text hurt, 6.42 → 4.16 (`rellik13-milestone2/WRITEUP.md:59`).

## Open items
- Reki's exact MS1 score and the per-model deltas in Franzen's runs are unknown.
- I did not verify the ARC Prize MS2 tweet or the lordhansolo write-up on Kaggle, because both were unreachable.
- The Qwen3.8-27B release date is unverified.
