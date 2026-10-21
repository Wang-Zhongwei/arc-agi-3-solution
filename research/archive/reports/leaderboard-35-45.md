# How teams reach 35–45 on the public leaderboard (snapshot 2026-10-08)

Two research passes on 2026-10-08: a browser pass (leaderboard, discussion, Code tab) and a code pass (Kaggle CLI pulls of the top public notebooks, mechanical diffs against Franzen's template, the full leaderboard CSV with 3,992 teams). Verified facts are marked [V]; others' claims or our inferences are marked [C] or [I].

## Short answer
No public notebook, thread or writeup explains a 35–45 score. In order of weight:
1. **Best-of-N luck on Franzen copies** explains most of 33–36 [I]. Hundreds of one-shot copies of Franzen's notebook score mean 25.77, SD 3.93, range about 20–34 (Son Pham, discussion 745062) [C]. A noise model fitted to the real board predicts about 11 teams ≥35 against 15 observed, and about 2 ≥37 against 8. So there is a real excess of about 8–10 teams at 36–39, while Majkel1337 (42.66) and the two 55s (Tufa Labs 55.89, Yi-Chia Chen 55.77) are clearly real [I].
2. **Serving throughput, KV memory and concurrency** is the only mechanism with public evidence [V/C]:
   - Lord Han Solo (#11, 36.61): vLLM e975732 plus patches, a mixed NVFP4/FP8 model, 14 streams, a 1.42M-token KV pool, 1,135 tok/s peak (Franzen 946). "The score follows the number of turns each game gets."
   - rellik13 / sirikilohit: 16 streams, 1,159 tok/s. FP8 KV spent on longer history (trim 37k→27k became 57k→45k) took 14.5 → 22.5.
3. **Custom quantization or models**, private [V datasets; I effect]:
   - richardcsaky (#6, 38.62) published `arc-expert-gptq-arc-v1` (64 GB, "GPTQ-ARC routed-expert overlay") and `arc-expert-int3-g16-v1` (49 GB).
   - Yi-Chia Chen has public models `opd-32b-v33-s200-gptq-w4a16` and a DFlash drafter (on-policy distillation?).

## Leaderboard snapshot [V]
Ranks 1–17: Tufa Labs 55.89, Yi-Chia Chen 55.77, Majkel1337 42.66, the last dance 39.30, dreach.ai 39.11, artificialagencylab.com 38.62, mtg 38.33, 復活の混テキスト 38.15, NVARC3 37.51, _hans 37.07, Lord Han Solo 36.61, AI_hwlee 36.54, fshindo 36.16, YUTO KOJIMA 36.14, Andrew Reed 36.14, lalalia 35.79, Kamal Kadakara 35.44. #20 is 34.36, #50 33.09, #100 31.76, #500 28.51. We are about #648 at 27.62. The public LB uses about 55 of the 110 hidden games; the 25 public games are not among them. One submission per day.

## Top public notebooks [V]
Every top-25 Franzen fork keeps his harness patch and serving CFG byte for byte (except df-wm), and all score 29–32 on single submissions: within noise. A notebook page shows the best score of any submission of that version, so Franzen's page shows 34.3 while his own entry is 27.89.

| Notebook | Notebook score | What it changes |
|---|---|---|
| sujanmajhisuzan/arc-agi-3-m2-top-submission | 31.93 | Franzen's extra guards, auto frame diff, level inventory; solved-level invariants and RESET directives; D′ |
| leoprovorov/fable-astra (F001) | 31.73 | Fable layer (we measured on 29.93 vs off 29.22) |
| shiiin9/affectify | 31.54 | D′ priority only |
| sigeward/milestone-2-solution | 31.27 | Input checks, no-spec fallback, soft end 120 s before the 9 h limit |
| desihnurseha/kinetic-v8 | 29.68 | Hand-written mechanics catalogue for 19 games |
| vladimiryakunin/df-wm | 29.30 | Executable world model (simulate, check_sim, BFS search, act_checked) |
| hknight888/hknight3-0 | 28.52 | Byte-identical to Franzen, which shows the forks above are inside noise |

## Serving comparison [V] (`competitors/tong-m2-comparison/kaggle/SUMMARY.md`, rellik13 WRITEUP)

| | Franzen (us) | Lord Han Solo | rellik13 |
|---|---|---|---|
| Streams | 10 | 14 | 16 (28 games live) |
| Peak decode | 946 tok/s | 1,135 tok/s | 1,159 tok/s |
| KV pool | 1.011M tokens (memfrac 0.96) | 1.417M (GPU util 0.98) | memfrac 0.97 + 48 GB host cache tier |
| History trim | 118k → 59k | 126k → 81.5k | 57k → 45k |
| Yield | 2,048 tokens | 120–150 s, time-based | — |
| Temperature | 0.7 | 0.6 | 0.6 |

Lord Han Solo also uses a 32k draft vocabulary and patches for Mamba align-state retention and prompt-tail state caching. Our earlier runs: temp06 45.92 (no effect), streams14 46.38 (noise despite +15% tok/s).

## What did not help (controlled reports) [C]
- Sergei Makarov (743723): consensus sampling, a reasoning-discipline block, higher-res images and an **executable world model with replay validation** all scored at or below baseline. 83% of stuck runs had already tried the decisive action type; the failure is early goal judgement.
- gedouluhui (743060): temperature 0.3 collapsed exploration; 64k context cost throughput; an archetype playbook scored below baseline. Click-candidate perception hints plus a scoring-rule line gave about +28% on a ~1.4 base.
- Scott Le Grand: most added enhancements were net negative.

## Implications for us
- Judge arms only by paired offline runs at matched wall-clock; LB rank is a poor guide below about +8 points.
- Throughput and KV are the best-supported lever: `tput-safe` (PR #69) is running; new candidates are in `notes/arms.md` (backlog "From the leaderboard research").
- The world-model port must be tested against the negative reports above, not assumed to help.

Sources: leaderboard and Code tab pages; discussions 744792 (M2 comparison), 743723, 743060, 745062, 746010, 745951; writeups kaggle.com/writeups/lordhansolo/arc-agi-3-milestone-2, github.com/LohitSiriki/arc-agi-3-milestone2-solution, github.com/da-fr/arc-agi-3-solution. Pulled notebooks are in `arc-agi-v4/competitors/kaggle/`.
