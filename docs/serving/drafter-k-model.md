# Draft length model (DFlash2 on the RTX PRO 6000)

How many draft tokens k a block drafter should propose, from a closed-form model of throughput. The inputs are the
Kaggle measurements in [drafters.md](drafters.md); log paths starting with `runs/` are relative to `ARC3-Inference/`.

## Summary

- **k = 7, the most DFlash2's 8-token block allows, is the right k.** The model's optimum is k* = 9 (+0.2% over
k = 7) with the physical cost model, and k* = 28 (+3.6%, within 0.1% of that from k = 15) with the cost fitted to k = 4
and k = 7.
- **The throughput curve is flat above k ≈ 6.** Acceptance per extra position decays like p^(k+1) = 0.727^(k+1), and
the verify cost of an extra position decays like γ^(k+1) = 0.759^(k+1). The two are close, so neither wins fast.
- **k is not why DFlash2 k = 7 trails MTP-3.** At equal prefill per generated token, the model puts k = 7 level with
MTP-3 (889 vs 893 tok/s). The measured gap comes from prefill (lower prefix-hit rate) and the slow start.

## The model

### Acceptance

The draft at position i is accepted with probability p^i: each position survives with the same probability p, given
that all earlier ones were accepted. Tokens produced per sequence per step, the bonus token included:

```
L(k) = 1 + p + p^2 + ... + p^k = (1 - p^(k+1)) / (1 - p)
```

### Step time

The target verifies N = B·(k+1) rows per step: k drafts plus one bonus row for each of the B sequences.

Expert reads dominate a decode step on the RTX PRO 6000. Each row routes to a experts out of n_e. With routing
treated as uniform and independent, one row misses a given expert with probability q = 1 - a/n_e = 502/512 = 0.980.
The B rows of one draft position (one row per sequence) all miss it with probability

```
gamma = q^B = (1 - a/n_e)^B = 0.980^14 = 0.759
```

All N rows of a step miss it with probability gamma^(k+1). A step therefore touches n_e·(1 - gamma^(k+1)) experts
per layer, and reading their weights takes

```
expert time = C_E · (1 - gamma^(k+1)),   C_E = N_l · n_e · b_e / W
```

where C_E is the time to read every expert of every layer once.

The full step:

```
t(k) = c_pf·rho·B·L(k) + c_r·B·(k+1) + C_E·(1 - gamma^(k+1)) + F
```

The terms are listed in the order a step's work is usually described (prefill, then the verify rows, then the
expert reads). In vLLM the prefill chunks and the decode rows run in the same forward pass, so the terms add up
rather than run one after another.

| Term | What it is |
|---|---|
| c_pf·rho·B·L(k) | prefill: rho prefill tokens per generated token, c_pf seconds each |
| c_r·B·(k+1) | per-row cost outside the MoE (attention, GDN state, lm_head compute) |
| C_E·(1 - gamma^(k+1)) | routed-expert weight reads |
| F | fixed cost: attention and GDN weights, dense layers, kernel launches, and drafting (DFlash2: one pass, the same for every k) |

For MTP, drafting is k sequential passes, so F becomes F' + k·t_m.

### Throughput

```
T(k) = B · L(k) / t(k)

     =                  B · (1 - p^(k+1)) / (1 - p)
       ---------------------------------------------------------------------------------------
       c_pf·rho·B·(1 - p^(k+1)) / (1 - p) + c_r·B·(k+1) + C_E·(1 - gamma^(k+1)) + F
```

Dividing through by B·L(k):

```
1 / T(k) = c_pf·rho  +  [c_r·B·(k+1) + C_E·(1 - gamma^(k+1)) + F] / (B·L(k))
```

The prefill term is a constant per generated token. It sets the level of T but not the best k.

### Best k

Setting dT/dk = 0 (D(k) = t(k) without the prefill term):

```
p^(k+1) · ln(1/p)          c_r·B + C_E · ln(1/gamma) · gamma^(k+1)
-----------------    =    ----------------------------------------
  1 - p^(k+1)                              D(k)
```

The left side is the relative gain in tokens per step from one more position; it decays like p^(k+1). The right
side is the relative growth in step time; with c_r = 0 it decays like gamma^(k+1). A finite optimum exists only
because p < gamma (0.727 < 0.759). The two are close, so the optimum lies far out and the curve is flat around it.

The optimum moves out when:
- p rises (better acceptance, e.g. fine-tuning the drafter);
- B rises (more streams: gamma falls, so the expert reads saturate sooner);
- c_r rises relative to C_E (pulls it back: per-row costs do not saturate).

## Values

| Symbol | Meaning | Value | Source |
|---|---|---|---|
| p | per-position acceptance | 0.727 | least-squares fit of log r_i = i·log p to dflash2-k7 v3's per-position rates |
| B | concurrent sequences | 14 | `max_num_seqs` |
| n_e, a | routed experts per layer, experts per token | 512, 10 | target `config.json` |
| N_l | MoE layers | 48 | target `config.json` (12 full attention + 36 linear attention) |
| b_e | bytes per routed expert | 2.53 MB | 3 × 2560 × 640 INT4 weights + FP16 scales per group of 128 |
| W | HBM bandwidth | 1.79 TB/s | RTX PRO 6000 |
| C_E | read every expert once | 34.7 ms | N_l · n_e · b_e / W |
| gamma | fraction of experts one draft position's rows leave untouched | 0.759 | (1 - 10/512)^14 |
| c_pf·rho | prefill per generated token | 216 µs | 150 prefill tokens per step × 68 µs, over B·L(7) tokens (k = 7, minutes 2–10) |
| F, c_r | fixed cost, per-row cost | calibrated below | |

The fitted p reproduces the measured rates:

| Position | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| measured (dflash2-k7 v3, whole run) | .740 | .529 | .376 | .270 | .197 | .148 | .111 |
| p^i, p = 0.727 | .727 | .528 | .384 | .279 | .202 | .147 | .107 |

### Calibration

With c_r = 0, F and C_E are set by the measured decode parts: 42.9 ms at k = 7 and 40.8 ms at k = 4 (minutes 2–10,
prefill subtracted at 68 µs per token; [k = 4 vs k = 7](drafters.md#k--4-vs-k--7)). Two calibrations bound the truth:

| Calibration | C_E | F | Decode part at k = 4 | Comment |
|---|---|---|---|---|
| physical | 34.7 ms (from the values above) | 12.0 ms | 38.0 ms (measured 40.8) | every touched expert costs a full read; likely too steep |
| fitted | 14.8 ms (through k = 4 and k = 7) | 29.7 ms | 40.8 ms | the k = 4 point is the BF16-drafter-KV run, whose KV was 98% full; likely too shallow |

Real routing is not uniform: popular experts are shared across rows, so fewer distinct experts are touched than
n_e·(1 - gamma^(k+1)). That flattens the cost curve, as in the fitted row.

## Results

T(k) in tok/s, relative to k = 7:

| k | L(k) | Physical: decode part | Physical: T | Fitted: decode part | Fitted: T |
|---|---|---|---|---|---|
| 1 | 1.727 | 26.7 ms | 757 (−14.9%) | 36.0 ms | 587 (−34.1%) |
| 2 | 2.254 | 31.5 ms | 823 (−7.5%) | 38.1 ms | 703 (−20.9%) |
| 3 | 2.638 | 35.2 ms | 855 (−3.8%) | 39.6 ms | 776 (−12.7%) |
| 4 | 2.917 | 38.0 ms | 872 (−1.9%) | 40.8 ms | 823 (−7.5%) |
| 5 | 3.119 | 40.1 ms | 882 (−0.8%) | 41.7 ms | 854 (−4.0%) |
| 6 | 3.266 | 41.7 ms | 887 (−0.3%) | 42.4 ms | 875 (−1.6%) |
| **7** | 3.373 | 42.9 ms | **889** | 42.9 ms | **889** |
| 9 | 3.507 | 44.5 ms | **891 (+0.2%, k\*)** | 43.6 ms | 906 (+1.9%) |
| 20 | 3.653 | 46.6 ms | 887 (−0.3%) | 44.5 ms | 921 (+3.5%; k\* = 28: +3.6%) |

- **MTP-3 in the same model:** v6's decode part of 34.7 ms and L = 2.743 at the same 216 µs prefill per token give
893 tok/s, level with DFlash2 k = 7.
- **Against Kaggle:** dflash2-k7 v3 measured 825 tok/s over its first 30 minutes and v6 918. The difference from the
model is prefill: k = 7 prefilled 199 tokens per step over the whole run (prefix hits 88–89% against v6's 92–93%),
and its first minutes ran with low hit rates.
- **k > 7 needs a drafter trained with a larger block.** incoai's checkpoint has `block_size` 8.

## Limits

- Uniform routing overstates the experts touched. The physical calibration is therefore an upper bound on how fast
the cost grows with k.
- c_r is not separable from F and C_E with two measured points, so it is set to 0. A positive c_r lowers k*.
- Geometric acceptance (a constant p) fits these seven positions, but a fine-tuned drafter may change the shape.
- Each calibration point is one Kaggle run. These are step-time comparisons, not scores.

## Reproduce

```sh
python3 - <<'EOF'
import math
r = [.740, .529, .376, .270, .197, .148, .111]  # drafter_stats accept runs/kaggle/jonathanwang2022/dflash2-k7/v03/vllm-openai-server.log
p = math.exp(sum((i + 1) * math.log(x) for i, x in enumerate(r)) / sum((i + 1) ** 2 for i in range(7)))
B, n_e, a, N_l, b_e, W = 14, 512, 10, 48, 2.53e6, 1.79e12
gamma, C_E = (1 - a / n_e) ** B, N_l * n_e * b_e / W * 1e3     # C_E in ms
L = lambda k: (1 - p ** (k + 1)) / (1 - p)
c_pf_rho = 150 * 0.068 / (B * L(7))                             # ms per generated token
for C in (C_E, (42.9 - 40.8) / (gamma ** 5 - gamma ** 8)):      # physical, fitted
    F = 42.9 - C * (1 - gamma ** 8)
    T = lambda k: 1e3 * B * L(k) / (F + C * (1 - gamma ** (k + 1)) + c_pf_rho * B * L(k))
    print(f"C_E {C:.1f} ms, F {F:.1f} ms, k* {max(range(1, 41), key=T)}:",
          " ".join(f"{k}:{T(k):.0f}" for k in (3, 4, 5, 7, 9, 20)))
print("MTP-3:", round(1e3 * B * 2.743 / (34.7 + c_pf_rho * B * 2.743)))
EOF
```
