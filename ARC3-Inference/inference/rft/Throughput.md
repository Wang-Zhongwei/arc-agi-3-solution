# 训练吞吐 10× 计划

目标：64K 样本一步从 137 s 降到约 14 s，即 474 → 约 4,700 token/s。一个 epoch（2,160 万 token）从 12.7 h 降到约 1.3 h。现状和每次 GPU 运行的记录见 [README](README.md) 和 [Logs.md](Logs.md)。

## 为什么 10× 有可能

- 每个 token 约 40 GFLOP：激活参数约 6.0B（GDN 2.08B、路由专家 2.36B、hyper-connection 0.63B、attention 0.6B、shared expert 0.24B），冻结基座加一次重算是 6 × 6.0B = 36 GFLOP，再加 QSA 注意力、indexer、GDN 递推和 lm_head 约 4 GFLOP。
- 64K 一步约 2.6 PFLOP。14 s 跑完需要约 190 TFLOP/s，是 H200 BF16 峰值（约 990）的 19%。[Yi-Chia Chen](https://huggingface.co/blog/ycchen/agent-factory-3-en) 的训练端按他的端到端吞吐推算约 18%。我们现在只有 2%。
- 矩阵乘本身已经不慢：单层 profile 里 MoE 的 grouped GEMM 约 300 TFLOP/s。137 s 里矩阵乘只有约 10 s，其余都是矩阵乘以外的开销。10× 就是把这些开销去掉。

## 进展

| 日期 | 64K 每步 | Token/s | 改了什么 |
|---|---|---|---|
| 10-18 | 137 s | 474 | 起点 |
| 10-19 | 83 s | 784 | 第 1 步（复用、锁页、异步拷贝、留 GPU）+ 第 2 步（parts） |
| 10-19 下午 | 约 24 s（估算） | 约 2,700（估算） | 第 3–5 步的大部分，单层基准估算，见下节；还没在 H200 上端到端量过 |
| 10-20 凌晨 | — | — | RTX PRO 6000 演练，32K（33,811 token）：旧路径 67.1 s → 新路径 20.6–23.7 s（2.8–3.3×）→ 再加前缀不回传梯度 15.8 s（前缀 49.5%，1.50×），见下节 |
| 10-20 凌晨 | **22.6 s** | **2,874** | H200 上量的全部优化（6.1×）；同一台卡上午的路径 80.0 s（811） |
| 10-20 凌晨 | **16.1 s** | **4,031** | 再加前缀不回传梯度（这个样本前缀 80%，8.5×）；这一步是 OOM 后退回 layer 模式跑的，parts 模式估计约 10 s |
| 10-26 | **20.3 s** | **3,196** | 完整反向（GRPO 的情形）：QSA 的 dK/dV 改回逐 query（`UNION_DKV` 关），单层 QSA 反向 343.5 → 238.9 ms；整步实测见下两节 |
| 10-26 | 110K：**36.9 s** | **2,878** | GRPO 中位长度（106,078 token），layer 模式，完整反向 |

### GRPO 形状的整步（10-26，H200 b5nxpzy3e86lk3，约 22 分钟，约 $1.35）

`full_run.sh` 加 `GRPO=1`（`make_jobs.py --grpo`）：每条回复都训练，没有可跳的前缀。全部 CUDA 测试通过，没有关掉任何功能。

| | 一步 | token/s | 模式 | 峰值显存 |
|---|---|---|---|---|
| 64K（64,893），热的两步 | 20.4 / 20.3 s | 3,175 / 3,196 | parts，GPU 上留 33–37 GB | 127–133 GB |
| 110K（106,078），热的两步 | 36.9 / 38.6 s | 2,878 / 2,750 | layer，GPU 上留 9–13 GB | 127–131 GB |

- 对 10-20 的 22.6 s（2,874）快 11%，和单层基准估的 −1.26 s 一致。
- 110K 每 token 只比 64K 慢约 12%：GDN 层 22.8 s（36 层），QSA 层 13.5 s（12 层，占 37%）。
- 8K 层对照（真实输入，对 transformers 的层、QSA 和 AutoRound 专家）：QSA 层（逐 query dK/dV）LoRA 梯度余弦 0.99984、输入梯度 0.99997；GDN 层 0.99977 / 0.99983。和 10-19 的标准（0.9996+）一致。
- 110K 的单层 profile 在同一个服务进程里跑完 110K 的步之后 OOM（137 GiB），没有数。
- 第一次开的 pod（ahqxsbxphw414k）在装包阶段 abort，`POD_END=delete` 让它连日志一起没了（约 $0.25）。很可能是没钉版本：flash-linear-attention 0.6.0 是 10-23 才发的。之后 pod 的包从 [pod/requirements-pod.txt](pod/requirements-pod.txt) 装，abort 在删 pod 前等 3 分钟让 watcher 拉日志。

### QSA dK/dV 拆分（10-26，H200 2a6lrwuu898ipd，约 9 分钟，约 $0.5）

[bench_qsa.py](bench_qsa.py)：一层 QSA 的真实尺寸（24 个 query head、2 个 KV head、D = 256、4 token 一块、top 512），64,893 token，`realistic_blocks` 的选择（16 个 query 并集 3.47 倍、最近 4,096 token 占 44%）。dK/dV kernel 拷一份，写法可切换：原子加、普通写（结果不对，写的字节一样）、不写（只留校验和）。拷贝的原子加版本和线上 kernel 对照，相对误差 2.5e-5。

| 一层 QSA，64K | ms |
|---|---|
| 前向（一次；一步跑两次：前向 + 重算） | 42.75 |
| dQ kernel | 45.63 |
| 并集表 `union_tiles` | 17.25 |
| 并集 dK/dV：原子加（线上） | 276.44 |
| 并集 dK/dV：普通写 | 277.93 |
| 并集 dK/dV：不写 | 206.28 |
| **整个反向，并集（dQ + 并集表 + dK/dV）** | **343.5** |
| **整个反向，逐 query（dQ、dK、dV 一个 kernel，每元素一次原子加）** | **238.9** |

- 原子加不比普通写慢（−1.5 ms）：原先以为并集省下的原子加是关键，H200 上不成立。并集 kernel 的开销是白算（21.0 TFLOP 里有用的 6.4 TFLOP，3.3 倍）加写回（约 72 ms）。
- 所以去原子加的两段式方案不做了；逐 query 的版本早就有、测试也在，只把默认值改了。整步的收益还要在下一次带模型的运行上确认（同时量 110K 的完整反向）。
- 110K 的同一基准在 170 s 的窗口里没跑完（选择生成和 autotune 更慢），没有数。
- 同一台 pod 上 `test_qsa_kernel.py` + `test_qsa_union.py` 10 个测试全过（对 FP32 稠密参照，两种 dK/dV 都测）。

### B200 对 H200（10-20 凌晨，B200 uas538q7c7lapw，secure US-NE-1，$7.99/h，约 24 分钟，约 $3.2）

[结果](../../runs/runpod/jonathanwang2022/rft-full-run/uas538q7c7lapw/)。只比默认配置（全部优化 + 前缀不回传梯度）的吞吐和最长样本会不会 OOM。

| 64K（64,893，前缀 80%），热的一步 | 一步 | token/s | 价格 | 每美元 token |
|---|---|---|---|---|
| H200（layer 模式，留 GPU 39.7 GB） | 16.1 s | 4,031 | secure $5.29 / community $3.59 | 2.74M / 4.04M |
| B200（parts 模式，留 GPU 78.3 GB） | 17.2 s | 3,772 | secure $7.99 / community $5.98 | 1.70M / 2.27M |

- 结论：H200 更划算。B200 这一步反而慢一点（前向 13.7 s 对 7.0 s，反向 3.5 s 对 9.1 s），价格却是 1.5 倍。B200 的前向被自动留 GPU 的 78 GB 拖慢（H200 上留得多时 GDN 前向也慢 1.7×），就算修好、按反向的优势估到 10–11 s，每美元 token 也只和 secure H200 持平，比 community H200 差。另外这套软件对 sm100 支持不好：torch 2.8 在 B200 上没有 `torch._grouped_mm`（走 Triton grouped GEMM），fla 的 GDN 反向要关掉流水线才能编译（a301c3a）。
- 最长样本（ft09，122,943 token，前缀 87.5%）：两张卡都不 OOM，parts 模式峰值 H200 111.9 GB、B200 111.8 GB（第一遍，不留 GPU）。B200 冷启动一步 38.7 s。
- 旧路径 64K：B200 67.7 s，H200 80.0 s。
- 逐层对照 B200 LoRA 梯度余弦 0.99973–0.99979（H200 0.99977–0.99983）。B200 上前缀不回传梯度两遍的 loss 差 0.6%（0.2789 / 0.2773；H200 上两遍一样），原因没查。

### H200 一次跑完（10-20 凌晨，4haw3pwub9vce4，secure US-NC-1，$5.29/h，约 36 分钟，约 $3.2）

[结果](../../runs/runpod/jonathanwang2022/rft-full-run/4haw3pwub9vce4/)。B200 没货。

| 64K（64,893 token），第二步 | 前向 | 反向 | 一步 | token/s | 对 10-18（137 s） |
|---|---|---|---|---|---|
| 上午的路径 | 18.0 s | 62.1 s | 80.0 s | 811 | 1.7× |
| 全部优化 | 6.7 s | 15.9 s | **22.6 s** | 2,874 | **6.1×** |
| + 前缀不回传梯度（前缀 52,033 = 80%，layer 模式） | 7.0 s | 9.1 s | **16.1 s** | 4,031 | **8.5×** |

- 32K（33,811）全部优化：12.6 s（2,675 tok/s）。
- 123K（122,943，ft09）：全部优化在 parts 模式仍然 OOM（fla 的 GDN 前向，在 1.41 GiB 的分配上），退回 layer 模式 183.8 s；前缀不回传梯度（前缀 87.5%）parts 模式放得下（峰值 111.9 GB），反向只要 6.9 s。两个都只跑了一遍，是冷启动（新长度的编译、主机缓冲增长），不能当速度用。
- 正确性：逐层对照 LoRA 梯度余弦 0.99977–0.99983（比 RTX PRO 6000 的 0.9996–0.9997 还好）；同一路径两遍 0.99993；前缀不回传梯度 parts 和 layer 两遍 0.99994；它和完整梯度余弦 0.689（前缀 80%，32K 前缀 49.5% 时是 0.78）。
- 前缀不回传梯度第二遍 OOM：`gpu_keep_gb: "auto"` 按第一遍的峰值（92 GB）留了 39.7 GB checkpoint 在 GPU，实际顶到 139.8 GB；重试（layer 模式）成功、没有泄漏。全部优化第三遍留了 46 GB 时 GDN 前向从 4.0 s 变成 7.0 s（峰值 132.7 GB）。留在 GPU 的量要留更多余量，是下一步。
- Hopper 上开头 4 个 CUDA 测试先失败：FP32 的参照模型被 transformers 送进 `torch._grouped_mm`（只收 BF16）。修好后在 H200 上重跑全部通过（包括 fast_gdn 在 Hopper 的 tilelang 反向、前缀不回传梯度），在第一个 job 开始前重新生成了 job（23119f7）。
- profile（64K，一层）：QSA 层里 QSA kernel 0.40 s 是大头；GDN 层剩下的主要是拷贝/拼接和逐元素运算（每层 0.08–0.18 s）。

### 演练（RTX PRO 6000，10-20 凌晨，dshsgfhx2rzdn3，约 43 分钟，约 $1.8）

[full_run.sh](pod/full_run.sh) 的完整流程（网络检查、装包、CUDA 测试、按测试结果生成 job、模型服务、结果、自删）先在便宜的卡上走一遍，长度减半（32K/16K，不跑 123K）。[结果](../../runs/runpod/jonathanwang2022/rft-full-run/dshsgfhx2rzdn3/)。

| 32K，第二步 | 前向 | 反向 | 一步 | 峰值显存 | LoRA 梯度（同一路径两遍） |
|---|---|---|---|---|---|
| 旧路径（上午的） | 23.9 s | 43.2 s | 67.1 s | 82.6 GB | — |
| 新路径（全部开关） | 6.8–9.1 s | 13.8–14.7 s | 20.6–23.7 s | 85.7 GB | 余弦 0.99993 |
| 新路径 + 前缀不回传梯度（前缀 16,736 / 33,811） | 7.4 s | 8.4 s | 15.8 s | 85.0 GB | 余弦 0.99994 |

- 逐层对照（8K，层 0/1/3，对 transformers 层 + AutoRound 专家 + transformers QSA）：LoRA 梯度余弦 0.9996–0.9997，和 10-19 一样。
- 前缀不回传梯度：1.50× 和按 3.5 份成本估算的 1.55×（前缀 49.5%）一致；转写里前缀平均 65%，估算 1.85×。它和完整梯度的余弦是 0.78：这是截断本身，不是舍入（同一路径两遍 0.99994），所以还需要训练 A/B。
- 找到并修好的问题：前缀的 GDN 短卷积状态是投影输出的视图，把整段前缀的投影（32K 时 12.1 GiB）留到反向，32K OOM（7127315）；OOM 后在 except 里重试，失败那一步的激活一直占着显存，后面的 job 全部 OOM；layer_check 留下的梯度算进了下一步的范数（341 而不是 0.38）（95dd435）。
- CUDA 测试 `test_prefix_stop_grad_cuda` 原来失败（0.063），是小模型 top-3/8 路由在近似并列时翻转，不是代码错；fla GDN kernel 用 TF32 点积，`dt_bias`/`A_log` 这种对所有 token 求和、互相抵消的梯度对 1e-7 的输入扰动就会变 2%（3138304）。
- pod 自删在活的 pod 上验证过（`pod removed`，REST 返回 404）。

### 真实模型整步（RTX PRO 6000 96GB，sm120，第二步）

H200 当天不是没货就是网络太差（冰岛 EUR-IS-4：HF 12 MB/s、PyPI 223 KB/s），按你的意思用便宜的卡证明。这张卡没有 `torch._grouped_mm`，旧路径的专家是逐个 matmul。

| 路径 | 16K（14,173 token） | 32K（33,811 token） |
|---|---|---|
| 上午的路径（分块 QSA，每块反量化） | 572 tok/s | 569 tok/s |
| QSA kernel + 专家缓存 + 编译 + 短卷积 | 1,101 | 1,081（带 keep_mixer） |
| + Triton grouped GEMM | **1,314** | **1,475**（2.6×） |

- 正确性：真实输入的逐层对照（8K，层 0/1/3，对 transformers 层 + AutoRound 专家 + transformers QSA 注意力，同一个选择）：LoRA 梯度余弦 0.9996–0.9997、输入梯度 0.9999，和上午的路径一样；整模型梯度对比分不出对错（见 Logs.md）。
- `keep_mixer`（QSA mixer 不重算）逐位正确，但 16K 反而慢 0.4 s（激活要 offload），没有提交。

### 单层基准（A100，64K，[bench_layer.py](bench_layer.py)）

不加载模型：用真实尺寸、随机权重的一个 GDN 层和一个 QSA 层（专家是打包的 INT4，按 AutoRound 的方式反量化，带 LoRA），量一步里每层做的事（checkpoint 下的前向，再加重算和反向）。整步按 36 个 GDN 层 + 12 个 QSA 层估算。随机权重让相邻 query 的选择几乎互不相关（16 个 query 的并集约 10 倍，真实模型约 3.5 倍），所以 QSA 层改用和真实统计一致的合成选择（最近 4,096 个 token 占 44%，16 个 query 并集 3.46 倍），indexer 照常运行以计入它的时间。

旧路径算出的 227 s 对应 H200 实测的 83 s，比例约 2.7。A100 上 MoE 的 grouped GEMM 是逐专家的循环（`torch._grouped_mm` 要 SM90），H200 上这部分更快，所以下表的比例在 H200 上只会更好。

| 改动 | GDN 层 | QSA 层 | 整步（A100 估算） |
|---|---|---|---|
| 旧路径（分块 gather + SDPA + `index_add`） | 2.11 s | 12.62 s | 227 s |
| 第 3 步：QSA 融合 kernel（每个 query 一个程序，dK/dV 用 FP32 原子加） | 2.12 s | 3.78 s | 122 s |
| 第 5 步：每层专家只反量化一次（缓存一层的 BF16 专家，约 5 GB），`split` 切块 | 1.65 s | 3.31 s | 99 s |
| 第 4 步：hyper-connection 和 MoE 合并用 `torch.compile` | 1.43 s | 3.11 s | 89 s |
| 512 个专家一次 grouped GEMM（不同步主机），gather 的反向不用 `index_put` | 1.40 s | — | — |
| 第 3 步：dK/dV 按 16 个相邻 query 的并集算（合成选择下） | — | 2.32 s | 78 s |
| 第 6 步：GDN 短卷积按通道在后计算（不转置），A100 的 grouped GEMM 只算 dX | **1.03 s** | **2.27 s** | **64 s** |

- 编译用静态形状：块长补到 2 的幂（512–8,192），每块的重算和前向走同一份图。全动态的图慢约 12%（GDN 层 1.17 s）。
- QSA kernel 现在 0.86 s（前向两遍各 0.085 s、dQ 0.09 s、dK/dV 0.6 s）。dK/dV 逐 query 原子加是 1.26 s。
- 剩下的大头：A100 上的 GEMM（GDN 层 0.44 s，H200 上应低得多）、QSA kernel、GPU 空等（GDN 层 1.03 s 里 GPU 只忙 0.81 s）。

## 时间都花在哪（64K，137 s）

按第三台 pod 的单层 profile 和逐层计时估算，第 0 步会实测。

| 类别 | 现在 | 目标 | 步骤 |
|---|---|---|---|
| 主机拷贝：写新页面的缺页（D2H）和同步的 H2D | ~41 s | ~1 s | 1 |
| QSA 注意力：gather、`index_add`、每次只有 12 行的注意力 kernel | ~40 s | ~3 s | 3 |
| hyper-connection、norm、注入等逐元素运算和拷贝 | ~20 s | ~2 s | 4 |
| MoE 的 token 排序/还原拷贝和反量化 | ~12 s | ~1.5 s | 5 |
| 矩阵乘 | ~10 s | ~8 s | 2 |
| 其他：GDN 递推、短卷积、PLE、视觉、CE、Python 开销 | ~14 s | ~2.5 s | 6 |
| 合计 | 137 s | ~18 s | |

只靠第 1–6 步预计到约 18 s（约 3,500 token/s，7.5×）。剩下的 4 s 要靠后面的备用杠杆。

## Chen 的做法给的启发

1. 冻结的量化部分，按原生格式写反向 kernel。他给 MXFP4 专家写了反向，没有反量化成 BF16 再用通用 GEMM。我们是后者：每个 8K token 块、每一遍重算都把 512 个专家反量化成 BF16 再调 `grouped_mm`。反量化本身每步只占约 3 s，更大的开销是 token 排序和还原的拷贝。
2. 现成的能用就用，只写模型特有的部分。他用 Liger 的融合 kernel。我们的结构接近 DeepSeek 那一套：hyper-connection、n-gram 记忆（PLE 类似 Engram）、按 token 选 top-k 的稀疏注意力。DeepSeek 的 [TileKernels](https://github.com/deepseek-ai/TileKernels) 有 mHC 和 Engram 的融合 kernel（带反向），NeMo AutoModel 有 TileLang 的 [sparse MLA 前向和反向](https://docs.nvidia.com/nemo/automodel/nemo-automodel/nemo_automodel/components/models/deepseek_v4/kernels/tilelang_sparse_mla_bwd)。
3. 他不需要 offload，我们需要，原因在残差宽度。GPT-OSS 的残差是 2,880 宽，我们的 4 路 hyper-connection 让每层 checkpoint 10,240 宽。能做的是让拷贝不阻塞计算。
4. 每个 kernel 都对照参考实现，并在启动时自检。他踩过 MXFP4 字节恰好是 BF16 NaN、拷贝时被规范化的坑。我们沿用 `check_experts` 的做法：启动时比对，超出误差就退回原实现。

## 分步计划

每一步只解决一个问题，做完先在 H200 上量速度和梯度，再进下一步。“预计”是累计的 64K 每步时间。

### 第 0 步：基线和梯度参照

- 问题：同一个 64K 步，两种 gather 的梯度范数差 18%。不先定下哪个是对的，后面的 kernel 就没法验收。
- 做法：QSA 的 K/V 梯度可选 FP32 累加（块内和块间都用 FP32），作为参照。保存每一步的 LoRA 梯度，比较逐模块的余弦相似度和范数差。同一步重复 3 次，量出噪声下限。offload 的拷贝单独计时。
- 验收：知道 18% 来自 BF16 累加还是别的原因；定下以后每一步的梯度容差。

### 第 1 步：主机拷贝不阻塞计算（预计 ~97 s）

- 问题：checkpoint 写进新分配的 pageable 内存时缺页，D2H 只有 1.7 GiB/s（约 35 s）；H2D 是同步的（约 6 s）。
- 做法：
  1. 复用主机 buffer（[host_offload.py](host_offload.py)，已完成）。
  2. 显存够的时候，后面几层的 checkpoint 留在 GPU。64K 峰值只有 99.3 GiB，还能留约 30 GiB。
  3. 把 arena 用 `cudaHostRegister` 锁页，拷贝放到单独的 stream 上：前向 D2H 和下一层计算重叠，反向提前预取下一层要用的 checkpoint。
- 验收：前向里不再有可见的拷贝时间；123K 仍然放得下。
- 结果（10-19）：前向 59 → 16.8 s，基本就是计算时间。

### 第 2 步：每部分只重算一次（预计 ~89 s）

- 问题：逐 token 部分（hyper-connection、router、MoE、shared expert）的分块 checkpoint 套在层级 checkpoint 里面，前向算了三遍。
- 做法：这一层不用层级 checkpoint，改为 mixer 和逐 token 部分各自 checkpoint。checkpoint 多存每层的 mixer 输入和输出（N × 2,560 各一份），相同的张量只 offload 一次。QSA 注意力的三遍留到第 3 步解决。
- 验收：小模型上 logits 和梯度完全一致，MoE 每步只算两遍。
- 结果（10-19）：小模型逐位一致；64K 显存峰值低 4.5 GiB。123K 在反向 OOM（一块 4.69 GiB 的 FP32 张量），原因待查，先退回 layer 模式。

### 第 3 步：QSA 稀疏注意力 kernel（预计 ~52 s）

- 问题：每个 query 选的 key 不同，现在按 query 把 key gather 出来再调 SDPA，每次只有 12 行（一个 GQA 组），tensor core 大多空着；K/V 梯度用 BF16 的 `index_add` 散回去（单层 1.19 s）。
- 做法：一个前向 + 反向的融合 kernel。按索引直接读 K/V，在线 softmax，前向只存输出和 LSE；反向在 kernel 里重算，dQ 直接写，dK/dV 在 kernel 里用 FP32 原子加（顺带解决第 0 步的累加精度）。重算时直接用前向选出的索引，不再算第二遍 indexer。起点是 NeMo/slime 的 TileLang sparse MLA kernel（输入是 [S, topk] 的 token 索引，和我们展开后的格式一样），改成 2 个 KV head、每组 12 个 query head、D = 256。动手前先统计相邻 query 的选择重叠：重叠大就按 query 块取并集，用更大的 tile。实测（10-19）：16 个相邻 query 的并集约是单个 query 的 3.8 倍，按 16 个一块取并集，K/V 读取量降到约 1/4，每次矩阵乘 16 × 12 = 192 行。
- 模板的细节（DeepSeek-V4 的 [sparse MQA 反向](https://docs.nvidia.com/nemo/automodel/nemo-automodel/nemo_automodel/components/models/deepseek_v4/kernels/tilelang_sparse_mla_bwd)）：每个程序处理一个 query 和一组最多 64 个 head；top-k 索引每 32 个一块 gather；dQ 在程序内累加后写一次；dKV 用 FP32 `atomic_addx4`，最后转 BF16；先算 Delta = rowsum(O · dO)。要改的地方：
  1. 它是 MLA，K 和 V 是同一个张量；QSA 要分开 gather K 和 V，dK = dPᵀQ、dV = PᵀdO 各一个 FP32 累加器。
  2. 它只有一个 KV head；我们有 2 个，每个带 12 个 query head，按 KV head 各跑一份（grid 多一维）。
  3. 12 个 head 要补到 16，它不屏蔽补出来的 head，要自己补零。
  4. D = 256 可以直接用（只要求能被 4 整除）。
- 验收：单个 QSA 层前向 + 反向从 3.6 s 降到 ≤ 0.3 s；输出、LSE 和梯度对照 FP32 参照。

### 第 4 步：hyper-connection 融合（预计 ~38 s）

- 问题：GDN 层 0.68 s 里矩阵乘只占约 20%，其余是 `copy_`、`mul`、`cat` 和逐元素运算。每层两个 gated residual，每个都对 10,240 宽的数据做 FP32 分组 RMSNorm、低秩混合、sigmoid、4 路加权平均，然后广播注入。
- 做法：先把 hyper-connection 写成纯函数，用 `torch.compile` 试一次（48 层共用一个编译结果），看能融合多少。不够的部分手写 kernel：一次读入 10,240 宽算完 norm 和混合，直接输出 2,560 宽；注入合成一次乘加；配反向。参考 TileKernels 的 mHC kernel（我们的公式没有 Sinkhorn，只能参考结构）。
- 验收：GDN 层前向 + 反向 ≤ 0.3 s；对照原实现一致。

### 第 5 步：MoE 数据搬运（预计 ~30 s）

- 问题：token 排序、按专家复制、还原（`index`、`cat`、`index_add`）都是显式拷贝；每个 8K 块都要把 512 个专家各反量化一遍。
- 做法：MoE 写成自己的 autograd 函数，按专家组处理整条序列：每组专家每一遍只反量化一次，排序后的读取和加权还原融合进 GEMM 前后的 kernel，SwiGLU 融合；反向按组重算。如果反量化还显眼，再写直接读 INT4 的 grouped GEMM（前向和 dX），这就是 Chen 给 MXFP4 做的事。
- 验收：MoE 部分只剩矩阵乘时间；`check_experts` 式启动自检。

### 第 6 步：剩下的小项（预计 ~18 s）

- 从源码编译 causal-conv1d，替换短卷积的参考实现。
- PLE 查表放到 CPU 线程里，和上一个样本的 GPU 计算重叠。
- 去掉剩下的 Python 循环和小 kernel，必要时对固定形状的块用 CUDA graph。
- 验收：64K 每步 ≤ 20 s。


### 备用杠杆（第 5 步之后决定）

1. 冻结权重的矩阵乘用 FP8：H200 FP8 峰值是 BF16 的 2 倍，矩阵乘 ~8 s → ~4.5 s。代价是数值和线上 W4A16 不完全一样，要看梯度和留出 loss。
2. 第一个训练 token 之前的上下文只做前向，不回传梯度：这部分不存 checkpoint、不重算、不反向。代价是 LoRA 学不到它对上下文表示的影响，要 A/B 留出 loss。先从数据集统计这部分占多少 token。

## 待你决定（10-19 晚）

按影响从大到小。前两项会改变梯度，要先 A/B；其余是取舍。

| # | 决定 | 选项 | 我的建议 | 依据 |
|---|---|---|---|---|
| 1 | 前缀不回传梯度：第一个训练 token 之前的上下文只做前向 | 做 A/B / 不做 | 已定（10-19 晚）：做。统计能省约 64% 的 token，一步约快 1.85×；实现后用真实样本的 LoRA 梯度余弦和短训练做 A/B。已实现（10-20）：32K 实测 1.50×（前缀 49.5%），和完整梯度余弦 0.78，A/B 等训练循环 | 代价是 LoRA 学不到怎么编码上下文（上下文是不是模型生成的不影响这一点） |
| 2 | 冻结权重的矩阵乘用 FP8 | 做 / 暂缓 | 暂缓 | H200 上矩阵乘只占一步的一小部分，估计省 1–2 s；数值和线上 W4A16 不再一致，也要 A/B |
| 3 | 晚上的验证用什么卡、花多少 | H200（secure US-NC-1，$5.29/h，网络好）/ B200（$7.99/h）/ 两个都测 | H200，约 30 分钟、$3–5；B200 有货就顺便测一次 | 冰岛的 H200 网络太慢不能用（HF 12 MB/s）。B200 显存 180 GB，能少 offload，123K 可能直接放下 |
| 4 | 正式训练用什么卡 | H200 / B200 / RTX PRO 6000 | 已定（10-19 晚）：B200 的提速倍数超过它的价格倍数（约 1.5–1.7×）就用 B200，显存大也省掉 OOM 的麻烦；两种卡都测，按每美元 token 数选 | RTX PRO 6000（96 GB，$1.69–2.49/h）每美元吞吐和 H200 差不多，但 64K 就放不下，训练样本有 50–100K，不合适 |
| 5 | QSA 选择的并列规则（已实现：分数相同时选最近的块） | 接受 / 改成别的规则 | 接受 | 并列很常见（relu 后很多块分数正好是 0）；原先的选法取决于 kernel 和批大小，不确定。线上 vLLM 用它自己的选法，训练无法和它逐块一致 |
| 6 | 123K 在 parts 模式下 OOM（现在自动退回 layer 模式，最长的样本会慢一些） | 修 / 先不管 | 等第 4 项定了再说 | 如果用 B200，很可能直接放得下 |
| 7 | 剩下几项不改数值的小优化（gather 融进 GEMM、直接读 INT4 的 GEMM、PLE 查表和计算重叠、CUDA graph） | 做 / 跳过 | 跳过 | 按 profile 估计每项在 H200 上都不到 5%；时间线显示 32K 时层外开销只剩约 0.46 s |

第 1 项的统计（[prefix_share.py](prefix_share.py)，三轮转写，样本 = 截断段 × 关卡，训练这个关卡的回复）：

| 运行 | 口径 | 样本 | 中位长度 | 训练 token | 前缀 | 一步提速 |
|---|---|---|---|---|---|---|
| reset-baseline v01 | 通关关卡 | 185 | 95K | 30.3% | 64.7% | 1.86× |
| context-64-128k v01 | 通关关卡 | 153 | 97K | 30.4% | 64.4% | 1.85× |
| context-60-110k v01 | 通关关卡 | 169 | 93K | 29.9% | 65.1% | 1.87× |

（所有关卡的口径几乎一样。提速按前缀只做前向 = 1 份、其余前向 + 重算 + 反向 = 3.5 份估算。）实现方式：先无梯度地跑前缀，留下每层的 QSA K/V、GDN 状态和短卷积尾部，再带梯度跑后面；需要 QSA kernel 支持查询只覆盖后半段、GDN 用初始状态，约一天。A/B 还需要训练循环，现在还没有。

不需要你决定、我晚上会做的：最后一批（`fast_gdn`、QSA 胶水编译、top-k 提速）先在真实模型上做逐层对照，通过再提交；然后在 H200 上测 64K 的真实速度。`fast_gdn` 用到的 fla GVA 在 Hopper 的 tilelang 后端上还没测过，不行就只关这一个开关。

## 正确性守则

- 每个新 kernel：小模型上和参考实现对齐的测试、真实尺寸下对照 FP32 参照、启动自检和自动回退。
- 每次端到端改动：同一个 64K 样本的 loss 完全一致，LoRA 梯度在第 0 步定下的容差内。

## 开发方式

- kernel 用 Triton 写，在便宜的卡上开发和单测（10-19 用的 A100），不加载 181 GB 的模型；本地 Mac 跑不了 Triton。速度用 [bench_layer.py](bench_layer.py) 的单层基准，脚本是 [pod/kernel_pod.sh](pod/kernel_pod.sh)（先跑 CUDA 对照测试，再跑基准）。
- 端到端计时在 H200 上用 `smoke_lora --serve` 热更新跑 job；每台 pod 一个脚本，结果存到 `runs/runpod/jonathanwang2022/rft-lora-smoke/<pod-id>/`，记录写进 [Logs.md](Logs.md)。

## 今天（10-19）的目标

- [x] T1 第 0 步代码：QSA 的 K/V 梯度可选 FP32 累加（job 里 `kv_grad_fp32`）；保存和比较 LoRA 梯度（`keep_grads`、`compare_grads`）；offload 拷贝计时（`offload_stats`）。
- [x] T2 第 1 步代码：后面几层的 checkpoint 留在 GPU（`gpu_keep_gb`，可设 `"auto"`）；锁页 + 单独 stream 的异步拷贝和反向预取（`async_copies`）；同一个张量只 offload 一次。在一张 RTX 3090 上验证（[结果](../../runs/runpod/jonathanwang2022/rft-t2-gpu-tests/17a68q43k5mv0y/results/)）：18 个测试通过，梯度逐位一致；锁页拷贝 23.5 GiB/s，和矩阵乘完全重叠。
- [x] T3 第 2 步代码：`recompute: "parts"`，mixer 和逐 token 部分各自只重算一次；小模型上 logits 和梯度逐位一致，MoE 每块算 2 遍（原来 3 遍，层级重算会提前停在最后一块之前，实际每层 3 × 块数 − 1 遍）。
- [x] T4 第 3 步准备：`qsa_overlap` job，统计相邻 4/16/64/256 个 query 的选择并集和最近 4,096 个 token 的占比。
- [x] T5 H200 pod（93so3gmik3uttk，约 49 分钟，约 $4.3）：64K 到 784 token/s（1.66×），32K 800 token/s。梯度差异的原因：MoE 合并用 BF16 原子加，前向不可复现，同一步两次梯度余弦只有 0.87；改成 FP32 固定顺序合并后 0.99989。parts 模式在 123K OOM，先退回 layer 模式。详见 [Logs.md](Logs.md)。
- [x] T6 结果写进 Logs.md、README 和 AGENTS.md。

下午（目标改为当天 10×）：

- [x] T7 第 3 步：QSA 融合 kernel（[qsa_kernel.py](qsa_kernel.py)，Triton）。前向每个 query 一个程序；反向 dQ 同样按 query，dK/dV 按 16 个相邻 query 的选择并集算。对照 FP32 稠密参照（输出、LSE、dQ/dK/dV）和整模型对照 transformers 的 QSA（[test_qsa_kernel.py](../../tests/test_qsa_kernel.py)、[test_qsa_fused_model.py](../../tests/test_qsa_fused_model.py)）。checkpoint 重算复用前向的选择，不再跑第二遍 indexer。
- [x] T8 第 4–6 步：专家缓存、单次 grouped GEMM、编译的 hyper-connection 和 MoE 合并、通道在后的短卷积；单层基准 227 → 64 s（A100）。每项都有对照测试（小模型对 transformers 逐位一致，或对 FP32 误差不超过 BF16 eager 的 1.5 倍）。
- [x] T9 正确性：整模型梯度对比在这个模型上分不出对错（任何两种实现都只有 0.79–0.87 余弦），改为真实输入的逐层对照（`layer_check`）：新路径和上午的路径一样接近原版（LoRA 梯度余弦 0.9996+）。
- [x] T11 晚上：PLE 查表每步一次、确定的 QSA 并列规则、Triton grouped GEMM（非 Hopper）、融合的 LoRA 更新；RTX PRO 6000 上 32K 1,470 tok/s（上午的路径 569）。`keep_mixer` 量过更慢，没提交。
- [ ] T10 H200 上 64K 端到端计时（当天的 H200 不是没货就是网络太差）。
