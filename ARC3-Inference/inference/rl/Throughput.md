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

## 正确性守则

- 每个新 kernel：小模型上和参考实现对齐的测试、真实尺寸下对照 FP32 参照、启动自检和自动回退。
- 每次端到端改动：同一个 64K 样本的 loss 完全一致，LoRA 梯度在第 0 步定下的容差内。

## 开发方式

- kernel 在便宜的 Hopper 卡（H100）上开发和单测，不加载 181 GB 的模型；TileLang 和 TileKernels 都要求 SM90 以上，本地 Mac 跑不了。
- 端到端计时在 H200 上用 `smoke_lora --serve` 热更新跑 job；每台 pod 一个脚本，结果存到 `runs/runpod/jonathanwang2022/rft-lora-smoke/<pod-id>/`，记录写进 [Logs.md](Logs.md)。

## 今天（10-19）的目标

- [x] T1 第 0 步代码：QSA 的 K/V 梯度可选 FP32 累加（job 里 `kv_grad_fp32`）；保存和比较 LoRA 梯度（`keep_grads`、`compare_grads`）；offload 拷贝计时（`offload_stats`）。
- [x] T2 第 1 步代码：后面几层的 checkpoint 留在 GPU（`gpu_keep_gb`，可设 `"auto"`）；锁页 + 单独 stream 的异步拷贝和反向预取（`async_copies`）；同一个张量只 offload 一次。在一张 RTX 3090 上验证（[结果](../../runs/runpod/jonathanwang2022/rft-t2-gpu-tests/17a68q43k5mv0y/results/)）：18 个测试通过，梯度逐位一致；锁页拷贝 23.5 GiB/s，和矩阵乘完全重叠。
- [x] T3 第 2 步代码：`recompute: "parts"`，mixer 和逐 token 部分各自只重算一次；小模型上 logits 和梯度逐位一致，MoE 每块算 2 遍（原来 3 遍，层级重算会提前停在最后一块之前，实际每层 3 × 块数 − 1 遍）。
- [x] T4 第 3 步准备：`qsa_overlap` job，统计相邻 4/16/64/256 个 query 的选择并集和最近 4,096 个 token 的占比。
- [x] T5 H200 pod（93so3gmik3uttk，约 49 分钟，约 $4.3）：64K 到 784 token/s（1.66×），32K 800 token/s。梯度差异的原因：MoE 合并用 BF16 原子加，前向不可复现，同一步两次梯度余弦只有 0.87；改成 FP32 固定顺序合并后 0.99989。parts 模式在 123K OOM，先退回 layer 模式。详见 [Logs.md](Logs.md)。
- [x] T6 结果写进 Logs.md、README 和 AGENTS.md。
