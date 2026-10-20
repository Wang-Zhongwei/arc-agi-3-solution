# RFT LoRA 冒烟测试日志

每次上 GPU 的发现、试错和调整，按时间顺序记录。当前用到的 patch 和最新结果见 [README](README.md)，坑点清单见 [AGENTS.md](AGENTS.md)。每台 pod 的日志和结果在 `runs/runpod/jonathanwang2022/rft-lora-smoke/<pod-id>/`。

## 10-18

### 第一台 pod（rml9igjz2p6u3m，1×H200 SXM，约 35 分钟，约 $3.1）

| 发现 | 原因 | 处理 |
|---|---|---|
| `device_map="auto"` 把全部 routed experts 放到了 CPU | AutoRound 换成 INT4 之前，按 BF16 估算专家大小（约 242 GB） | 写 `gpu_device_map`：除 PLE 表外全部放 cuda:0 |
| 加载时 OOM：已分配 66.6 GiB，另有 72.7 GiB 预留但没用上 | 加载器大量小拷贝把显存切碎 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| 加载成功（318 s，68.5 GiB），第一步 32K 就 OOM（申请 2.13 GiB，正好是 33,811² 个 bf16） | transformers 的 QSA 是参考实现：每个 query 一次 Python 循环，外加 N×N 的稠密 mask | 删 pod，回本地写分块 QSA |

### 本地（第一台之后）

- 分块 QSA（[qsa.py](qsa.py)）：每 256 个 query 一块，只 gather 每个 query 选中的约 2,051 个 key，每块套 checkpoint。迷你模型上和参考实现对齐；选择只在并列分数处可能不同（relu 之后很多 block 正好是 0 分，topk 随便取）。RoPE 只转每个 head 的前 25%，必须用模型自己的 `apply_rotary_pos_emb`。
- 激活 offload（`save_on_cpu`）和分块 cross-entropy，同样在迷你模型上和原实现对齐。

### 第二台 pod（23ickggvkfukd3，1×H200 SXM；两台合计约 2.3 小时、约 $12）：六次运行

| 次 | 现象 | 原因 | 处理 |
|---|---|---|---|
| 1 | 32K 前向申请 24.04 GiB 时 OOM，开 offload 也一样 | SDPA 的 `enable_gqa` 在有 mask 时走 math kernel，把 gather 出来的 key 按 head 展开并升到 FP32 | 把共享一个 KV head 的 query head 排成 SDPA 的 query 位置，不用 `enable_gqa` |
| 2 | 32K OOM（已分配 137.6 GiB）；开 offload 后容器把进程 OOM 杀掉 | `from_pretrained` 返回 eval 模式，transformers 只在 train 模式下做层 checkpoint，所有激活都留着 | 每步前 `model.train()`；结果里报告 `checkpointed_layers` |
| 3 | 反向报 RuntimeError | torch 2.8 自带的 Triton 3.4 在 Hopper 上算错 fla 的 GDN 反向，fla 直接报错（fla #640） | 装 tilelang，设 `CUDA_HOME`；fla 在 import 时检查 tilelang，必须重启进程 |
| 4 | 跑通：32K 200 tok/s（前向 19.7 s，反向 149.8 s），开 offload 197 tok/s；之后进程又被容器 OOM 杀掉 | pinned 主机内存按 2 的幂取整并缓存：32K、64K 两步之后占约 144 GB，加上 102.5 GB 的 PLE 表，超过 251 GB | `pin_memory=False` |
| 5 | 整模型 profiler（带 `profile_memory`）让主机内存涨了 55 GB，GPU 空转 20 多分钟 | 整个模型的事件太多 | 停掉；改成训练时每秒采样 GPU 利用率和功耗 |
| 6 | 64K（offload）199 tok/s，峰值 127.6 GiB；100K OOM（申请 4.72 GiB） | AutoRound 的 MoE 一次 gather 全部 N × top-10 行（100K 时 4.7 GiB），再对 512 个专家逐个循环；显存每 token 涨约 0.9 MB | 准备分块 + 分组计算 MoE，见“第三台 pod 之前的调整” |

第二台的结果（o = 层 checkpoint 放到主机内存）：

| 样本 | Token | 图片 | 训练 token | 前向 | 反向 | Token/s | 显存峰值 |
|---|---|---|---|---|---|---|---|
| 32K | 33,811 | 15 | 5,664 | 19.7 s | 149.8 s | 200 | 123.4 GiB |
| 32K o | 33,811 | 15 | 5,664 | 21.0 s | 150.8 s | 197 | 98.4 GiB |
| 64K o | 64,893 | 32 | 8,141 | 63.8 s | 261.9 s | 199 | 127.6 GiB |
| 100K o | 99,072 | 55 | 9,942 | MoE 前向 OOM | | | > 139.8 GiB |
| 32K o，加 shared expert LoRA（3,350 万参数） | 33,811 | 15 | 5,664 | 38.2 s | 156.5 s | 174 | 99.3 GiB |

GPU 采样：反向 80–90% 时间在跑 kernel，但只有约 350 W（满载 700 W）；前向受 CPU 限制，只有 15–40% 忙。按 200 tok/s 算，数据集（160 条轨迹，2,160 万 token）要约 30 小时。

### 第三台 pod 之前的调整

| 上次的问题 | 这次的调整 | 本地验证情况 |
|------------|-----------|--------------|
| 100K 在 MoE 前向时 OOM：显存随长度增长，每个 token 约 0.9 MB | 把每层里逐 token 的部分（hyper-connection、路由、shared expert、MoE、注入）切成 8,192 token 一块计算，每块套检查点；PLE 层整体套检查点。这部分显存不再随长度增长 | 迷你模型上，logits 和全部参数的梯度都和 transformers 原实现一致 |
| 反向是大量低效的小 kernel，前向受 CPU 限制：每层用 Python 循环 512 个专家，每次调用都要反量化一遍权重 | MoE 改成分组计算：按专家排序 token，每 128 个专家一组，用 AutoRound 自带的 Triton kernel 一次反量化成 BF16，再调用 torch 的 grouped_mm。每层从 512×3 次小调用变成大约 12 次大调用 | 和逐专家循环的结果一致。GPU 上启动时还会自检一次：与 AutoRound 的 INT4 原实现比对，误差超过 5% 就自动退回原实现 |
| 短卷积退回了 PyTorch 参考实现 | 加载时加 --use-kernels，从 HF Hub 下载编译好的 causal-conv1d；失败就不加这个选项再加载一次 | 要在 GPU 上验证 |
| OOM 会直接中断运行 | OOM 后清理显存，把 token 块减半重试，最多两次；还不行就跳过这个样本并记录原因 | 代码路径已测（CPU 上无法真正触发 OOM） |
| 主机内存被打满，服务两次被容器的 OOM 杀掉 | 开 offload 前先估算这一步检查点需要的主机内存，和容器剩余可用量比较，不够就跳过 | 同上 |
| 爆显存要到后面才发现 | 先跑最长的真实样本（cd82 那局，约 110K），再跑 32K 和 64K 测提速 | 按任务指定 transcript 已在本地测过 |
| 整模型 profiler 太重，pod 空转 20 多分钟 | 不再用 profiler，改成训练时每秒采样 GPU 利用率和功耗 | — |
| 选了 secure SXM | 开 pod 前先按"能放下的最便宜机型"查库存 | 刚查过，见下 |

### 第三台 pod（i1snr1yuwi9qf9，secure 1×H200，US-NC-1，43 分钟，约 $3.8）

过程（UTC）：

1. 22:32 开机：社区云没有满足内存要求的 H200，用了 secure。装包 43 s，下载 checkpoint 55 s（169 GB）。
2. 22:35 发现这台主机是 cgroup v1。主机内存保护读的是 v2 的 `memory.max`，读不到就静默跳过。改成两种都读，在第一个 job 开始前热更新上去，读到 232 GiB 可用。
3. 22:38 `--use-kernels` 在整个加载（约 4 分钟）结束后才失败：kernels 0.17.2 去 Hub 查 `kernels-community` 的发布者信任状态时出错。回退的重新加载写在 `except` 里，traceback 还引用着失败的模型，第二次加载时显存已到 130 GB。停掉服务，把重试挪到 `except` 外面（先 `gc.collect()` 和 `empty_cache()`），脚本去掉 `--use-kernels`，22:41 重启。共浪费约 7 分钟。
4. 22:46 加载完成：257 s，显存 68.5 GiB，进程只占 2.9 GB 主机内存（PLE mmap 生效）。专家自检：分组路径对 AutoRound INT4 原路径的最大相对误差 0.3%。
5. job 01：cd82 最长的请求，106,686 token，峰值 119.6 GiB，275 tok/s，没触发 OOM 重试。GPU 采样：前向只有 26–29% 忙，反向 70–80%。
6. job 02：32K 251 tok/s，64K 328 tok/s。反向快了一倍，但 32K 前向从 21.0 s 变慢到 38.6 s。先后排除了 PLE 缺页（cgroup 总共只有 7,187 次 major fault）和 CPU 限流（只有 1.4% 的周期被限流）。当时以为是新长度的预热，后来证明是错的（见第 11 步）。
7. 你问 110K / 118K 会不会 OOM：在 128K 预算那次运行（ticbudja/context-64-128k v01）里找到最长的请求（ft09，估计 126.5K token），排进队列。
8. job 03：单层 profile（64K，不 offload）。GDN 层前向 0.24 s、前向 + 反向 0.68 s；PLE 额外 0.46 / 1.09 s；QSA 层 0.57 / 7.28 s，其中 4.8 s 是 key/value gather 的反向（`k[idx]` 走排序版的 `index_put`）。
9. job 04：ft09，122,943 token，峰值 129.0 GiB（拟合预测 128.8），348 tok/s。同时把 gather 改成 `index_select`（反向是原子 `index_add`），本地 QSA 测试通过后热更新。job 05：QSA 层前向 + 反向 7.28 → 3.61 s；job 06：64K 328 → 474 tok/s。
10. 同一个 64K 样本两次的梯度范数不同：0.248（旧 gather）和 0.293（新）。GPU 小测试：两种 gather 的 K/V 梯度相对 FP32 都差约 1.7%（都在 BF16 里累加），FP32 累加只差 0.33%。所以不是新 gather 变差，差异原因还没找到。
11. offload 小测试：GPU → 新分配的 pageable 内存只有 1.7 GiB/s，写进复用的 buffer 有 16.5 GiB/s，`THP_MEM_ALLOC_ENABLE=1` 也只有 3.4 GiB/s。逐层前向加起来约 16 s，64K 前向却要 58 s，差的约 35 s 就是首次写入新内存的缺页。第二台 pod 的主机约 25 GiB/s，所以之前没暴露。第 6 步的"预热"判断由此更正。
12. 约 23:15 保存日志和结果，删除 pod。

上面调整的验证：

| 调整 | 结果 |
|---|---|
| 每层切成 8,192 token 一块 | ✅ 106,686 token 峰值 119.6 GiB，122,943 token 129.0 GiB；每千 token 的显存增长从约 0.9 GiB 降到约 0.45 GiB |
| MoE 分组计算 | ✅ 反向：64K 从 262 s 降到 134 s，自检误差 0.3%。❌ 前向没变快：瓶颈是 offload，不是专家 |
| `--use-kernels` | ❌ 去掉：Hub 信任检查失败，拉到的是推理 kernel；GDN 层本来就便宜（0.68 s） |
| OOM 后减半重试 | ⚪ 没触发，保留 |
| 主机内存保护 | ✅ 修了 cgroup v1 之后才生效；123K 时主机峰值 183 GB（上限 251 GB），没触发 |
| 先跑最长的样本 | ✅ 两个长样本第一次就放下了 |
| 不用整模型 profiler | ✅ 每秒采样看出前向利用率低；单层 profile（约 2 分钟）找到了两个真正的瓶颈 |
| 先查最便宜的机型 | ⚠️ 社区云没货，还是 secure。PLE 改成 mmap 后内存需求下降，下次内存门槛可以放到约 200 GB |

下一步：

1. offload 复用主机 buffer：64K 每步约省 35 s，也不再依赖主机的缺页速度。
2. QSA 的 K/V 梯度改为 FP32 累加，并重复同一步，看梯度的波动有多大。
3. 然后是训练循环、数据集（留出部分游戏做评估）、merge 后和 vLLM 的对齐。

## 10-19

### 本地：吞吐计划和前三步的代码

- 写了 [Throughput.md](Throughput.md)：目标 10×（64K 137 s → ~14 s），分 0–6 步，每步一个问题，每步都有测试。
- 第 0 步：`kv_grad_fp32` 让 QSA 的 K/V 梯度在 FP32 里累加（块内和块间）；`keep_grads` / `compare_grads` 保存并比较 LoRA 梯度。测试里每个 key 被约 3,000 个 query 选中时，FP32 累加的误差 0.44%，BF16 是 0.69%；剩下的是每一项本身的 BF16 舍入。
- 第 1 步：[host_offload.py](host_offload.py) 的 arena 保留 16 GiB 的段并在每步复用；同一个张量只拷一次；最后 `gpu_keep_gb` 留在 GPU；`async_copies` 用锁页的段在单独的 stream 上拷贝，反向时预取。写去重时踩了一个坑：pack 之后原张量会被释放，新张量可能落在同一地址、同一形状、同一版本号上，按地址去重会拿错数据，改成按基张量对象（弱引用）去重。
- 第 2 步：`recompute="parts"` 关掉层级 checkpoint，mixer 单独 checkpoint。小模型上 logits 和梯度逐位一致。数调用次数时发现：层级重算会在最后一块之前提前停下（PyTorch 非重入 checkpoint 的 early stop），所以 layer 模式每层跑 3c − 1 遍（64K 是 23 遍），parts 是 2c 遍（16 遍）。
- 本地 16 个测试通过，1 个 CUDA 测试待在 pod 上跑。都还没上 GPU。
- 决定第一轮不做 shared expert LoRA：旧代码上 32K 慢 12%，还要单独 A/B；RFT 收益小再回头看。训练参数先取偏保守的默认值。

### 本地：吞吐计划和今天的代码

- 写了 [Throughput.md](Throughput.md)：10× 的分步计划。
- 第 0 步：QSA 的 K/V 梯度可选 FP32 累加（`kv_grad_fp32`）；保存 LoRA 梯度并逐模块比较（`keep_grads`、`compare_grads`）。
- 第 1 步：host_offload 加了三样东西。同一个张量只 offload 一次（按存活的 base 张量识别，第一版按地址识别，释放后的内存被新张量复用，梯度出错）；显存够时最后的 checkpoint 留在 GPU（`gpu_keep_gb`，`"auto"` 按上一次的峰值算）；锁页的 16 GiB 段 + 单独 stream 的异步拷贝，反向时预取。
- 第 2 步：`recompute: "parts"`，mixer 和逐 token 部分各自 checkpoint，逐 token 部分每块算 2 遍（原来每层 3 × 块数 − 1 遍：层级重算会在最后一块之前提前停止）。
- 第 3 步准备：`qsa_overlap` job。

### 轻量 GPU（17a68q43k5mv0y，社区 RTX 3090，约 15 分钟，约 $0.06）

- 只跑单元测试和拷贝测速，不加载模型（[pod/gpu_tests.sh](pod/gpu_tests.sh)）。
- 发现：CPU 上的小模型测试在 x86 上梯度不逐位一致。原因是 host buffer 一律连续存放，而 `.cpu()` 保留 stride，反向 kernel 走了不同路径。改成稠密张量保留原 stride 后 18 个测试全部通过。
- 拷贝速度：新分配的 pageable 1.4 GiB/s，复用的 pageable 13.8 GiB/s，锁页 23.5 GiB/s；锁页拷贝放在单独 stream 上和矩阵乘完全重叠。

### 第四台 pod（93so3gmik3uttk，secure 1×H200，US-NC-1，约 49 分钟，约 $4.3）

社区云没有 ≥ 240 GB 内存的 H200，用了 secure。同一个 64K 样本（64,893 token，训练 8,141）：

| 配置 | 前向 | 反向 | Token/s | 显存峰值 | 说明 |
|---|---|---|---|---|---|
| 原来的 `save_on_cpu` | 59.0 s | 78.6 s | 472 | 99.3 GiB | 和第三台的 474 一致 |
| 复用 pageable buffer | 33.6 / 44.0 s | 79.9 / 80.7 s | 571 / 520 | 100.7 GiB | 第二次前向反而慢，原因未查 |
| 加 FP32 K/V 梯度 | 33.1 / 36.3 s | 93.7 / 91.9 s | 512 / 506 | 100.7 GiB | 反向慢 12–14 s |
| parts + 同步拷贝 | 29.5 s | 90.7 s | 540 | 96.2 GiB | 每层 ~30 次同步 unpack，反向变慢 |
| **parts + 异步 + 留 GPU** | **16.8 s** | **66.0 s** | **784** | 121.1 GiB | 留 35.6 GiB 在 GPU；第一次锁页 112 GiB 时 522 |
| 同上，32K | 11.5 s | 30.7 s | **800** | 126.5 GiB | 几乎全部留在 GPU；第三台 32K 是 251 |

- 前向 16.8 s 基本就是 GPU 计算时间，拷贝不再挡路。现在反向占约 80%，主要是 QSA。
- ft09（122,943 token）parts 模式三次都 OOM：反向要一块 4.69 GiB = N × 10,240 × 4 B（整条序列、hyper-connection 宽度的 FP32 张量），减小 token 块没用。原因未查；先在 OOM 时退回 layer 模式（123K 在 layer 模式下放得下）。
- 留在 GPU 的张量被保存两次时重复计数，下一步留得比预算少（应留 47 GiB，实际 24 GiB）。已修，热更新后生效。
- **梯度不可复现**：同一步重复两次，LoRA 梯度的余弦只有 0.87–0.90，相对差 45–52%；FP32 K/V 累加也一样；loss 在 0.2745–0.2789 之间变。原因是 MoE 用 BF16 的 `index_add`（原子加）合并 10 个专家的输出，每次加法顺序不同，前向就不同，近似并列的路由和 QSA 选择在 48 层里被放大。改成先 gather 回原顺序、再在 FP32 里对 10 个专家求和之后：两次 loss 都是 0.2750，梯度余弦 0.99989，相对差 1.5%（剩下的来自反向的原子加）。这两步只有 424–433 token/s，但当时锁页 arena 已涨到 192 GiB（ft09 那次），主机内存紧张，速度要下次单独量。
- QSA 选择重叠（64K，12 个 QSA 层）：相邻 4 / 16 / 64 / 256 个 query 的并集是单个 query 选择的 1.9–2.2 / 3.4–4.3 / 5.7–7.3 / 8.9–11.1 倍；33–47% 的选择落在最近 4,096 个 token 里。
- 装了 fla 的 pod 上 CPU 小模型测试会失败（fla 的 Triton kernel 拒绝 CPU 张量），`smoke_pod.sh` 改为只跑 CUDA 测试。

下一步：

1. 下次 H200：单独量确定性合并的速度（arena 不超过 112 GiB）；查 parts 在 123K 的 4.69 GiB 分配。
2. 第 3 步 QSA kernel：16 个 query 一块取并集，或按 query 读、靠 L2 复用；在 H100 上开发。


### 下午：目标改为当天 10×，第 3–6 步（kernel pod ux3e2aop5mnv1t，secure A100 SXM 80GB，约 52 分钟，约 $1.55）

你的要求：先把速度做上去，123K 的 OOM 先不管；显存可以多用一点，实在不够开 B200；每步先在便宜的卡上测；每次优化都要和标准答案对照。H200/H100 都没货，48 GB 的卡也没货，用了 A100。

做法：写了单层基准 [bench_layer.py](bench_layer.py)（真实尺寸、随机权重的 GDN 层和 QSA 层，INT4 专家 + LoRA，量 checkpoint 下的前向和重算 + 反向），按 36 GDN + 12 QSA 估算整步。旧路径算出 227 s，H200 实测 83 s，比例约 2.7。

| 轮次 | 改动 | GDN 层 | QSA 层 | 整步（A100） |
|---|---|---|---|---|
| r1 | 旧路径 | 2.11 s | 12.62 s | 227 s |
| r1 | QSA 融合 kernel（Triton，每个 query 一个程序，dK/dV FP32 原子加） | 2.12 s | 3.78 s | 122 s |
| r2 | 专家缓存（一层 BF16 专家约 5 GB）+ `split` 切块 | 1.65 s | 3.31 s | 99 s |
| r2 | + 编译 hyper-connection 和 MoE 合并 | 1.43 s | 3.11 s | 89 s |
| r3 | + 512 个专家一次 grouped GEMM、gather 反向不用 `index_put`、dK/dV 按 16 个 query 的并集 | 1.40 s | 3.56 s* | 93 s* |
| r3b | 同上，QSA 用合成的真实统计选择 | — | 2.32 s | 78 s |
| r4 | + 通道在后的 GDN 短卷积、A100 的 grouped GEMM 只算 dX | 1.03 s | 2.27 s | 64 s |
| r5 | 编译改为全动态形状 | 1.17 s | 2.40 s | 71 s |
| r6 | 编译改为块长补到 2 的幂的静态图 | **1.03 s** | **2.27 s** | **64 s** |

\* 随机权重让相邻 query 的选择几乎独立（16 个 query 并集 9.98 倍，真实约 3.5 倍），并集 kernel 白做 3 倍的活。之后基准用合成选择：每 16 个 query 共享的噪声 + 每个 query 的噪声 + 最近窗口加分，再取 top-k；最近 4,096 个 token 占 44%，16 个并集 3.46 倍，和 10-19 上午实测一致。

正确性（每项都对照了标准答案）：

- QSA kernel：输出、LSE、dQ/dK/dV 对照 FP32 稠密参照，误差不超过旧分块路径（`test_qsa_kernel.py`，两种 dK/dV 都测）；整模型（小模型带图、开 gradient checkpointing）的 logits 和注意力梯度对照 transformers 的 QSA，FP32（`test_qsa_fused_model.py`），并确认重算复用了选择（indexer 每层每步只跑一次）。
- 并集表：CPU 上对照暴力循环（`test_qsa_union.py`）。
- 专家缓存、单次 grouped GEMM、合并、两个 gather 的反向、`split`：小模型上对 transformers 逐位一致；只算 dX 的 grouped GEMM 对逐组 matmul 逐位一致。
- 编译和短卷积：对照 FP32，误差不超过 BF16 eager 的 1.5 倍（`test_compile_hc.py`）。

踩到的坑：

- 并集 dK/dV kernel 越界：query 正好结束一个块时，它的尾块是下一个块，最后一个 tile 的尾块在 N 之后。kernel 只判断了块号 ≥ 0，读到 K/V 之外的垃圾，`0 × NaN` 让 dK 的越界写变成 NaN，正好落在紧挨着分配的 dV 开头。只有 dV 出错，且只在部分顺序下出现（新进程里是 device-side assert）。改为 `tok < N` 并把屏蔽位置的 dS 显式置 0。
- 编译版的路由梯度误差是 eager 的 6 倍。不是编译错了：小模型 top-3 of 8，有个 token 第 3、4 名专家只差 1.2e-5，编译版的舍入让它换了专家（200 个 token 里 1 个），MoE 输出就差 5%。对照测试改为每个 token 用全部专家（路由连续）后通过。先后排除了"前向和重算路由不同"（没有）和"函数本身不准"（单独测比 eager 还准）。
- A100 上 `torch._grouped_mm` 不可用，transformers 的后备是逐专家循环，反向还给冻结的专家算权重梯度。基准里 GEMM 因此偏慢；加了只算 dX 的后备。
- `ssh ... 'cd DIR && tar ... && nohup ... &'` 又让 ssh 卡住（AGENTS.md 里记过）：要用 `;`，不要 `&&`。

整模型对照（T9）在第二台 A100（ih0v0tcp42cfey，260 GB 磁盘）上跑：8K 样本，原版模型（`"reference": true`：transformers 的 QSA 和层、AutoRound 的专家循环）、旧路径、新路径各两遍，比较 loss 和 LoRA 梯度。

### 整模型对照（8K，r4ii27zn88nx8j，secure RTX PRO 6000 96GB；A100 80GB 放不下真实样本）

A100 上 8K（这局游戏最短的请求也有 7.4K）三条路径全部 OOM：68.5 GiB 模型 + ~10 GiB 固定开销。换 96 GB 的 RTX PRO 6000（sm120）。这台主机连不上 PyPI（pypi.org 和镜像都超时），在 Mac 上用 [pod/fetch_wheels.py](pod/fetch_wheels.py) 下好 wheel 传过去离线装。7 个 CUDA 对照测试在 sm120 上也都通过。

同一个 8K 请求，各跑两遍：

| 对照 | 梯度余弦 | 相对差 | loss |
|---|---|---|---|
| 同一路径两遍（三条路径都是） | 0.9999 | 1.4–1.6% | 完全一致 |
| 旧路径 vs 原版模型 | 0.833 | 58% | 0.5079 vs 0.5176 |
| 新路径 vs 原版模型 | 0.665 | 99% | 0.5172 vs 0.5176 |
| 新路径 vs 旧路径 | 0.660 | 98% | |

第一个怀疑是 QSA 选择的并列：relu 之后很多块分数正好是 0，正分不足 512 个时剩下的名额在并列里挑，transformers 逐 query 的 topk、我们 256 和 2,048 一块的 topk 挑的都不一样。改成确定的并列规则（稳定排序，并列选最近的块），参照模型也用同一个选择，重跑：

| 对照（同一个选择） | 梯度余弦 | loss |
|---|---|---|
| 旧路径 vs 原版 | 0.833 | 0.5096 vs 0.5126 |
| 新路径 vs 原版 | 0.786 | 0.5172 |
| 只换 QSA 注意力（旧路径 vs transformers 注意力） | 0.819 | 0.5096 vs 0.5113 |
| 只换层和专家（fused QSA + transformers 层 vs 原版） | 0.851 | 0.5190 |

结论：任何两种实现之间只有 0.79–0.85 的余弦，连只换注意力实现（小模型上逐 1e-5 一致）也是 0.82。上午只是 BF16 原子加顺序不同也是 0.87。top-10/512 的路由和后面各层的 top-k 选择把舍入级的差异变成换了专家、换了块，48 层放大。所以整模型梯度对比在这个模型上分不出“正确但舍入不同”和“有 bug”。改为在真实模型上逐层对照：用真实输入，单层里的翻转只影响少数 token，而且数出来。

### 晚上：逐层对照、grouped GEMM、fast_lora（同一台 RTX PRO 6000，约 77 分钟，约 $3.2）

- 真实输入逐层对照（8K，层 0 GDN、1 GDN+PLE、3 QSA；对 transformers 层 + AutoRound 专家 + transformers QSA 注意力，同一个选择）：上午的路径 LoRA 梯度余弦 0.99974 / 0.99976 / 0.99962，新路径 0.99970 / 0.99975 / 0.99964，输入梯度都约 0.9999；路由翻转 11–17% 的 token（BF16 logits 并列）。`keep_mixer` 和新路径逐位相同。
- 整步（第二步）：16K 上午路径 572 → 新路径 1,101 tok/s；`keep_mixer` 1,064（offload 激活的开销大于省下的重算，撤回）。
- profile（16K）：GDN 层一半时间是专家 GEMM 的逐专家循环（6,163 次 matmul）。写了 Triton grouped GEMM：16K 1,314、32K 1,475 tok/s（上午的路径 569）；AutoRound 自检误差 0.75% → 0.53%；逐层对照不变。
- `fast_lora`：PEFT 的 FP32 adapter 更新编译成一个函数；和 PEFT 逐位一致（CPU），逐层对照不变；同一次运行 32K 1,421 → 1,470 tok/s。
- 时间线（32K）：层内前向 7.3 s、反向 15.5 s，层外合计 0.46 s；GDN 层占 66%，QSA 层 32%。
- 服务循环不热重载：新加的 job 参数要重启服务才生效（第一次量 fast_lora 时没生效，是这个原因）。
- 冰岛的 H200（smpzdlhal2bt5q）网络太差（HF 12 MB/s、PyPI 223 KB/s），开了约 25 分钟没跑任何 job 就删了，约 $2.2。

## 10-20 凌晨：完整流程演练（RTX PRO 6000，dshsgfhx2rzdn3，约 43 分钟，约 $1.8）

目的：B200/H200 上要一次跑完、中途不做决定，所以先在便宜的卡上把 [full_run.sh](pod/full_run.sh) 整个走一遍。B200、H200 SXM、H200 NVL 当时全部没货。

- 开头的 CUDA 测试里 `test_prefix_stop_grad_cuda` 失败（最差 0.063），make_jobs 自动关掉了前缀不回传梯度的 job。排查：只开 `compile_hc` 时 `dt_bias`/`A_log` 梯度差 19%。但编译的 hyper-connection 函数和 eager 在 FP32 下差 3e-7；换成每个 token 走全部专家后降到 1.1%；未改动的模型输入乘 (1 + 1e-7 噪声) 也会让这两个梯度变 1.5–2%；改用 transformers 的 torch GDN（不用 fla）后只有 0.03%，而且 `compile_hc` 一点不变。结论：19% 是 top-3/8 路由翻转，剩下的是 fla GDN kernel 的 TF32 点积，这两个参数的梯度是对所有 token 求和、互相抵消的量，被放大。测试改用全专家路由后通过。
- 32K：旧路径 67.1 s，新路径 20.6 s（3.25×）。旧路径第一步 LoRA 梯度范数 341（之后 0.379，loss 一样）：layer_check 只在每个变体之前清梯度，最后一个变体的梯度留在层上，`step` 在算范数之后才清。改成 `step` 开头先清。
- 补跑的前缀不回传梯度 job 在 32K OOM，之后 16K 的普通 job 也 OOM。两个问题：
  - OOM 的重试写在 except 里，异常的 traceback 还拿着失败那一步的帧、激活和计算图（第二次尝试一开始就占了 92 GiB）。改成出了 except 再 `gc.collect()` 后重试，之后的 job 不再受影响。
  - 前缀多占的显存：加了 `memory_snapshot`（CUDA 分配记录，前向结束和 OOM 时导出）和 [mem_report.py](mem_report.py)。16K 前向结束时多出 36 块 `lora_add` 分配的张量（3.6 GiB），32K OOM 时 12.1 GiB。原因：前缀的 GDN 短卷积状态取的是 `conv_in[:, -3:]` 的视图，留住了整段前缀的 `in_proj_qkv` 输出（p × 10,240 BF16 每层），后半段的 checkpoint 一直拿着它到反向。改成 `.clone()`。
- 修好后 32K：新路径 23.7 s，加前缀不回传梯度 15.8 s（1.50×，前缀 49.5%），峰值 85.0 vs 85.7 GB；同一路径两遍梯度余弦 0.99993/0.99994；前缀不回传梯度和完整梯度余弦 0.78（截断本身）。
- 冷启动的第一步（从没跑过的 8K 形状）和第二步 loss 一样（0.5114），所以 autotune 不改结果；不同服务进程之间同一个样本的 loss 差约 0.3%（0.4662 vs 0.4678），是舍入经路由放大。
- pod 自删：演练时 pod 上的 full_run.sh 还是改环境变量之前上传的旧版，回退成“只停进程”；上传当前版本后在这台 pod 上实测：`pod "dshsgfhx2rzdn3" removed`，REST get-pod 404。本地 watcher 改成拉取结果成功后才删 pod（失败最多重试 5 次）。
