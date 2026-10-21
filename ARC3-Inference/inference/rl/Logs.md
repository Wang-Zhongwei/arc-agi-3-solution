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

## 10-20 凌晨：H200 一次跑完（4haw3pwub9vce4，secure US-NC-1，$5.29/h，03:56–04:32，约 $3.2）

- 03:58 网络检查通过，42 s 装完包，checkpoint 下载不到 4 分钟。
- CUDA 测试：`compile_layer`、`fast_gdn`、`fused_model`、`prefix_cuda` 先失败，都是同一个原因：测试里的 FP32 参照（小模型的 FP32 拷贝或未打补丁的 transformers 模型）在 Hopper 上被送进 `torch._grouped_mm`，它只收 BF16（`Expected mat_a to be BFloat16 matrix got Float`）。A100 和 RTX PRO 6000 不走这条路，演练没发现。`layer_patches.grouped_mm` 对非 BF16 改用逐组 matmul，测试的 conftest 把 transformers 的判断限制在 BF16。make_jobs 已经按旧结果关掉了 `compile_qsa` 和前缀不回传梯度；在模型加载的 5 分钟里重跑这些测试（全过，包括 fast_gdn 的 tilelang 反向），重新生成 job 文件，第一个 job 开始前换好。
- 64K：上午的路径 80.0 s（811 tok/s）→ 全部优化 22.6 s（2,874）→ 加前缀不回传梯度 16.1 s（4,031，前缀 80%，layer 模式）。对 10-18 的 137 s：6.1× / 8.5×。
- 前缀不回传梯度第二遍在 `gpu_keep_gb: "auto"` 留了 39.7 GB 后 OOM，重试成功，显存回到 74 GB 起步（except 外重试的修复生效）。
- 123K：全部优化 parts 模式 OOM，layer 模式 183.8 s（冷）；前缀不回传梯度 parts 模式放得下（111.9 GB）。
- 梯度：同一路径两遍 0.99993；前缀不回传梯度 vs 完整 0.689（前缀 80%）。
- watcher 在 FULL_DONE 后拉回结果并删 pod（HTTP 204，get-pod 404）；另有每 5 分钟一次的备份拉取。

## 10-20 凌晨：B200（uas538q7c7lapw，secure US-NE-1，$7.99/h，04:38–05:02，约 $3.2）

- CUDA 测试：fla 的 GDN 反向 kernel（`chunk_gated_delta_rule_bwd_kernel_dhu`）在 sm100 + Triton 3.4 下编译失败（`PassManager::run failed`，TritonGPUAssignLatencies），所有路径的 GDN 反向都用它。在 pod 上把它的 num_stages 改成 1（fla 已经对 Blackwell 的前向做了类似处理），测试全过，写进 full_run.sh（a301c3a）。
- 按你的要求只留默认配置（前缀不回传梯度）的吞吐和最长样本：删掉了全部优化路径、梯度对比、32K 和 profile 的 job；旧路径当时已经在跑，让它跑完（杀掉要重新加载模型）。
- 64K 前缀不回传梯度：17.2 s（3,772 tok/s，留 GPU 78.3 GB），H200 16.1 s（4,031）。每美元 token：B200 1.70M（secure）/ 2.27M（community），H200 2.74M / 4.04M。H200 更划算。
- 123K：不 OOM，峰值 111.8 GB（H200 111.9 GB）。
- 想补一个固定留 40 GB 的 64K job 做公平对比，加进去时服务已经停了，没跑成。

## 10-20 上午：训练集和训练循环

你的要求：选数据最要用心；通关率阈值从 90% 降到 80%（第一轮保守）；按动作数为主，但 token（全部生成的 token，不只是 thinking）用得太多的不要；每关用 percentile 而不是固定前 k；每个样本按它那次运行自己的配置重建（上下文窗口、图片放大倍数）。

| 发现 | 处理 |
|---|---|
| 一个回合的动作在回合结束时才执行（下一回合的提示写 "The code executed N actions"），`analysis` 事件在动作之后才记录 | 请求的关卡 = 标题 `action` − 1 个动作之后的关卡；`context_rebuild` 给每个请求和用户消息记下标题的 action |
| 约 170 个"失败"是从没轮到的游戏（0 个回合） | 尝试 = 该关至少有 1 个回合 |
| 同一关内动作数和生成 token 的 Spearman 中位数 0.65，但动作最少的 1/4 里有 21% 的 token 超过该关中位数 | 规则：动作在该关前 35%，且生成 token ≤ 该关中位数；阈值用所有运行的通关算 |
| transcript 只有 `[image attached]`，前一行标签说明画的是什么；`intermediate_states.pkl` 存了每个动作的全部帧 | [rft_images.py](rft_images.py) 按运行自己的 `MULTIMODAL_*` 重画；4 个有请求日志的运行里 12,778 张图逐像素一致 |
| bp35、lf52 的 pickle 帧和事件里的棋盘对不上（数组按引用保存，后来被改写） | 棋盘一律取事件；动画帧和过关帧只在整局帧都对得上时才用，否则丢掉含这些图的请求（155 个） |
| 游戏结束的回合，最后一个动作是自动 RESET，动画帧应取致命动作的 | 修正（g50t） |
| python 工具的描述各版本不同（5 种） | 从各运行的源码快照里用 ast 读出；其中 2 种和请求日志逐字一致 |
| 排除规则 `lordhansolo/` 误伤了我们自己的 `bp35-lordhansolo` 7 个运行 | 改成 `kaggle/lordhansolo/`，重建 |
| 单进程建数据 45 分钟 | 按运行并行，83 秒 |

结果（[README](README.md#dataset-v1-2026-10-20)）：训练 213 条轨迹、354 个样本、3,360 万 token（训练 730 万，前缀 64%），验证 12 条轨迹、34 个样本；压缩后 31 MB。[check_rft_data.py](check_rft_data.py) 用真实 processor 全部分词一遍：0 个问题，精确长度比估计高 2.8%。

训练循环 [train_rft.py](train_rft.py)：AdamW（lr 1e-4，预热 5%，余弦到 1e-5，裁剪 1.0），每步 4 个样本、每个训练 token 权重相同，前缀不回传梯度，每 5 步存 `latest`（adapter + 优化器 + 位置），每个 epoch 存 adapter，崩了从 `latest` 续跑。CPU 小模型测试：逐个样本累加的梯度等于整批的梯度（一个种子下逐元素容差太紧，FP32 舍入 9e-5，改成按范数比较）。

## 10-20 中午：第一次训练（fi846v8f7bt93s，secure H200，US-NC-1，$5.29/h）

LoRA r 16、α 32，2 个 epoch × 89 步；预计每个 epoch 约 2.9 h，总共约 6.5 h、约 $35。

## 10-20 下午：PLE 用 FP8 对梯度的影响（zn3ccmdmt8mm2t，RTX PRO 6000，约 15 分钟，约 $0.7）

起因：训练 pod 上 arena 长到约 157 GiB，加上 102.5 GB 的 PLE 页缓存超过 251 GB，PLE 被挤出缓存，每步从磁盘重读，GPU 空等。NVIDIA 的 NVFP4 版里 PLE 是 FP8（51.2 GB，整表一个 scale），想看换成它对训练有多大影响。

同一个 32K 样本（33,811 token，前缀 16,736，训练 5,664），训练配置（前缀不回传梯度）：

| PLE | loss | LoRA 梯度范数 | 和 BF16 的梯度 cosine | 查出的行和 BF16 的相对差 |
|---|---|---|---|---|
| BF16（第二遍） | 0.4637 | 0.2950 | 0.99994（重复噪声） | — |
| NVIDIA FP8 | 0.4625 | 0.3239 | 0.784 | 2.7% |
| 每行 scale 的 FP8 | 0.4647 | 0.3054 | 0.850 | 2.6% |

loss 只差 0.2–0.3%，但整模型梯度方向差很多。这和之前的结论一致：任何舍入级的差别都会经过 top-10/512 路由和 QSA 的 top-k 被放大（两种实现之间也只有 0.79–0.87），所以"cosine ≥ 0.999"这个标准对 FP8 达不到，也不能说明 FP8 比别的数值差别更坏。

## 10-20 下午：修 PLE 被挤出缓存，10 个样本演练（okshr3l1za6nmm，secure H200，US-NC-1，约 35 分钟，约 $3；前一台 l9xfuzwqdsp5n5 开机 9 分钟容器没起来，删掉重开，约 $0.8）

改动（都先在 CPU 小模型上和原实现对照）：

| 改动 | 数值和理由 |
|---|---|
| arena 固定大小，启动时一次分配好，满了剩下的留在 GPU 上 | 大小 = 容器上限 − PLE 表 − 进程 − 12 GiB（预取的两批）− 10 GiB 余量。这台容器上限 233.8 GiB（不是 251），PLE 95.4，进程 3.0 → 113.4 GiB，pin 用了 42.5 s |
| 每个样本选 parts 或 layer | 前缀之后的 token × 每 token 字节数 ≤ arena 就用 parts。实测 parts 约 2.0 MB/token（默认 2.4），layer 约 1.2–1.6 MB；分界约前缀之后 58K token，352 个样本里 332 个 parts、20 个 layer |
| 不再往 GPU 上留 checkpoint | 留下的视图会拖住整个底层张量：上一轮留 19.3 GB 时在 95 GB 峰值处 OOM |
| PLE 行在预取线程里查好 | 和下一批的分词一起做，每个样本约 1.2–1.8 s，和 GPU 并行 |
| 验证集每次现分词 | 一直存着 16 个分词好的样本要占 10–16 GB 主机内存 |
| 每个样本记录主线程缺页、内联 PLE 时间、编译的图、cgroup 内存、起止时间；每秒采样 GPU | 下次能直接看出慢在哪里 |

结果（每样本，`train_report.py`）：

| token | 前缀之后 | 模式 | 前向 | 反向 | token/s | GPU 忙 | 说明 |
|---|---|---|---|---|---|---|---|
| 97,366 | 33,385 | parts | 62.9 | 77.7 | 692 | 47% | 第一步：编译 24 个图 |
| 108,479 | 33,275 | parts | 15.1 | 9.8 | 4,368 | 92% | |
| 90,559 | 85,367 | layer | 28.0 | 37.8 | 1,377 | 73% | |
| 130,561 | 70,655 | layer | — | — | — | — | GPU OOM（4.98 GiB），跳过 |
| 110,446 | 48,019 | parts | 18.9 | 20.4 | 2,806 | 79% | |
| 119,656 | 58,052 | parts | 12.3 | 22.6 | 3,423 | 86% | |
| 119,051 | 82,533 | layer | 12.5 | 28.3 | 2,918 | 98% | 14 GB 溢出留在 GPU |
| 118,804 | 85,515 | layer | 12.1 | 28.7 | 2,910 | 100% | 16 GB 溢出留在 GPU |
| 102,662 | 34,567 | parts | 20.2 | 10.3 | 3,370 | 83% | |
| 99,334 | 34,044 | parts | 10.0 | 8.3 | 5,411 | 100% | |

- 所有样本主线程缺页 0（第一个 17），PLE 一直在缓存里（cgroup file 约 220 GiB）。PLE 被挤出的问题解决了。
- 按每种模式拟合的步时间算全部 352 个样本：平均约 3,860 token/s，每个 epoch 约 2.4 h（只算步，不含启动、验证）。这 10 个样本故意挑了长的，它们自己的平均是 2,216。
- 验证 loss：0.300 → 0.288（5 步之后，2 个验证样本）。
- OOM 的原因：模型最后的 hyper-connection mixer 在整段序列上把 [N, 4 × 2,560] 升到 FP32（130,561 × 10,240 × 4 B = 4.98 GiB）。在 CPU 上用 dispatch 记录整段长度的 FP32 分配找到的；改成按 token 分块（613ebff），输出和 LoRA 路径的梯度逐位一致。
- 演练结束后 watcher 把 pod 删了；你要求以后停掉而不是删除：现在 `POD_END=stop|delete`，默认 stop，pod 要带 /workspace 卷，停掉后 checkpoint 和环境都还在。

## 10-20 晚上 – 10-21 凌晨：两个 epoch 正式训练（51aqerqvxsbucu，secure H200，US-NC-1，$5.29/h，约 4.95 小时，约 $26）

- 704 个样本（354 × 2 epoch），0 个失败；6 个在 fla 的 chunk 内核里 OOM，换成 layer 重算后都跑完。582 个 parts、122 个 layer。平均 4,675 tok/s，墙钟 4.56 小时。
- 验证集损失：0.310（训练前）→ 0.28773（epoch 1）→ 0.28742（epoch 2）。两个 epoch 之间只差 0.1%（16 个验证样本，只算一次），分不出好坏；训练损失 0.2849 → 0.2771。选 epoch 2（`adapter_final` 和 `adapter_epoch2` 是同一个文件，sha256 a98679ba）。
- wandb 回放：https://forge.coreweave.com/wandb/jonathan_wang1999-epsilla/arc3-rft/runs/fxftrdw7
- 适配器、日志、数据集（rft_v1c）都拉回到 `runs/runpod/jonathanwang2022/rft-train/51aqerqvxsbucu/`，sha256 和 pod 上一致。H200 已停（卷 250 GB 还在，里面只有可以重下的东西）。

## 10-21 凌晨：合并 + 发布（CPU pod，$0.96/h，约 1 小时，约 $1）

- 合并：W' = W + (α/r)·B·A 存回 BF16。228 个张量、14 个分片；BF16 保留的更新比例中位数 1.03、最小 0.977；方向余弦中位数 0.95、最小 0.80（o_proj，更新只占权重的 0.16%）。这是普通 LoRA（只加在 BF16 的注意力和 GDN 投影上），不是 QLoRA；INT4 专家没动。
- 没在 GPU 上验证合并后的验证集损失（用户决定）。脚本 `verify_merged_pod.sh` 已写好，需要时在 H200 上从 HF 下载合并后的模型跑一遍。
- HF：`helldriver007/qwen3.8-flash-next-w4a16-rft-v1`（私有，账号升级成 PRO 才放得下）：完整 181.2 GB（含 22 个 PLE 文件）+ `rft/` 下的适配器和优化器状态。66 秒传完——HF 的 Xet 存储会跳过它已有的数据，只有合并过的分片是新的。上传后逐个比对大小和 PLE 的 sha256，全部一致。
- Kaggle：`jonathanwang2022/qwen3-8-flash-next-w4a16-rft-v1/Transformers/default/1`（私有，实例 768829）：只有不含 PLE 的 29 个文件（78.8 GB），PLE 用 `dfranzen/intel-qwen3.8-flash-next-w4a16-autoround` 的（逐字节相同）。用户定的规则：以后 HF 和 Kaggle 都不下载、不上传 PLE。Kaggle 的 slug 不能带点（`qwen3.8` 被拒），改成 `qwen3-8`。
- 坑：
  - 训练 pod 的卷有配额，硬链接按整份算；
  - 本地拆分 PLE 放不下；
  - CPU pod 不能挂持久卷；
  - 取消倒计时要先杀子 shell 再杀 sleep：顺序反了，第一台 CPU pod 被自己停掉，重开一台；
  - 被拒绝的工具调用其实在 pod 上跑了，差点同时跑两份脚本。
  - 详见 AGENTS.md。
- 还没做：serving 端把 base 模型的 `model-ple-*` 链接进合并后的模型目录（不做的话 Kaggle 这份不能单独加载）；`merge_upload_pod.sh` 加一个默认跳过 PLE 的选项。
