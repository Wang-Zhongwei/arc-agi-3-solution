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

