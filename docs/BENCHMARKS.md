# Benchmark and roofline protocol

状态：M0/M1 baseline  
日期：2026-08-07

## 1. Example × benchmark × provider matrix

| Query/workload | Public example | Accuracy oracle | Roofline | Comparable providers |
|---|---|---|---|---|
| weighted aggregation / GCN | native `examples/gcn.py` | `torch.sparse.mm` | `benchmarks/sparse_compute/weighted_aggregation.py` | Torch sparse, Torch scatter, Triton CSR, optional PyG/DGL |
| CPU fused weighted relation | native `examples/gcn.py` | provider-native CSR SpMV | `benchmarks/sparse_compute/cpu_relation.py` | GraphForge LLVM relation, SciPy CSR, Torch sparse |
| CPU pointwise fusion | native `examples/tensor_autograd.py` | identical Torch expression | `benchmarks/compiler/cpu_pointwise.py` | GraphForge Vector/LLVM, Inductor, Torch eager |
| scalar diffusion | native `examples/diffusion.py` | Torch `index_add` | `benchmarks/sparse_compute/diffusion_roofline.py` | Torch sparse Laplacian, Torch scatter, benchmark Triton CSR, optional PyG |
| generated radius aggregation | generated TTIR + runtime-owned CUDA launch | small all-pairs reference | `benchmarks/graph_operations/radius_build.py` + `radius_pipeline.py` | torch-cluster build, Torch sparse consume, external build+index_add |
| exact kNN build | `Graph.knn` procedural relation | selected Euclidean distance multiset | `benchmarks/graph_operations/knn_build.py` | exhaustive cdist/top-k |
| horizontal sum×max fusion | MLIR fusion tests | unfused Torch/Triton | `benchmarks/sparse_compute/fusion.py` | two benchmark Triton kernels vs one product-reducer oracle |
| MessagePassing source backward | compiler-derived transpose relation | explicit scatter/transpose SpMV | `benchmarks/autograd/message_passing_backward.py` | generated VJP, Torch autograd, Torch sparse transpose, handwritten Triton |
| MessagePassing edge-weight backward | compiler-derived checkpointed VJP | saved/recomputed gather-multiply | `benchmarks/autograd/message_passing_backward.py --gradient dweight` | generated VJP, Torch autograd, Torch explicit, matched handwritten Triton |
| dense FP16 matmul | native `examples/tensor_matmul.py` | `torch.mm` | `benchmarks/neural_networks/dense_matmul.py` | GraphForge direct TTIR, cuBLAS through Torch |
| causal linear recurrence | `examples/linear_recurrence.py` | official FLA recurrent | `benchmarks/neural_networks/linear_attention.py` | GraphForge structural map-scan-contract fusion, FLA recurrent/chunk |
| dynamic tile-pruned softmax | `examples/tile_pruned_attention.py` | exact Flash SDPA accuracy budget | `benchmarks/neural_networks/sparse_attention.py` | GraphForge generated conditional streaming reducer, official flash-sparse-attn, exact Flash SDPA |
| lower-triangular dense reducer | `Graph.triangular()` | causal Flash SDPA | `benchmarks/neural_networks/dense_attention.py --causal` | GraphForge triangular TTIR, tuned Triton parity oracle, Flash SDPA |
| grouped-query dense reducer | ordinary dense graph with Hq/Hkv grouped lanes | Flash SDPA GQA | `benchmarks/neural_networks/dense_attention.py --kv-heads 4` | GraphForge dense TTIR, tuned Triton parity oracle, Flash SDPA |
| GPU heatmap raster preparation | `examples/gpu_heatmap.py` | identical scalar-to-RGB expression | `benchmarks/visualization/heatmap.py` | GraphForge prepared Tensor TTIR, torch.compile/Inductor, Torch eager |

Radius 暂不宣称 SOTA comparison：必须把 neighbor build、materialization/generate-consume、
force kernel 和 rebuild/reuse policy 一起计时。只比较已经物化好的 edge consumer 会奖励错误的
系统边界。候选基线应在 cell-list backend 完成后选择 PyG radius graph、Taichi、HOOMD-blue
或手写 CUDA 中语义和边界条件真正一致者。

当前 tensorized uniform cell-list 已避免默认 Euclidean radius 的 `N×N` matrix；本机 N=32,768、
3D、目标 degree=32 的 build median 为 3.127 ms，展开 5.95M candidates、接受 0.978M edges。
case-local `output/roofline/radius_graph_build/<case>/roofline.json` 保存 raw samples，
同目录 PNG 显示 build latency 与
broad-phase amplification。`radius_pipeline.py` 进一步报告 build-only、consume-only 与
build+consume；默认 Euclidean distance-weighted sum 会融合 endpoint geometry，避免物化
displacement/distance/message。已接入并逐边验证 `torch_cluster.radius_graph`；该 uniform/
2D/3D × non-periodic/periodic box/skew-cell 现在明确拆分 consume-only、relation-reuse、
logical-rebind 与真正递增 position version 的 topology-rebuild。严格 `1.00× + CI low` gate 下，
consume/reuse/rebind、non-periodic rebuild 与 periodic box/skew rebuild 均已通过；完整 24-case
matrix 的 periodic rebuild 为 `3.861–6.064×`、CI low `3.769–5.909×`，不能用 rebind 的
cache hit 代替真实 topology rebuild。
非均匀密度与用户 metric/select 仍未外推；kNN 使用独立 workload/gate，不能由 radius 结果外推。

Dense matmul 已有独立而非 attention 内嵌的 calibration：native
`gf.Tensor` 保留 `gf_tensor.matmul` contraction；自动 provider 对合法 contiguous FP16
contraction 选择 Torch-free cuBLASLt，调试时可强制直接 tiled `tt.dot`。注册的 2048³
case 为 89.29 TFLOP/s，对比 external torch.mm/cuBLAS 88.86 TFLOP/s；
严格 gate 是 1.005x、95% CI low `1.004`，因此为 **PASS**，
artifact 位于
`output/roofline/dense_matmul_calibration/m2048_n2048_k2048_fp16/`。

exact kNN 的注册 N=8192/D=3/k=32 build+consume case 检查全部 N² pair，使用 provider
Euclidean cdist 再做 top-k；固定 degree 的 row pointer 可复用，但 column indices 每次重建并
重绑到同一个 compiler-generated TTIR weighted-consume executable，不触发重编译。Prepared
GraphForge 为 3.9072 ms，matched cdist/top-k/gather-multiply-sum 为 3.9137 ms，严格 gate
1.0017x、CI low 1.0005。该结论不外推其他 N/D/k 或 approximate kNN；build-only artifact
只保留为历史辅助结果。

线性 recurrence 的注册 `L=64,T=512,K=V=16,FP32` case 只向 compiler 提交普通
`broadcast × cumsum × reduce` Tensor IR。结构匹配将 map→scan→contract 融成一个 TTIR
kernel，rank-4 prefix state 不落 HBM；GraphForge 0.1056 ms，对 official FLA 最快 peer
0.1095 ms，严格门禁 1.037x、95% CI low 1.025。FLA chunk 的重关联误差单独记录，不冒充
bitwise oracle。

动态 tile-pruned reducer 的注册 `B=1,H=16,N=4096,D=64,FP16` case 使用与 official FSA
一致的阈值 `D/N`。阈值是 reducer IR 中的近似语义，generic dense lowering 在 key dot 后生成
动态 `scf.if`，只在 admitted tile 内加载 value/更新 accumulator；core 没有 attention 命名算子。
GraphForge 0.9073 ms、official FSA 1.0724 ms，严格门禁 1.182x、CI low 1.177。两种近似均
对 exact Flash SDPA 审计 relative-L2；provider-native layout repack 和 cold JIT 分开报告。
Roofline 的 x 轴使用相同 logical useful FLOP/common byte，未用无法观测的 admitted-tile FLOP
伪造更高算力。

Dense streaming 同时登记 full Cartesian 与 lower-triangular relation。两者都由同一个用户
MessagePassing UDF 和 online reducer lowering 产生；triangular boundary 不是 attention 特调，
而是 `Graph.triangular()` 的关系语义。B1/H16/N4096/D64/FP16 下 full 为 0.7720 ms、对
Flash SDPA 1.074x（CI low 1.069）；triangular 为 0.4459 ms、对 causal Flash SDPA
1.117x（CI low 1.100）。两者都与 tuned benchmark-only Triton parity oracle 单独列出。
Hq=16/Hkv=4 的 grouped-query case 不复制 KV，TTIR 以 query-lane group 映射 source lane；
0.7769 ms 对 Flash SDPA 0.8424 ms，为 1.084x（CI low 1.078）。

GPU heatmap 的注册 2048² FP32 case 计时同一 scalar-to-RGB 表达式与稳定输出 buffer。
GraphForge prepared Tensor TTIR 为 0.0996 ms，torch.compile/Inductor 为 0.1148 ms，严格门禁
1.153x、CI low 1.140；394 ms cold JIT 与 PNG encoding 都在热 kernel 门禁之外单列。

## 2. Query × cache 覆盖矩阵

`roofline.py`/`roofline_diffusion.py` 当前公开这些轴：

| Axis | Values | Purpose |
|---|---|---|
| topology | regular, irregular | fixed degree 与 zero/high-degree ragged rows |
| source locality | local, random | cache-friendly stencil-like 与随机 gather |
| cache regime | hot, cold | steady reuse 与显式大 buffer flush 后执行 |
| degree | CLI integer | traversal crossover / reducer work |
| feature width | 1, 16, 64 by default | scalar、vectorized SpMM、working-set growth |
| device | CPU, CUDA | reference portability 与 GPU provider |

推荐 nightly 至少覆盖：

```text
(regular, local,  hot) × degree {4,16,64} × feature {1,16,64,128}
(regular, random, cold) × degree {16,64} × feature {1,16,64}
(irregular, random, hot/cold) × mean-degree {16,64} × feature {1,16,64}
```

后续必须增加 index width `{i32,i64}`、dtype `{fp16,bf16,fp32,fp64}`、degree distribution
以及 reordered/original graph。结果 JSON 要保存完整 schema、软件版本和 GPU 名称，不能只保存
一条吞吐数字。

已登记的 social-like power-law slice 固定 N=131072、90% degree-8 / 9% degree-64 /
1% degree-256，覆盖 source locality `{local,random}`、index `{i32,i64}` 与 cache
`{hot,cold}`。public `prepared_auto` 逐 topology autotune chunked worklist 与 reusable-output
native CSR，八个 gate 全过，CI low `1.255–1.849`。i32 的 compiler-generated
CDF-bucket + register-resident chunked-tail TTIR 也单独 gate：local hot/cold 为
`1.508x/1.404x`，random 为 `1.353x/1.525x`，CI low `1.337–1.495`。该矩阵按 provider
逐样本轮转交错，避免长时间 cold flush 把温度/频率漂移归因给测量顺序。

连续 log-normal 与 exponential degree case 使用相同语义条件 `degree_max > 64` 进入
compiler plan，不按 topology 名称特调。generated TTIR hot/cold 分别为
`1.196x/1.157x` 与 `1.204x/1.439x`，四项 CI low 均大于 1；结果同时保留 auto/native
选择，因而可以区分编译器 kernel 能力与外部库 dispatch。

固定度 vector CSR 也走同一 public API，经 Domain/Iter/Kernel IR 生成
row-neighbor-feature TTIR。registered random/i32/N=131072/degree16 case 中，F16 的 hot/cold
speedup 为 `4.322x/3.801x`（CI low `4.245/3.759`），F64 为 `1.966x/1.302x`
（CI low `1.956/1.288`），baseline 均为 `torch.sparse.mm`。普通 lazy call 与
`prepared_auto` 分开计时，首次 provider JIT 单列为 `compile_ms`。

bounded-ragged vector CSR 会从 degree bounds 选择独立的 masked
row×neighbor×feature schedule，而不是复用 fixed-degree 地址公式。registered
irregular/random/i32/N=131072/degree 0–32 中，F16 hot/cold 为 `4.213x/3.270x`
（CI low `4.153/3.230`），F64 为 `1.595x/1.283x`（CI low `1.584/1.270`）。

## 3. Roofline definition

每次运行现场测量 hierarchical roofs：

- DRAM sustained bandwidth：256 MiB Tensor device-to-device copy；
- L2-sized bandwidth：source + destination working set 保守控制在 L2 容量内；
- FP32 compute ceiling：关闭 TF32 的 dense matrix multiplication；
- kernel FLOPs：由 workload 的语义公式计数；
- no-reuse algorithmic bytes：每条 edge 的 source/destination Field 都按 DRAM 读取；
- ideal-cache bytes：index/edge data 流式读取，node Field 理想地只读一次，output 写一次。

hot case 使用 L2 bandwidth，cold/flush case 使用 DRAM bandwidth。`roof%` 使用 ideal-cache
bytes 得到对应层级的 optimistic roof：

```text
optimistic_roof = min(measured_FP32_peak,
                      measured_L2_or_DRAM_bandwidth × FLOPs / ideal_cache_bytes)
```

`algorithmic_gbs_no_reuse` 可能高于物理 DRAM bandwidth，因为 cache reuse 会减少真实 DRAM
流量；它是算法流量指标，不冒充 profiler 测得的 DRAM bytes。后续用 Nsight Compute/厂商
profiler 增加真实 L2/DRAM transaction 数据。

不能对 hot graph 强套 DRAM roof：本机 RTX 5070 Ti 有 48 MiB L2，degree-16 scalar CSR 的
index/weight working set 可以大部分驻留其中；这样会出现 `>100% DRAM roof` 的假异常。

## 4. Fairness rules

- graph/build/preprocess、cold JIT、warm kernel、peak memory 分开报告；
- sparse linear algebra baseline 可以 amortize 固定 CSR preprocessing，但必须标注；
- dynamic graph 的 end-to-end 数字必须包含 builder；
- 所有 provider 先做 differential accuracy，再计时；
- optional provider 缺失或 ABI 初始化失败输出 `SKIP + reason`；
- GraphForge reference 只作为语义和 overhead baseline，不参与“最快 backend”结论；
- 数学表达、dtype、index width 与 determinism 必须一致。framework comparison 允许各自的
  native CSR/COO physical layout，但 conversion/preprocess 在 warm number 中被 amortize，
  必须另报成本；因此它是 provider-native execution comparison，不是同 kernel-layout 对比。

## 5. SOTA acceptance gate

GraphForge 生成 binary 不等于 performance backend 完成。每个公开宣称的
workload × target × dtype/index × shape bucket 都必须与该环境中实测最快的等价实现比较：

- 候选 baseline、输入 bucket 和调参预算在测量前登记，不能看到结果后删掉难例；
- 线性 sparse pattern 必须包含 vendor sparse library/`torch.sparse`；能够无额外物化地
  dispatch 到它们是合法且优先的 lowering；
- 自定义 fusion/nonlinear/generated relation 必须包含最佳可运行 Triton、TileLang 或
  target-native handwritten kernel；
- 每个 advertised bucket 要求 `best_baseline_median / graphforge_median >= 1.00x`，且速度比的
  bootstrap 95% 置信区间下界也达到 `1.00x`；不能只用 geomean 抵消退化，统计不确定时
  结论为未通过；
- warm consume、cold-cache consume、cold JIT、disk-cache hit 和包含 build/preprocess 的
  amortized end-to-end 分别判定，不能互相替代；
- provider 只有 correctness 而未过性能 gate 时保持 opt-in；默认 planner 必须 dispatch 到
  更快路径。

所有结果保存 raw samples、median、置信区间、provider/compiler commit、artifact hash、完整
设备信息和调参次数。微秒级 kernel 应增加独立进程重复，避免一次时钟或 cache 状态偶然性。

## 6. Commands

```bash
export PYTHONPATH="$PWD/python"

python3 -m benchmarks.sparse_compute.weighted_aggregation \
  --topology regular --locality local --cache both \
  --nodes 131072 --degree 16 --features 1,16,64

python3 -m benchmarks.sparse_compute.weighted_aggregation \
  --topology irregular --locality random --cache cold \
  --nodes 131072 --degree 32 --features 16,64

python3 -m benchmarks.sparse_compute.diffusion_roofline \
  --topology regular --locality local --cache hot \
  --nodes 131072 --degree 16

# Machine-readable per-bucket gate. This currently fails for the naive oracle.
python3 -m benchmarks.sparse_compute.weighted_aggregation \
  --topology regular --locality local --cache hot \
  --nodes 131072 --degree 16 --features 1,16,64 \
  --gate-provider triton.csr \
  --gate-baselines torch.sparse.mm \
  --fail-on-gate --json roofline.json

python3 -m benchmarks.graph_operations.radius_build \
  --device cuda --particles 32768 --dimensions 3 \
  --target-degree 32 --repeat 20

# Optional external peer matching Torch 2.11 + CUDA 12.8:
python3 -m pip install torch_cluster \
  -f https://data.pyg.org/whl/torch-2.11.0+cu128.html

python3 -m benchmarks.graph_operations.radius_pipeline \
  --device cuda --particles 32768 --dimensions 3 \
  --target-degree 32 --repeat 20 --fail-on-gate

# Uses an isolated temporary cache; it never deletes the user's Triton cache.
python3 -m benchmarks.compiler.jit_latency \
  --nodes 131072 --degree 16 --features 16 \
  --json jit-latency.json

# Read-only capacity plan for Graph500, Graphalytics, Friendster and OGB.
# Large datasets are never downloaded implicitly.
python3 -m benchmarks.large_graphs.plan \
  --json output/large_graphs/capacity_plan.json

# Backward-only gates. Relation/checkpoint materialization and JIT are reported
# separately and excluded from the warm backward measurement.
python3 -m benchmarks.autograd.message_passing_backward \
  --nodes 131072 --degree 16 --repeat 100 \
  --locality random --index-dtype i64 --fail-on-gate

python3 -m benchmarks.autograd.message_passing_backward \
  --gradient dweight --checkpoint auto --fail-on-gate
```

## 7. RTX 5070 Ti baseline（2026-08-07）

环境：PyTorch 2.11.0+cu128、Triton 3.6、regular local CSR、131,072 nodes、degree 16、
FP32、int64 index。PyG 2.8.0.post1 临时安装到 `/tmp`；DGL 2.1.0 wheel 与当前
Torch/Python dependency stack 未成功初始化，因此没有伪造 DGL 数字。

现场 hierarchical ceilings：

| DRAM copy | L2-sized copy | FP32 matmul（TF32 off） |
|---:|---:|---:|
| 766 GB/s | 1.54–1.55 TB/s | 33.4–33.9 TFLOP/s |

Weighted aggregation，hot/L2，数值为 CUDA-event median latency；`graphforge.auto` 的 event
在公开 Python 调用前记录，因此包含 dispatch 造成的 GPU idle，provider-native CSR/COO
construction 已 amortize。括号内是相对
`torch.sparse.mm` 的速度：

| Provider | F=1 | F=16 | F=64 | F=128 |
|---|---:|---:|---:|---:|
| Torch sparse | 0.028 ms (1.00x) | 0.204 ms (1.00x) | 0.280 ms (1.00x) | 0.450 ms (1.00x) |
| GraphForge auto | 0.026 ms (1.06x) | 0.033 ms (6.22x) | 0.125 ms (2.24x) | 0.243 ms (1.85x) |
| naive Triton CSR | 0.294 ms (0.10x) | 0.288 ms (0.71x) | 0.477 ms (0.59x) | 0.517 ms (0.87x) |
| GraphForge reference | 0.205 ms (0.14x) | 3.090 ms (0.07x) | 5.647 ms (0.05x) | omitted |

Scalar diffusion，hot/L2：

| Provider | latency | Gedge/s | hierarchical roof efficiency |
|---|---:|---:|---:|
| Torch sparse Laplacian | 0.036 ms | 58.4 | 49.2% |
| Torch `index_add` | 0.119 ms | 17.6 | 14.8% |
| GraphForge auto | 0.021 ms | 101.0 | 85.1% |
| GraphForge reference | 0.216 ms | 9.7 | 8.2% |
| generic dynamic-row Triton | 0.312 ms | 6.7 | 5.6% |

Torch sparse Laplacian 的 `row_weight` 在计时前预计算；这是 frozen CSR/weight 的合法 amortized
路径，但 dynamic weights 的 end-to-end benchmark 必须把该 preprocessing 加回来。

这些数字产生两个直接的 compiler 决策：

1. 线性 `mul/copy + sum` apply 必须有 structured recognition，优先 dispatch 到经过验证的
   sparse-library SpMM/SpMV，而不是强制生成自有 kernel；GraphForge 的价值是保留 relation
   语义后仍能识别回该结构。
2. generic dynamic-row Triton 不能替代 fixed-degree specialization。现有固定 degree=16 的
   `benchmarks/sparse_compute/diffusion.py` CSR oracle 明显更快；planner 至少需要 regular/fixed-degree 与
   ragged/dynamic 两套 guarded skeleton。

`output/irregular/`、`output/skewed_i32/`、`output/skewed_i64/` 是旧矩阵快照，不再承担当前
发布 gate。当前正式 evidence 只来自 `benchmarks/evidence_manifest.json` 登记 case；未重新登记的
i32、宽 feature 和 cold 组合不外推。Domain/Iter/Kernel/Task→TTIR 已是当前执行主链。

2026-08-13 的 compiler-generated scalar diffusion fixed-degree 2/4/8/16/32/64 hot-cache bucket
均通过最快 peer gate。相同 N=131,072、目标平均 degree=16 的 irregular `[0,32]` 为 `1.068x`
（CI low `1.053`），skewed `{8,32,64}` 为 `1.031x`（CI low `1.024`）。通用优化包括 destination
row-hoist、packed degree worklist、generic UDF task ABI、JIT autotune 与经 Task DAG 证明的同流
prepared batch；没有根据 Diffusion 类名选择 kernel。旧 `0.840x/0.73x` 数据只作为历史失败，
不代表当前 binary。

`degree_max=256` 的 scalar/no-node path 现复用 compact degree worklist，并让 `degree>64` 的每行
在单个 TTIR program 中跨 64-edge tiles 合并 reducer state；它不分配 partial buffer，也不产生
第二个 finalize launch。正式 artifact 使用 131,072 行、90% degree-8、9% degree-64、1%
degree-256、random locality、i64、hot cache：GraphForge public `auto` 0.0600 ms，Torch CSR
0.0618 ms，`1.030x PASS`，95% CI `[1.027, 1.039]`。cold compile 单独报告，不混入 warm
execute。JSON、roofline 和 latency plot 位于
`output/roofline/weighted_aggregation/powerlaw_random_cuda_i64_n131072_degree16_f1/`。此前
whole-domain chunk loop（0.0943 ms）以及更宽 direct-filter/multi-short-bucket 实验均已删除或
回退。当前登记 bucket 已接入 `auto`；size、tail ratio、i32/i64、local/random 与 cold cache
仍必须分别补 gate，再与 edge-balanced/merge-path、persistent CTA 比较。

Radius pipeline（uniform 3D、N=32,768、978,044 edges）使用 synchronized wall-clock：build-only
GraphForge 3.117 ms vs torch-cluster 5.958 ms（1.912x，95% CI `[1.908, 1.915]`）；frozen
snapshot consume 为 GraphForge 0.039 ms vs Torch sparse 0.052 ms（1.319x，95% CI
`[1.304, 1.350]`）；build+consume 为 3.245 ms vs 同一 GraphForge builder 加 Torch sparse
3.300 ms（1.017x，95% CI `[1.011, 1.020]`），external torch-cluster + index_add 为 6.077 ms。
最快 end-to-end 对照隔离 consumer；external pipeline 也保留为 eligible peer。
原始样本和 phase/CI 图在对应 case-local artifact 目录。

## 8. MessagePassing backward baseline（2026-08-13）

首个 backward gate 固定为 static regular CSR、131,072 nodes、degree 16、random locality、
FP32、i64，以及任意 dense upstream cotangent。前向 UDF 是
`y[dst] = sum(weight[e] * x[src])`；compiler 从 edge/reducer IR 推导
`dx = transpose(relation, weight) @ dy`，构造一次 immutable transpose snapshot，并将反向
apply 重新送入同一 Domain→Iter→Kernel→TTIR provider pipeline。用户没有写 backward，
GraphForge 源码也没有手写 Triton 反向 kernel；手写 kernel 只存在于 benchmark oracle。

| Provider | backward-only latency | Gedge/s | 相对最快 peer |
|---|---:|---:|---:|
| GraphForge generated VJP | 0.0585 ms | 35.85 | 1.012x vs handwritten |
| handwritten Triton oracle | 0.0592 ms | 35.43 | 1.000x |
| Torch sparse transpose SpMV | 0.0713 ms | 29.39 | 0.830x |
| Torch autograd | 1.2863 ms | 1.63 | 0.046x |

GraphForge 对最快 peer 的 bootstrap 95% CI 为 `[1.007, 1.020]`，因此该**单一 bucket**
通过 SOTA gate。冷成本单独为 compiler/provider JIT 333.07 ms、transpose snapshot
materialization 56.38 ms；它们没有混入 warm backward。结果与双面板 roofline 在
`output/roofline/message_passing_backward/regular_random_cuda_i64_n131072_degree16_dx/`。
第二个 gate 使用同一前向 UDF 自动导出
`dweight[e] = dy[dst(e)] * x[src(e)]`。PyTorch autograd 会保存 `(E,)` 的 `x[src]`，所以公平
比较必须区分 saved-state 与 recompute，而不能让只做一次 gather 的实现对比两次 gather 的
oracle。GraphForge 的 `auto` 在 canonical VJP IR 中插入 checkpoint candidate；
`gf-plan-tensor-checkpoints` 在 byte budget 下选择后，GPU memory physicalization 才将 selected
checkpoint 变成 backward input：

| Provider | policy | backward-only latency | Gedge/s |
|---|---|---:|---:|
| GraphForge generated VJP | save 8 MiB | 0.0158 ms | 132.40 |
| handwritten Triton matched | save 8 MiB | 0.0198 ms | 106.05 |
| Torch autograd | save 8 MiB | 0.0425 ms | 49.37 |
| handwritten Triton | recompute | 0.0543 ms | 38.64 |
| Torch explicit gather | recompute | 0.0891 ms | 23.53 |

相对最快 matched peer 为 1.248x，95% CI `[1.238, 1.257]`，通过该单桶 gate。首次 native
extension load 27.58 ms、MLIR checkpoint planner 1.57 ms、checkpoint compiler 260.22 ms、
materialize 5.88 ms、backward compiler 44.56 ms 均单独报告；合计 cold compile 为 333.93 ms。此 operation
的 roofline x 轴对所有 provider 完全相同；每种 policy 的 physical ideal bytes 与 saved bytes
保存在 JSON。copy-calibrated L2 线只是 indexed traffic 的诊断 proxy；该 kernel 超过 copy
proxy 不解释成超过物理带宽，必须等 profiler 的 L2/DRAM transaction counter 才报告真实 roof
efficiency。artifact 位于
`output/roofline/message_passing_backward/regular_random_cuda_i64_n131072_degree16_dweight/`。

scalar 结果不外推到 vector feature；后者已新增两个独立 matched bucket。对于 `weight[E,1]` 和 feature field
`x[N,F]`，compiler 从广播规则导出
`dweight[e] = sum_f(dy[dst(e),f] * x[src(e),f])`，生成 gather + per-edge reduction：

| F | GraphForge | matched saved Triton | speedup / 95% CI | saved primal |
|---:|---:|---:|---:|---:|
| 16 | 2.0489 ms | 2.0559 ms | 1.003x / `[1.003,1.004]` | 128 MiB |
| 64 | 2.3039 ms | 2.3126 ms | 1.004x / `[1.003,1.004]` | 512 MiB |

artifact 分别位于同一 operation 下的
`regular_random_cuda_i64_n131072_degree16_dweight_f16/` 与 `..._f64/`。每个 bucket 内所有
provider 仍共享同一 x 坐标。余量非常窄，且目前只覆盖 contiguous FP32/i64；power-law、任意
stride/dtype 与更多 reducer family 必须另建 matched gate。

## 8.1 Online reducer 与 dense matmul（2026-08-13）

standalone segmented weighted-softmax case 不调用内置 attention 算子。benchmark 定义一个名字
无关的三状态 reducer；compiler 从 identity/lift/combine/finalize 证明稳定 weighted algebra，
再关联为 max-reduce → exp → denominator/numerator sum-reduce。N=131,072、degree=32、FP32/i64
下 GraphForge 为 0.1792 ms（23.40 Gedge/s），手写 matched Triton 为 0.1717 ms
（24.43 Gedge/s），严格 gate 0.958x、95% CI low 0.949，故该 case 是可执行 measured evidence，
尚未达到 SOTA。artifact 位于
`output/roofline/online_softmax/cuda_n131072_degree32_f1/`。

native `gf_tensor.matmul` 始终保留 contraction IR。GPU auto 对 contiguous FP16 rank-2 contraction
选择 runtime-owned、Torch-free cuBLASLt library call；`GRAPHFORGE_MATMUL_PROVIDER=ttir` 可强制
compiler-emitted tiled `tt.dot` 以检查 TTIR/PTX。2048³ auto 为 0.1924 ms（89.29 TFLOP/s），
external `torch.mm`/cuBLAS 为 0.1933 ms（88.86 TFLOP/s），严格 gate 1.005x、95% CI low
1.004。library dispatch 和 baseline 使用同一 Tensor 语义、dtype/layout 与 stream，GraphForge
点不经过 Torch。artifact 位于
`output/roofline/dense_matmul_calibration/m2048_n2048_k2048_fp16/`。

## 9. JIT latency baseline（2026-08-07）

`benchmarks/compiler/jit_latency.py` 在两个独立且已完成 CUDA 初始化的 worker 中复用同一个临时
`TRITON_CACHE_DIR`。第一个 worker 面对空 cache，第二个 worker 面对第一个写入的 disk cache；
Python import、CUDA context 初始化和 Tensor 构造均在计时外。计时 kernel 是手写 generic
CSR SpMM oracle，不是 GraphForge 生成代码。

RTX 5070 Ti、Triton 3.6、131,072 rows、degree 16、F=16：

| phase | provider compile/cache | first launch/setup | ready-to-result | warm kernel |
|---|---:|---:|---:|---:|
| empty provider cache | 442.4 ms | 127.8 ms | 570.1 ms | 0.2862 ms |
| new process, disk hit | 252.6 ms | 2.0 ms | 254.6 ms | 0.2867 ms |

`module_load_ms` 在该 Triton 版本的公开 `CompiledKernel.function` 访问上接近零；实际 lazy
driver/launcher setup 体现在 first-launch 数字中，因此不能据此声称 CUDA module load 免费。
输出 JSON 同时保存 warm raw samples 和 `source/TTIR/TTGIR/LLVM IR/PTX/cubin` artifact sizes。

同日 quick gate（8,192 rows、degree 16、F=16、hot）的一次隔离运行将 naive
`triton.csr` 判为 `FAIL: 0.820x vs torch.sparse.mm, 95% CI [0.730, 0.884]`。这验证 gate
会保存 raw samples、考虑统计不确定性并拒绝当前 oracle，而不是为了让 CI 通过降低门槛。
更重要的是，cold ready-to-result 比 warm kernel 高约三个数量级；M1 必须复用
specialization、缩小生成 region、持久化更靠后的 artifact，并测量 topology reuse 的
break-even。

## 10. Visualization output

安装绘图依赖并从 machine-readable JSON 生成 PNG/SVG：

```bash
python3 -m pip install -r requirements-plot.txt
python3 -m benchmarks.common.plotting \
  --roofline output/roofline/weighted_aggregation/<case>/roofline.json \
  --output-dir output/roofline/weighted_aggregation/<case>
```

当前每个注册 case 输出 `roofline.png`、厂商榜单式竖向 subplot 的
`provider_latency.png`、`roofline.json` 和嵌入图像/SOTA gate 的 `REPORT.md`。每个 subplot
对应一个 kernel/config，method 使用跨图稳定 hue。绘图只读取 JSON，不重新运行
benchmark；因此报告可以在无 GPU 的机器生成。长期 GraphForge-native GPU visualization
设计见根目录 `PROJECT.md` 第 11 节，不能与当前 Matplotlib report 混为同一 implementation claim。

面向人的跨 case 视图由正式 manifest 生成：

```bash
python -m benchmarks.common.plot_collections
python -m benchmarks.common.plot_diagnostics
python -m benchmarks.common.plot_cases
```

`output/roofline/<operation>/summary.png` 在同一张图片内按数学条件分面；只有
topology/locality/dtype/periodic 等条件相同、仅输入规模变化的点才会连线。provider
颜色由完整方法名稳定映射，在所有图片中保持一致。数字 marker `1/2/3...` 标识方法；
当测量点重合时，数字 badge 只在显示坐标中绕真实锚点排开，不会用 jitter 篡改
roofline 坐标。`output/roofline/dashboard.png` 汇总正式 evidence coverage。

非 roofline JSON（编译生命周期、halo、层级存储、provider conformance、容量规划等）
也必须有同目录 PNG。`plot_diagnostics` 从原始 JSON 回放这些图，不重新执行 benchmark。
`output/` 是可再生本地产物并被 Git 忽略；公开文档所需的精选静态图应放在
`docs/assets/`，而不是提交整棵测量输出。
