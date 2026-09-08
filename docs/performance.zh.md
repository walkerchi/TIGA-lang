# 性能方法论 { #performance-methodology }

一项性能结论要被 Tiga 接受，前提是对比双方的程序语义、dtype、
shape、索引约定、缓存状态和计时边界完全一致。事实依据是原始样本和
设备元数据，不是截图。证据矩阵见
[benchmark results](benchmark-results.md)：它由
`benchmarks/evidence_manifest.json` 生成，注册的用例、过滤器或
provider 一旦有缺失，生成就直接失败——过期的图表不可能被悄悄沿用。

## 语义匹配 { #semantic-matching }

对比双方执行的必须是同一个数学程序：

- 数学表达式、dtype、索引宽度和确定性约定完全相同；
- 允许使用 provider 原生的 [CSR](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_%28CSR%2C_CRS_or_Yale_format%29)/COO 布局，但布局转换和预处理的开销单独报告，
  不计入热态数字——比较的是各家原生执行，而不是同一
  [kernel](https://en.wikipedia.org/wiki/Compute_kernel) 布局下的较量；
- 计时之前，先与声明的 oracle 做差分精度校验；
- Tiga 参考实现只充当语义与开销基线，不参与"最快 backend"的
  评定；
- 可选 provider 缺失或运行失败时记为 `SKIP + reason`，绝不编造数字。

## 缓存状态与覆盖维度 { #cache-states-and-coverage-axes }

热态用例复用稳定的工作集；冷态用例在执行前先用一次显式的大缓冲区遍历
把缓存冲掉。两种状态分开测量、分开设门禁——工作集装得进
[L2](https://en.wikipedia.org/wiki/CPU_cache) 的热态图，不能拿
[DRAM](https://en.wikipedia.org/wiki/Dynamic_random-access_memory) 档的
上限来要求。稀疏基准暴露以下维度：

| 维度 | 取值 | 作用 |
|---|---|---|
| topology | regular, irregular | 固定度数；或含零度/高度数行的不规则分布 |
| source locality | local, random | 缓存友好的模板访问；或随机 gather |
| cache regime | hot, cold | 稳定复用；或显式冲刷 |
| degree | CLI 整数 | 决定遍历策略的切换点和 reducer 工作量 |
| feature width | 默认 1, 16, 64 | 标量、向量化 SpMM、工作集增长 |
| device | CPU, CUDA | 参考实现可移植性；GPU provider |

推荐的每夜覆盖组合：

```text
(regular, local,  hot) × degree {4,16,64} × feature {1,16,64,128}
(regular, random, cold) × degree {16,64} × feature {1,16,64}
(irregular, random, hot/cold) × mean-degree {16,64} × feature {1,16,64}
```

结果 JSON 保存完整 schema、软件版本和设备名称，绝不只落一个吞吐数字。

## 测量规则 { #measurement-rules }

- 冷态的捕获/provider 编译与热态执行分开报告。
- 计时边界取到结果就绪为止，包含必需的同步。
- 采样循环内交错运行各 provider，避免温度和顺序造成偏差。
- 保存中位数、原始样本和 [bootstrap](https://en.wikipedia.org/wiki/Bootstrapping_%28statistics%29) [置信区间](https://baike.baidu.com/item/置信区间)。
- 全套图表中同一 provider 固定同色；重合的点用数值标记区分，测量位置
  不动。
- 只有语义和条件一致、输入规模是唯一变量时，才把数据点连成线。
- 分发到外部库的路径和正确性 oracle 必须明确标注，二者都不得冒充
  编译器生成的 TTIR。

## Roofline 定义 { #roofline-definition }

每次运行都在被测机器上实测分级 [roofline](https://en.wikipedia.org/wiki/Roofline_model) 上限：

- DRAM 持续带宽：256 MiB 张量的 device-to-device 拷贝；
- L2 档带宽：源和目标工作集都保持在 L2 内；
- FP32 算力上限：禁用 TF32 的稠密 [matmul](https://en.wikipedia.org/wiki/Matrix_multiplication)；
- kernel 的 FLOPs 按工作负载的语义公式计数；
- 无复用算法字节数：每条边的源/目标字段都计入 DRAM 流量；
- 理想缓存字节数：索引/边数据只流式读一遍，节点字段只读一遍，输出只写
  一遍。

热态用例对 L2 档，冷态/冲刷用例对 DRAM 档。`roof%` 按对应层级的
乐观上限计算：

```text
optimistic_roof = min(measured_FP32_peak,
                      measured_L2_or_DRAM_bandwidth × FLOPs / ideal_cache_bytes)
```

`algorithmic_gbs_no_reuse` 是算法层面的流量指标：缓存复用会压低真实
DRAM 流量，所以它可以超过物理带宽——绝不能把它当作 profiler 实测的
DRAM 字节数。

## 产物契约 { #artifact-contract }

```text
output/roofline/<operation>/<case>/
  roofline.json          # raw samples, statistics, model and metadata
  roofline.svg           # scalable static roofline
  provider_latency.svg   # human comparison; stable method colors
  roofline.png           # raster fallback only
  provider_latency.png   # raster fallback only
  REPORT.md              # boundary, peer, accuracy and exclusions

output/roofline/<operation>/
  summary.svg            # matching cases connected across input size
  summary.png            # fallback
  SUMMARY.md

docs/assets/charts/
  compiler-performance-report.html  # interactive registered-case view
```

绘图只读 JSON、绝不重跑基准，因此没有 GPU 的机器也能构建报告。
`output/` 可复现且被 Git 忽略；只有选定发布的产物会复制进
`docs/assets/`，优先 SVG/HTML，PNG 兜底。

## 公平性规则 { #fairness-rules }

- 构建/预处理、冷态 [JIT](https://en.wikipedia.org/wiki/Just-in-time_compilation)、热态 kernel 和峰值内存分开报告。
- 动态图的端到端数字包含构建开销；缓存命中复用绝不能记成重建。
- 基线可以把 CSR 构建这类固定预处理摊到多次运行里，但必须标注。
- 每个声明的用例都保留原始样本、中位数、bootstrap 置信区间、
  provider/编译器 commit、产物哈希、完整设备信息和调优预算。
- 微秒级 kernel 要在独立进程里重复运行，排除偶发的时钟或缓存状态。

## 验收门禁 { #acceptance-gate }

能编出二进制不等于有性能结果。每一个对外宣称的
工作负载 × 目标 × dtype/索引 × shape 桶，都必须在相同环境下击败实测
最快的等价实现：

- 候选基线、输入桶和调优预算在测量前注册；看到结果后不得丢掉难例；
- 线性稀疏模式必须拿厂商稀疏库或 `torch.sparse` 作对照——在不额外
  物化的前提下分发到这些库，本身就是合法且优先的 lowering；
- 自定义融合和生成的关系，必须拿可运行的最佳 Triton、TileLang 或目标
  平台原生手写 kernel 作对照；
- 一个桶通过的条件是 `best_baseline_median / graphforge_median >= 1.00×`，
  且 bootstrap 95% 置信区间下界也达到 1.00×；[几何平均值](https://baike.baidu.com/item/几何平均数)补不了失败的桶，
  统计不确定性一律按未通过处理；
- 热态复用、冷缓存复用、冷态 JIT、磁盘缓存命中和摊销端到端分开评判，
  互不替代；
- 通过正确性但没通过性能门禁的 provider 保持 opt-in；默认规划器必须
  分发到更快的路径。

## 复现 { #reproduce }

```bash
export TIGA_OPT="$PWD/build/bin/gf-opt"
export TIGA_TRANSLATE="$PWD/build/bin/gf-translate"

python -m benchmarks.compiler.provider_gate --fail-on-gate
python -m benchmarks.sparse_compute.weighted_aggregation
python -m benchmarks.graph_operations.radius_roofline
python -m benchmarks.neural_networks.dense_attention
python -m benchmarks.distributed.automatic_overlap

python -m benchmarks.common.check_outputs
python -m benchmarks.common.plot_cases
python -m benchmarks.common.plot_collections
```

最后一条命令重新生成各操作的汇总图、SVG/PNG 发布报告和交互式 HTML
报告。当前结论与排除项见
[benchmark results](benchmark-results.md)。
