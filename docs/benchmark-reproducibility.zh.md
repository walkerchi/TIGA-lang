# 图表复现 { #reproduce-the-charts }

**从存档重画图**不需要 GPU；**运行新测量**需要匹配的设备、依赖和计时范围。下文分别给出操作步骤。

## 无 GPU 重画 { #redraw}

```bash
python -m pip install matplotlib==3.11.1 numpy==2.4.4
python tools/verify_benchmark_evidence.py
python -m benchmarks.common.plot_docs_results --root benchmarks/evidence_snapshot/output
python tools/verify_benchmark_evidence.py --zip docs/assets/results/evidence.zip
```

## 重画全部已注册 case { #all-cases}

```bash
mkdir -p output/chart-reproduction
cp -R benchmarks/evidence_snapshot/output/roofline output/chart-reproduction/
python -m benchmarks.common.plot_cases --root output/chart-reproduction/roofline
python -m benchmarks.common.plot_collections --root output/chart-reproduction/roofline --report-output output/chart-reproduction/report.png --showcase-output output/chart-reproduction/showcase.png --interactive-report-output output/chart-reproduction/report.html
```


重画环境为 Python 3.12.3、Matplotlib 3.11.1、NumPy 2.4.4。SVG 去除时间戳并固定元素 ID；同一环境下字节稳定。跨平台字体差异可能改变排版，因此不保证跨字体环境的 SVG 字节一致。输入校验使用 [SHA-256](https://en.wikipedia.org/wiki/SHA-2)。

`output/chart-reproduction` 应使用新目录，避免把旧图误当本次产物。这里只写派生图片，不修改存档数值；退出码非零表示不能完成重画。

## 图表清单 { #inventory }

| 图 | 比较内容 | 必須保留的口径 |
|---|---|---|
| Sparse relations | Prepared Tiga / Torch sparse | 拓扑、F、索引 dtype、hot cache |
| Transformations | 四个固定编译器工作负载 | 各自基线，不是综合分数 |
| Primitive comparisons | 固定 manifest panel | 每行匹配数学语义与 provider |
| CPU relations | Tiga / SciPy 和 Torch | N=16k 与 131k、度 16、16 线程 |
| Edge-NN forward | Tiga / 手写与 eager 对标 | 仅前向，N=262144 |
| Edge-NN memory | Eager 中间量 / 融合表示 | 逐边存储估算，不是实测峰值 |
| Edge-NN backward | Compiled / eager autograd | 前向+反向；峰值分配是另一指标 |
| GAT | Compiled / eager autograd | 辅助 GAT case，不是 dense SDPA |
| Attention | Dense、linear、sparse attention | 不同工作负载、不同对标，不取平均 |
| Dynamic boundaries | 构建、消费、复用、全新管线 | 单阶段不能标成端到端 |
| Distributed overlap | 通信与内部计算 trace | 双 CPU 进程、受控延迟、sample 与中位数 |
| Provider comparison（Reference） | 同一个 CSR 切片的全部 provider | 只测 PyG gather-scatter，不代表所有 PyG 算子 |

这 12 组 SVG/PNG 均有[输入索引](assets/results/chart-inputs.json)，包含精确 JSON 路径与 SHA-256。[下载包](assets/results/evidence.zip)还包含全部 44 个已注册 case 和自己的 `index.json`。这些图表生成器不再手抄测量常数。

## 运行新测量 { #measure }

安装[基准依赖](benchmark-suite.zh.md)，匹配 case 的配置、dtype、设备、provider 版本、warmup、重复次数、seed 和计时边界。先查模块的 `--help` 再选择参数。以下命令记录一次新测量，不替换存档证据：

```bash
PYTHONPATH=python python tools/run_benchmark.py --record output/fresh-smoke/provenance.json --module benchmarks.sparse_compute.weighted_aggregation -- --quick
```

包装器记录精确命令、UTC 时间、退出码、Git revision、dirty worktree 指纹、包版本、CPU/platform 和可获取的 GPU/驱动信息。不把缺失 seed 或旧 revision 当作已知。实际输出目录与随机 seed 行为仍由基准模块决定；其有效参数和结果应与 provenance 文件一起保存。

将环境和源码元数据与对应测量的样本一起保存。评估新版本性能时，使用匹配对标并重复采样，参见[计时协议](performance.zh.md)。

## 图表与证据覆盖范围 { #audit }

文档生成器覆盖 12 组图和完整矩阵；通用生成器覆盖全部 44 个已注册 case 的 latency 图、43 张 roofline、五个多 case 汇总、dashboard 及静态/交互报告。缺少 roof 校准的 case 有意不生成 roofline。

Operation 汇总每张图最多三个条件，`SUMMARY.md` 链接全部分页。`benchmarks/common/diagnostic_plotting.py` 单独处理额外本地 JSON；Tensor fusion 的 kernel-only 与 end-to-end 计时分开呈现。报告使用的输入由 evidence manifest 指定。
