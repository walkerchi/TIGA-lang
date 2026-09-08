---
hide:
  - navigation
  - toc
---

<section class="gf-hero">
  <div class="gf-hero__copy">
    <div class="gf-eyebrow"><span></span> 面向 message passing 编程模型的编译器</div>
    <h1>写好 message passing，<br><em>高效运行在任何硬件上。</em></h1>
    <p class="gf-hero__lead">
      用 <code>Graph</code> 声明谁和谁相连，用 <code>MessagePassing</code>
      UDF 声明每条边上算什么；一次普通调用，编译器就把同一份代码 lower 成
      CPU/GPU 上的融合可执行程序，梯度自动生成。稀疏、稠密、生成式、分页
      与分布式关系，同一套写法。
    </p>
    <div class="gf-actions">
      <a class="gf-button gf-button--primary" href="getting-started/">构建第一个 kernel <span>→</span></a>
      <a class="gf-button" href="compiler-pipeline/">探索编译器</a>
    </div>
    <div class="gf-install"><code>pip install tiga-lang</code><span><strong>T</strong>arget-<strong>I</strong>ndependent <strong>G</strong>raph <strong>A</strong>cceleration——递归一点：Tiga Is a Graph Accelerator</span></div>
  </div>
  <div class="gf-hero__trace" aria-label="Tiga lowering stages">
    <div class="gf-trace__top"><span>捕获的程序</span><code>Diffusion()(graph, u)</code></div>
    <div class="gf-trace__rail" aria-hidden="true"></div>
    <div class="gf-trace__stage"><span>01</span><div><strong>Domain IR</strong><small>关系 · UDF · reducer</small></div><code>gf.apply</code></div>
    <div class="gf-trace__stage"><span>02</span><div><strong>Iter IR</strong><small>稠密 · 稀疏 · 生成式</small></div><code>gf.iter</code></div>
    <div class="gf-trace__stage"><span>03</span><div><strong>Kernel + Task IR</strong><small>tile · 内存 · halo 事件</small></div><code>gf.kernel</code></div>
    <div class="gf-trace__stage gf-trace__stage--accent"><span>04</span><div><strong>Provider 交接</strong><small>序列化的 TTIR 或 LLVM</small></div><code>TTIR</code></div>
  </div>
</section>

<div class="gf-proof-strip">
  <div><strong>Domain → Iter → Kernel</strong><span>渐进式 lowering，每一级都能查看</span></div>
  <div><strong>兼容 Torch</strong><span>零拷贝适配器；原生运行时无需 PyTorch</span></div>
  <div><strong>存储感知</strong><span>HBM、RAM、NVMe 与 halo Task IR</span></div>
  <div><strong>实测而非推测</strong><span>注册用例与置信度门槛</span></div>
</div>

## 精简的语义接口 { #a-small-semantic-surface }

<div class="gf-code-story" markdown>

```python
import tiga as gf

class WeightedAggregation(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        return edge.weight * src.x

out = WeightedAggregation()(
    graph=graph,
    src={"x": x},
    edge={"weight": weight},
)
```

<div class="gf-code-story__copy">
  <span class="gf-kicker">惰性 JIT，就是一次普通调用</span>
  <h3 id="no-explicit-compile-step-is-required">没有单独的编译步骤。</h3>
  <p>第一次调用时捕获并编译出一个带守卫条件的变体；只要图、形状、dtype、布局、目标和 provider 守卫仍然成立，之后的调用直接复用这份可执行文件。</p>
  <div class="gf-inspect-list"><code>program.explain()</code><code>program.ir("domain")</code><code>program.ir("kernel")</code><code>program.ir("gf.kernel.ttir")</code><code>program.code("ptx")</code></div>
</div>

</div>

## 保留的结构会改变算法 { #retained-structure-changes-the-algorithm }

这里只报告对齐过的、登记在册的工作负载。全部用例、
[置信区间](https://baike.baidu.com/item/置信区间)和排除项见[基准测试结果](benchmark-results.md)。

<figure class="gf-figure gf-figure--showcase">
  <object type="image/svg+xml" data="/assets/compiler-performance-overview.svg" aria-label="Tiga 编译器在六个对齐工作负载上的性能">
    <img src="/assets/compiler-performance-overview.png" alt="Tiga 编译器在六个对齐工作负载上的性能">
  </object>
  <figcaption>编译生成的路径里，既有与成熟原语持平的，也有靠消除物化、改变调度换来的结构性收益。<a href="benchmark-results/">查看证据 →</a></figcaption>
</figure>

## 从语义到 provider 代码 { #from-semantics-to-provider-code }

[GPU](https://en.wikipedia.org/wiki/Graphics_processing_unit) 一侧交接的是序列化的 TTIR；CPU 一侧是与之平行的
[MLIR](https://en.wikipedia.org/wiki/MLIR_%28software%29) 到 [LLVM](https://en.wikipedia.org/wiki/LLVM) 路径。

<figure class="gf-figure gf-figure--architecture">
  <object type="image/svg+xml" data="/assets/compiler-pipeline-overview.svg" aria-label="Tiga 编译器流水线：从 Python 捕获到 TTIR 或 LLVM">
    <img src="/assets/compiler-pipeline-overview.svg" alt="Tiga 编译器流水线：从 Python 捕获到 TTIR 或 LLVM">
  </object>
  <figcaption><a href="compiler-pipeline/">逐级查看 IR 与 pass 边界</a> · <a href="/assets/compiler-pipeline-overview.svg">打开完整尺寸的 SVG</a></figcaption>
</figure>

## 现在能跑什么？ { #what-is-executable-today }

| 方向 | 已可执行 | 明确边界 |
|---|---|---|
| 稀疏关系 | 标量/向量 CSR、有界不规则行、实测过的自然度数调度 | 覆盖范围仍取决于形状、dtype、索引宽度与分布 |
| 动态关系 | 生成式欧氏半径图、任意 `k≤64` 的精确 [kNN](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) | 通用度量 UDF、大 k 的 spill/任务计划、kNN 的反向传播仍未完成 |
| 稠密关系 | 笛卡尔/三角遍历、lane 分组、online reducer、tensor-core TTIR | 已注册的 [attention](https://en.wikipedia.org/wiki/Attention_%28machine_learning%29) 与 matmul 形状不代表普适的性能结论 |
| [张量](https://baike.baidu.com/item/张量) + VJP | 视图、广播、归约、扫描、matmul、复数、关系感知的自动 VJP | 不支持的组合会报错，或回退到明确标注的 correctness oracle |
| 内存 + 分布式 | 单节点 HBM/RAM/NVMe 规划、[MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) halo 执行、CUDA 任务排序 | 真实 2+ GPU 的 NCCL/RCCL 性能尚未声明 |

## 选择路径 { #choose-a-path }

<div class="gf-link-grid">
  <a class="gf-link-card" href="getting-started/"><span>01 · 使用</span><strong>构建并运行</strong><p>编译原生工具，跑通一个可微程序，并查看它的编译产物。</p></a>
  <a class="gf-link-card" href="programming-model/"><span>02 · 建模</span><strong>关系与 reducer</strong><p>理解 Graph、MessagePassing、reducer 代数与惰性特化。</p></a>
  <a class="gf-link-card" href="performance/"><span>03 · 验证</span><strong>复现结果</strong><p>了解对齐边界、原始证据、置信门槛，以及生成报告怎么读。</p></a>
</div>
