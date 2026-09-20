# 分布式与内存层级 { #distributed-and-memory-hierarchy }

多进程图执行与显式的内存层级核算。两个示例都不需要改动用户
[kernel](https://en.wikipedia.org/wiki/Compute_kernel)；放置与数据交换都由下层自动推导。

- [`python examples/distributed_halo.py`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/distributed_halo.py)
- [`python examples/hierarchical_memory.py`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/hierarchical_memory.py)
- [`python examples/paged_giant_graph.py`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/paged_giant_graph.py)
- [`python examples/auto_offload.py`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/auto_offload.py)
- [`python examples/paged_cuda.py`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/paged_cuda.py)

CUDA 分页例子先保存图，再以 128 KiB 原生 buffer 预算在 CUDA 上打开，
用 NumPy 独立验证三轮递推前向。高级分页路径使用 `tg.Tensor`、host staging
及完整驻留输出；不支持 CUDA 分页 backward。参见[内存契约](../memory.zh.md#large-graphs-current-boundary)。

<span id="distributed-execution"></span>

## 双进程 halo 交换 { #two-process-halo-exchange }

**它是什么。** *[Sharding](https://en.wikipedia.org/wiki/Shard_(database_architecture))*（分片）把一张图切分到多个协作进程上，使
每个进程只拥有一部分节点及其数据。当某个节点的计算需要读取属于
另一个进程的邻居时，该邻居就是一个 *halo*（ghost）节点，其
值必须跨进程边界复制——这就是 *halo 交换*。该示例还对整个计算
自动求导：[VJP](https://en.wikipedia.org/wiki/Automatic_differentiation)（向量–雅可比积）把标量损失的梯度传播回输入，
无需手写任何反向传播。具体来说，图是一个 8 节点环，节点值为
`x[i] = i`；每个输出对节点自身及其两个环邻居的值求和。rank 0
拥有节点 0–3，rank 1 拥有节点 4–7，因此 `out[3]` 和 `out[4]` 各
需要来自对方 rank 的一个值。

![一个 8 节点环被切分到两个进程：rank 0 拥有节点 0–3，rank 1 拥有节点 4–7；x[4] 的 ghost 副本跨越边界，使 rank 0 可以计算 out[3]，其梯度也随之回流](../assets/examples/distributed-halo.svg)

$$
\mathrm{out}_i = x_{(i-1)\bmod 8} + x_i + x_{(i+1)\bmod 8},
\qquad x_i = i,
\qquad \frac{\partial \sum_i \mathrm{out}_i}{\partial x_j} = 3 .
$$

该示例保存一个带版本号的 `.gfg`，在两个进程上重新打开同一个普通
`Graph`，只读取各 rank 的目标节点页与边页，然后运行 rank 本地的
`Graph.halo()` [MessagePassing](https://en.wikipedia.org/wiki/Message_passing) 以及自动 VJP。前向的 owner→ghost
交换和反向的 ghost-cotangent→owner 累加，都由编译器/运行时在
未改动的用户 kernel 之下自动推导。

```python
--8<-- "examples/distributed_halo.py:core"
```

??? example "完整源码：examples/distributed_halo.py（可直接运行）"

    ```python
    --8<-- "examples/distributed_halo.py"
    ```

## 跨进程与跨机器运行 { #running-across-processes-and-machines }

上面的完整程序验证两个 CPU 进程的前向/VJP，不是多 GPU launcher。
两个 rank 的预期输出分别为 `[8,3,6,9]`、`[12,15,18,13]`；
源字段梯度均为 `[3,3,3,3]`。

验证 MPI transport 时，安装 `mpi`、`benchmarks` extras 及可用的 MPI launcher，
运行仓库中实际存在的
[mpi_halo_exchange.py](https://github.com/walkerchi/TIGA-lang/blob/main/benchmarks/distributed/mpi_halo_exchange.py)：

```bash
mpiexec -n 2 python -m benchmarks.distributed.mpi_halo_exchange \
  --entities 64 --features 4 --repeats 3 --output output/mpi-sanity/results.json
```

成功时 JSON 中 `correct=true`、`gate="PASS"`。这个命令验证 halo 字节传输，
不是 MessagePassing 梯度。跨机 MPI 还需要 launcher/网络配置、相同环境，
以及各 rank 都能访问的图快照，不能只替换一个 transport 名称便忽略部署条件。

**两台机器上的 CUDA 前向/VJP**使用[完整的双机验证命令](../memory-and-distributed.zh.md#two-host-cuda-sanity)。
每台机器启动一个进程，本地设备均可为 `cuda:0`。TCP 通过主机内存 staging；
[NCCL 路径](../memory-and-distributed.zh.md#two-host-nccl) 通过 NCCL 交换
device buffer，同样支持前向/VJP。

TCP 的 `host()` 与 `join()` 是**不同进程**中的阻塞入口，不能在同一个脚本中依次调用。
peer 必须可信（对象消息使用 pickle）、网络互通，并为冷 JIT 留出足够超时。
自动 halo 调度不包括发现机器或启动远程进程。

<span id="memory-hierarchy"></span>

## 把张量换出到磁盘 { #spilling-tensors-to-disk }

`spill()` 只改变驻留位置，保留逻辑设备与 autograd 历史；`cpu()` 是设备拷贝入口。`save/load` 则读写显式路径的持久化数值快照。预算、共享 view、文件生命周期和 CPU 分页限制见[内存与存储](../memory.zh.md)。

```python
--8<-- "examples/hierarchical_memory.py:core"
```

例子验证梯度为 `[0, 2, 4, 6, 8, 10, 12, 14]`，且快照恢复后的数值与原值一致。退出 execution 不会使 Tensor 失效；例子自身的临时目录负责清理演示快照。

## 磁盘上的超大图 { #a-disk-resident-giant-graph }

**它是什么。** 当图本身——而不只是图上的张量——超出 RAM 时，*paging*
（分页）把拓扑（CSR 邻接结构）留在磁盘上，按有界的目的行页面读取，
prefetch 可让多页同时驻留，并隐藏部分 I/O 延迟，不保证计算永不等待磁盘。
目的行相互独立，因此受支持的分页 reducer 与操作可保持逐行语义。
字段也可以随拓扑一起落盘：
`tg.save(graph, path, fields={"src": ..., "edge": ...})` 把它们写成定长
行，`tg.load(path).fields("src", requires_grad=True)` 返回惰性读取的壳，
此后前向与 `tg.autograd.grad` 可在受支持的 CPU/原生路径上分页执行。
构图、驻留字段、prefetch 队列和物化输出仍占内存；分页不是进程 RSS 的硬上限。

整个往返在同一个未改动的 kernel 上分三个阶段：

$$
\text{build} \xrightarrow{\texttt{tg.save}} \texttt{.gfg on disk}
\xrightarrow[\text{stream pages}]{\texttt{tg.load} + \text{MessagePassing}}
\text{result} \xrightarrow{\texttt{out.disk(name=...)}} \text{NVMe}
\xrightarrow{\texttt{tg.from\_disk}} \text{reattach} .
$$

该示例构建一个 1000×1000 的四邻点 stencil（100 万节点，约 400 万条
边），用 `tg.save` 持久化一次，再用 `tg.load` 重新打开——只读
manifest，CSR 留在磁盘上——然后以 10 万行为一页、开启 prefetch 执行
一步 Jacobi 平滑，把结果以具名方式换出到磁盘，最后用 `tg.from_disk`
重新挂载，验证校验和在磁盘 → JIT → 磁盘的往返后保持不变。

在 2000×2000 stencil（400 万节点，约 1600 万条边，NVMe ext4，Ryzen
7 255）上实测（`benchmarks/memory_hierarchy/paged_giant_graph.py`，
各行校验和完全一致）：

| page_rows | prefetch | 页数 | 耗时 | 峰值 RSS |
|---|---|---|---|---|
| 4,000,000（单页） | 关 | 1 | 34.1 s | 3.34 GB |
| 100,000 | 关 | 40 | 37.7 s | 1.08 GB |
| 100,000 | 开 | 40 | 36.6 s | 1.09 GB |

这组存档测量中，100,000 行的页将峰值 RSS 从 3.34 GB 降至约 1.09 GB，
完整调用耗时有所增加。耗时包含页读取与计算；使用目标硬件和页大小运行 benchmark，
可评估这一时间与内存的取舍。

```python
--8<-- "examples/paged_giant_graph.py:core"
```

??? example "完整源码：examples/paged_giant_graph.py（可直接运行）"

    ```python
    --8<-- "examples/paged_giant_graph.py"
    ```

## RAM 预算下的自动 offload { #automatic-graph-offload }

此阈值在 **CSR 构造完成后** 才检查，不是外存构图算法，也不限制进程 RSS
或完整工作集。构图、输入字段、prefetch 队列与输出仍须计入实际内存需求。

**它是什么。** 上面的 `tg.save`/`tg.load` 显式往返是为跨进程交接准备的。
每个 CSR 的大小阈值可以自动触发分页：
`tg.runtime.auto_offload(ram=...)` 生效期间，每个 CSR 构造器
（`Graph.from_csr`、`Graph.stencil`、`Graph.cat` 等）都会检查 CSR 字节
数，超出预算就把拓扑持久化并返回 paged 图——kernel 调用不变，自动按页
流式执行并带 prefetch，全程不需要 `tg.save`/`tg.load`：

```python
--8<-- "examples/auto_offload.py:core"
```

??? example "完整源码：examples/auto_offload.py（可直接运行）"

    ```python
    --8<-- "examples/auto_offload.py"
    ```

流水线深度——并发预取的页数——默认为 2，可按调用调整
（`prefetch_depth=4`）或按进程调整
（`TIGA_PAGED_PREFETCH_DEPTH`）；页大小由 `page_rows=` 或
`TIGA_PAGED_PAGE_ROWS` 控制。offload 的图落在进程级临时目录、退出
时清理；若需要跨进程存活，设置 `TIGA_SPILL_DIR`。此路径仅支持 CPU/原生图及受支持的分页操作 图；字段也可经 `tg.save(..., fields=...)` 落盘，且前向与反向都能
分页执行。
??? info "本机编译产物（Ryzen 7 255 · RTX 5070 Ti 实测）"

    === "本机 / this machine"

        ```text
        ### Smoothing
        backend: cpu:0
        provider: tiga-runtime
        lowering: gf-tensor-relation-autograd
        passes: bind-static-csr-snapshot, capture-message-passing-udf, analyze-reducer-algebra, lower-csr-to-gather-segment, fuse-edge-node-regions
        remark: [planning] edge/node UDFs remain in the differentiable Tensor DAG
        remark: [planning] gf-tensor-vjp generates CSR gather/segment-sum adjoints
        remark: [planning] reducer lowering: builtin-additive-state
        variant cache: hits=0, misses=10
        ```
