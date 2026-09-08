# 分布式与内存层级 { #distributed-and-memory-hierarchy }

多进程图执行与显式的内存层级核算。两个示例都不需要改动用户
[kernel](https://en.wikipedia.org/wiki/Compute_kernel)；放置与数据交换都由下层自动推导。

- [`python examples/distributed_halo.py`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/distributed_halo.py)
- [`python examples/hierarchical_memory.py`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/hierarchical_memory.py)
- [`python examples/paged_giant_graph.py`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/paged_giant_graph.py)
- [`python examples/auto_offload.py`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/auto_offload.py)

### 分布式执行 { #distributed-execution }

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

上面的示例为了可移植性在本机 spawn 两个进程。换成 MPI transport 后，
同一个程序就能跑在真实的 MPI launcher 下——rank 和 world size 都来自
MPI communicator，Tiga 不读任何 rank 环境变量：

```python
from mpi4py import MPI
import tiga as gf
from tiga.distributed import DistributedRuntime

world = MPI.COMM_WORLD
graph = gf.load("ring.gfg").halo(gf.DeviceMesh("cpu", world.Get_size()), depth=1)
with DistributedRuntime.from_provider("mpi", communicator=world):
    out = NeighborSum()(graph=graph, src={"x": local_x}, dst={})
```

```bash
mpiexec -n 2 python mpi_halo.py                  # 本机两个 rank
mpiexec -np 2 -H host1,host2 python mpi_halo.py  # 跨机器
```

跨机器运行不需要任何 Tiga 侧的改动：mpi4py 把 communicator（含
网络传输）交给运行时，runtime 从图推导出同样的 owner/ghost halo map。
又因为图是分页的 `.gfg`，每个 rank 只从磁盘读自己的行区间——没有
rank 会把完整邻接结构放进 RAM。已验证的配置是本机两个 MPICH 进程
（前向和 VJP，逐字节一致）；跨主机走完全相同的 API，NCCL 设备
transport 目前只验证了 rank-one 回环。

零依赖的替代方案是 stdlib TCP transport：rank 0 充当 rendezvous
监听方，其余 rank 通过它加入，随后整个世界建立带 rank 握手校验的
全连接 socket 网格：

```python
from tiga.distributed import DistributedRuntime, TCPTransport

transport = TCPTransport.host(rank=0, world_size=2, port=29617)                    # rank 0 监听
transport = TCPTransport.join(rank=1, world_size=2, host="10.0.0.1", port=29617)  # rank 1 拨号
with DistributedRuntime(transport):
    out = NeighborSum()(graph=graph, src={"x": local_x}, dst={})
```

MPI 是跨机器运行久经考验的路径；stdlib TCP transport 是零依赖的
选项，目前仅在 loopback 上验证过。

### 内存层级 { #memory-hierarchy }

## 把张量换出到磁盘 { #spilling-tensors-to-disk }

**它是什么。** 计算机把数据存储在*内存层级*中：小容量、高速、
昂贵的层级在前，后面是更大、更慢、更便宜的层级。当工作集超出
RAM 时，*spilling*（换出）会把张量移动到 [NVMe](https://en.wikipedia.org/wiki/NVM_Express)（Non-Volatile
Memory Express，非易失性存储）上并释放其缓冲区；之后的任何读取
都会透明地把它*恢复*回来。API 是张量上两个可链式调用的方法：
`.disk()` 写出，`.cpu()` 读回——文件由 Tiga 管理，无需
手动干预。

![内存层级阶梯——SMEM/寄存器、GPU HBM、RAM 与 NVMe，及其典型容量、带宽与延迟——以及 gf 管理的 RAM 与 NVMe 之间的 .disk()/.cpu() 通道](../assets/examples/hierarchical-memory.svg)

生命周期约定：

- **匿名** `x.disk()`——换出文件存放在每个进程各自的临时目录中，
  张量被回收或进程退出时即被删除；
- **具名** `x.disk(name="...")`——文件持久保存在
  `TIGA_SPILL_DIR`（或 `~/.cache/tiga/spill`）中，
  另一个进程可以用 `gf.from_disk(name)` 挂载它，载荷在首次使用时
  惰性加载。

该示例先把一个八元素张量沿 RAM → NVMe → RAM 换出，再以具名方式
重新换出一次。底层实现中，同一份字节会送入带容量/版本核算的
`HierarchyRuntime` 实例，供编译器 bundle 计划使用——参见
[内存层级与分布式执行](../memory-and-distributed.md)。

```python
--8<-- "examples/hierarchical_memory.py:core"
```

??? example "完整源码：examples/hierarchical_memory.py（可直接运行）"

    ```python
    --8<-- "examples/hierarchical_memory.py"
    ```

## 磁盘上的超大图 { #a-disk-resident-giant-graph }

**它是什么。** 当图本身——而不只是图上的张量——超出 RAM 时，*paging*
（分页）把拓扑（CSR 邻接结构）留在磁盘上，按有界的目的行页面读取，
任一时刻只有一页驻留内存。一个后台线程在计算第 k 页的同时从磁盘
*prefetch*（[预取](https://en.wikipedia.org/wiki/Cache_prefetching)）
第 k+1 页，因此 [kernel](https://en.wikipedia.org/wiki/Compute_kernel)
永远不必等待磁盘。由于目的行之间相互独立，每一页都走未改动的原生
MessagePassing 路径，任何 [reducer](https://en.wikipedia.org/wiki/Fold_(higher-order_function))
都能得到与全局完全一致的逐行结果。字段也可以随拓扑一起落盘：
`gf.save(graph, path, fields={"src": ..., "edge": ...})` 把它们写成定长
行，`gf.load(path).fields("src", requires_grad=True)` 返回惰性读取的壳，
此后前向与 `gf.autograd.grad` 都能分页执行——仅剩的边界是仅支持 CPU。

整个往返在同一个未改动的 kernel 上分三个阶段：

$$
\text{build} \xrightarrow{\texttt{gf.save}} \texttt{.gfg on disk}
\xrightarrow[\text{stream pages}]{\texttt{gf.load} + \text{MessagePassing}}
\text{result} \xrightarrow{\texttt{out.disk(name=...)}} \text{NVMe}
\xrightarrow{\texttt{gf.from\_disk}} \text{reattach} .
$$

该示例构建一个 1000×1000 的四邻点 stencil（100 万节点，约 400 万条
边），用 `gf.save` 持久化一次，再用 `gf.load` 重新打开——只读
manifest，CSR 留在磁盘上——然后以 10 万行为一页、开启 prefetch 执行
一步 Jacobi 平滑，把结果以具名方式换出到磁盘，最后用 `gf.from_disk`
重新挂载，验证校验和在磁盘 → JIT → 磁盘的往返后保持不变。

在 2000×2000 stencil（400 万节点，约 1600 万条边，NVMe ext4，Ryzen
7 255）上实测（`benchmarks/memory_hierarchy/paged_giant_graph.py`，
各行校验和完全一致）：

| page_rows | prefetch | 页数 | 耗时 | 峰值 RSS |
|---|---|---|---|---|
| 4,000,000（单页） | 关 | 1 | 34.1 s | 3.34 GB |
| 100,000 | 关 | 40 | 37.7 s | 1.08 GB |
| 100,000 | 开 | 40 | 36.6 s | 1.09 GB |

峰值 RSS 跟随页大小而非图大小（此处低 3.1 倍，图越大差距越大）。
prefetch 收回了分页引入的大部分额外开销；剩余差异来自页构建而非磁盘
延迟——存储越慢，overlap 的价值越大。

```python
--8<-- "examples/paged_giant_graph.py:core"
```

??? example "完整源码：examples/paged_giant_graph.py（可直接运行）"

    ```python
    --8<-- "examples/paged_giant_graph.py"
    ```

## RAM 预算下的自动 offload { #automatic-graph-offload }

**它是什么。** 上面的 `gf.save`/`gf.load` 显式往返是为跨进程交接准备的。
当目标只是"别让拓扑撑爆 RAM"时，预算机制替你做出分页决定：
`gf.runtime.auto_offload(ram=...)` 生效期间，每个 CSR 构造器
（`Graph.from_csr`、`Graph.stencil`、`Graph.cat` 等）都会检查 CSR 字节
数，超出预算就把拓扑持久化并返回 paged 图——kernel 调用不变，自动按页
流式执行并带 prefetch，全程不需要 `gf.save`/`gf.load`：

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
时清理；若需要跨进程存活，设置 `TIGA_SPILL_DIR`。仅剩的边界是仅支持
CPU 图；字段也可经 `gf.save(..., fields=...)` 落盘，且前向与反向都能
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
        executable cache: hits=0, misses=10
        ```
