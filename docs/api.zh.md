# Python API { #python-api }

按模块组织的 API reference。右侧目录用于选择模块；每个模块开头提供简短的方法索引，
各条目保留完整签名、参数类型与使用示例。完整可执行代码见 [API 例子](api-examples.zh.md)。

示例统一使用 `import tiga as tg`；`tg` 是 Python 别名，不是 IR namespace。

## 默认应用接口：Torch { #torch-default }

使用此接口时，先单独安装 Torch，再安装 Tiga。Torch 不是默认依赖；
未安装 Torch 时，使用原生 `tg.tensor`、`tg.MessagePassing` 和 `tg.autograd`，
见[原生执行](execution.zh.md#native-execution)。

常规程序用 `torch.tensor` / `torch.randn` 创建字段，直接传给 `tg.Graph`
和 `tg.MessagePassing`，再对 Torch 输出使用 PyTorch autograd；不需要包装或另一套 Tensor API。
[完整 CPU 例子](api-examples.zh.md#torch-default) 包含空目标行，并验证源字段和边字段梯度。

| 意图 | 常规应用接口 | 原生编译器接口（高级） |
|---|---|---|
| 创建值 | `torch.tensor(...)` | `tg.tensor(...)` |
| 求导 | `torch.autograd.grad` / `loss.backward()` | `tg.autograd.grad` |
| 设备拷贝 | Torch `.to(device)` | 原生 `Tensor.to(device)` |
| 检查编译 | kernel `explain()` / 可用 `ir()` 产物 | 原生 `Tensor.mlir()` / `generated_code()` |
| 存储预算 | Torch 分配不纳入 Tiga LRU | `tg.execution` 管理作用域内原生 buffer |

`tg.Tensor` 是可独立运行的原生类型，也承载高级功能，不是 `torch.Tensor` 的别名。
下文 `Tensor.*` 均指**原生类型**。直接 Tensor 编译、staged loop 驱动器和
分布式/存储内部仍有原生专用约束；应用默认使用 Torch 不等于这些接口已自动支持任意 Torch 运算。

Torch CSR 的部分 CUDA provider 使用编译前向，并在 backward 通过 Torch 重放
固定拓扑上的表达式求导。这条路径可能物化逐边中间值，不等于编译后的 TTIR backward；
具体路径见 `kernel.explain()`。原生 VJP 和 edge-nn 专用编译 VJP 是不同的执行路径。

## 原生 Tensor 与 autograd（高级） { #tensor-and-autograd }

`float16/float32/float64`、`complex64/complex128`、`int32/int64` 与 `bool`
均为 Tiga 原生 dtype。只有浮点与[复数](https://baike.baidu.com/item/复数)值可以要求梯度。
广播遵循右对齐的 NumPy 语义。

### tg.tensor(data, *, dtype=None, device=None, requires_grad=False) { #gftensor }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `data` | <code>bool &#124; int &#124; float &#124; complex &#124; Sequence</code> | 标量或规则嵌套数值序列。 |
| `dtype` | <code>tg.DType &#124; None</code> | 原生数值类型；允许 None 的构造器按输入推断。 |
| `device` | <code>str &#124; tg.Device &#124; None</code> | 执行设备；None 按本条构造器的默认规则处理。 |
| `requires_grad` | <code>bool</code> | 是否记录浮点/复数输入的求导关系；默认 False。 |

返回类型：`tg.Tensor`。

??? example "最小用法"

    ```python
    import tiga as tg
    x = tg.tensor([1., 2.], requires_grad=True)
    ```

<!-- typed-contract:end -->

从嵌套 Python 序列或标量创建一个由 Tiga 原生运行时支撑的 Tensor。
省略 `dtype` 时按数据推断。`device=None` 使用当前 `tg.execution` 的默认设备，作用域外为 CPU。

### tg.empty(shape, *, dtype=tg.float32, device=None, requires_grad=False) { #gfempty }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `shape` | <code>int &#124; Sequence[int]</code> | 非负维度长度；元素值未初始化。 |
| `dtype` | <code>tg.DType</code> | 原生元素类型，默认 tg.float32；不接受 None。 |
| `device` | <code>str &#124; tg.Device &#124; None</code> | 执行设备；None 按本条构造器的默认规则处理。 |
| `requires_grad` | <code>bool</code> | 是否记录浮点/复数输入的求导关系；默认 False。 |

返回类型：`tg.Tensor`。

??? example "最小用法"

    ```python
    import tiga as tg
    buffer = tg.empty((2, 3), dtype=tg.float32)
    ```

<!-- typed-contract:end -->

分配一个未初始化的连续 Tensor。

### tg.zeros_like(value, *, requires_grad=False) { #gfzeros_like }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `value` | <code>tg.Tensor</code> | shape、dtype 和设备的模板。 |
| `requires_grad` | <code>bool</code> | 是否记录浮点/复数输入的求导关系；默认 False。 |

返回类型：`tg.Tensor`。

??? example "最小用法"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    result = tg.zeros_like(x)
    ```

<!-- typed-contract:end -->

分配一个与 `value` 的形状、dtype 和设备相同的全零 Tensor。

### tg.ones_like(value, *, requires_grad=False) { #gfones_like }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `value` | <code>tg.Tensor</code> | shape、dtype 和设备的模板。 |
| `requires_grad` | <code>bool</code> | 是否记录浮点/复数输入的求导关系；默认 False。 |

返回类型：`tg.Tensor`。

??? example "最小用法"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    result = tg.ones_like(x)
    ```

<!-- typed-contract:end -->

分配一个与 `value` 的形状、dtype 和设备相同的全一 Tensor。

### tg.from_torch(value, *, requires_grad=None) { #gffrom_torch }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `value` | <code>torch.Tensor</code> | 稠密 Torch Tensor；包装共享存储，不连接两套 autograd 历史。 |
| `requires_grad` | <code>bool &#124; None</code> | None 继承源标志；否则设置原生叶子的求导标志。 |

返回类型：`tg.Tensor`。

??? example "最小用法"

    ```python
    import torch
    import tiga as tg
    native = tg.from_torch(torch.tensor([1., 2.]))
    ```

<!-- typed-contract:end -->

零拷贝包装一个稠密 Torch tensor。示例：
[Torch 接口与原生存储桥](examples/programs-and-interop.zh.md#optional-torch-interoperability)。

### Tensor.reshape(*shape) { #tensorreshape }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `*shape` | <code>int &#124; Sequence[int]</code> | 新形状；最多一个 -1，总元素数不变。 |

返回类型：`tg.Tensor`。

??? example "最小用法"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    result = x.reshape(4)
    ```

<!-- typed-contract:end -->

返回一个新形状的惰性视图；一个 `-1` 维度会被推断。

### Tensor.permute(*axes) { #tensorpermute }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `*axes` | <code>int &#124; Sequence[int]</code> | 包含每个轴且不重复的排列。 |

返回类型：`tg.Tensor`。

??? example "最小用法"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    result = x.permute(1, 0)
    ```

<!-- typed-contract:end -->

返回轴重排后的惰性视图。

### Tensor.transpose(dim0, dim1) { #tensortranspose }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `dim0 / dim1` | <code>int</code> | 交换的两个轴；负轴编号会被规范化。 |

返回类型：`tg.Tensor`。

??? example "最小用法"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    result = x.transpose(0, 1)
    ```

<!-- typed-contract:end -->

返回交换两个轴后的惰性视图。`Tensor.T` 反转全部轴。

### Tensor.squeeze(dim=None) { #tensorsqueeze }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `dim` | <code>int &#124; None</code> | 删除指定的长度 1 轴；None 删除全部长度 1 轴。 |

返回类型：`tg.Tensor`。

??? example "最小用法"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    result = x.squeeze()
    ```

<!-- typed-contract:end -->

移除长度为 1 的维度；给定 `dim` 时只作用于该维。

### Tensor.unsqueeze(dim) { #tensorunsqueeze }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `dim` | <code>int</code> | 插入长度 1 的新轴的位置。 |

返回类型：`tg.Tensor`。

??? example "最小用法"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    result = x.unsqueeze(0)
    ```

<!-- typed-contract:end -->

在 `dim` 处插入一个长度为 1 的维度。

### Tensor.broadcast_to(shape) { #tensorbroadcast_to }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `shape` | <code>Sequence[int]</code> | 按右对齐规则可广播的目标形状。 |

返回类型：`tg.Tensor`。

??? example "最小用法"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    result = x.broadcast_to((3, 2, 2))
    ```

<!-- typed-contract:end -->

返回惰性广播视图。`Tensor.expand(*shape)` 是同一视图的变参写法。

### 算术、比较与 matmul 运算符 { #tensor-operators }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `other` | <code>tg.Tensor &#124; bool &#124; int &#124; float &#124; complex</code> | 兼容的标量或 Tensor 操作数；@ 要求矩阵操作数。 |

返回类型：`tg.Tensor`。

??? example "最小用法"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    result = (x + 1) * x
    mask = x > 2
    ```

<!-- typed-contract:end -->

`+`、`-`、`*`、`/`、一元 `-`、比较运算与 `@` 构建惰性表达式节点。
将 `Tensor` 用作 Python 布尔值（`if tensor:`）会抛出 `TypeError`。

### Tensor.matmul(other) { #tensormatmul }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `other` | <code>tg.Tensor</code> | 收缩维匹配的 rank-2 右操作数。 |

返回类型：`tg.Tensor`。

??? example "最小用法"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    result = x.matmul(x)
    ```

<!-- typed-contract:end -->

rank-2 矩阵乘法；复数输入使用共轭 Wirtinger VJP。示例：
[矩阵乘法与自动推导的梯度](examples/programs-and-interop.zh.md#matrix-multiplication-with-derived-gradients)。

### Tensor.exp() { #tensorexp }

<!-- typed-contract:start -->

无显式参数（实例方法的 `self` 省略）。

返回类型：`tg.Tensor`。

??? example "最小用法"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    result = x.exp()
    ```

<!-- typed-contract:end -->

逐元素指数。

### Tensor.sqrt() { #tensorsqrt }

<!-- typed-contract:start -->

无显式参数（实例方法的 `self` 省略）。

返回类型：`tg.Tensor`。

??? example "最小用法"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    result = x.sqrt()
    ```

<!-- typed-contract:end -->

逐元素平方根。

### Tensor.conj() { #tensorconj }

<!-- typed-contract:start -->

无显式参数（实例方法的 `self` 省略）。

返回类型：`tg.Tensor`。

??? example "最小用法"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    result = x.conj()
    ```

<!-- typed-contract:end -->

逐元素复共轭；实数 dtype 下为恒等。示例：
[复数 VJP](examples/programs-and-interop.zh.md#complex-vjp)。

### Tensor.cumsum(dim, *, reverse=False) { #tensorcumsum }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `dim` | <code>int</code> | 包含式扫描的轴。 |
| `reverse` | <code>bool</code> | True 从末端开始扫描；默认 False。 |

返回类型：`tg.Tensor`。

??? example "最小用法"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    result = x.cumsum(1, reverse=True)
    ```

<!-- typed-contract:end -->

沿一个轴的包含式扫描。示例：
[用 map/cumsum/contract 拼出线性递推](examples/programs-and-interop.zh.md#linear-recurrence-from-mapcumsumcontract)。

### Tensor.sum(axis=None, *, keepdims=False) { #tensorsum }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `axis` | <code>int &#124; Sequence[int] &#124; None</code> | 归约轴；None 归约全部轴。 |
| `keepdims` | <code>bool</code> | 保留长度 1 的归约轴；默认 False。 |

返回类型：`tg.Tensor`。

??? example "最小用法"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    result = x.sum(axis=1, keepdims=True)
    ```

<!-- typed-contract:end -->

沿一个轴、多个轴或全部轴归约。这里是原生接口 `axis` / `keepdims`；
普通 Torch Tensor 使用 `dim` / `keepdim`。

### Tensor.gather(index) { #tensorgather }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `index` | <code>tg.Tensor</code> | 同设备一维整数行号；允许重复。 |

返回类型：`tg.Tensor`。

??? example "最小用法"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    result = x.gather(tg.tensor([1, 0, 1], dtype=tg.int64))
    ```

<!-- typed-contract:end -->

按行取值；输入不能是标量。`index` 必须为同设备上的一维原生整数 Tensor。
输入形状 `(N, *F)`、索引长度 `K`，输出为 `(K, *F)`；索引须在 `[0, N)` 内。
VJP 是 segment sum，重复索引的梯度累加。

### Tensor.segment_sum(index, num_segments) { #tensorsegment_sum }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `index` | <code>tg.Tensor</code> | 每个输入行对应一个整数段号，与输入同设备。 |
| `num_segments` | <code>int</code> | 非负输出行数；所有段号须在范围内。 |

返回类型：`tg.Tensor`。

??? example "最小用法"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    result = x.segment_sum(tg.tensor([0, 0], dtype=tg.int64), 2)
    ```

<!-- typed-contract:end -->

按每行的段索引把行求和进 `num_segments` 个桶。对输入 `(N, *F)`，
`index` 是同设备、长度 `N` 的一维原生整数 Tensor；输出为 `(num_segments, *F)`。
`num_segments` 是非负整数，索引须在 `[0, num_segments)` 内，空桶输出零。

### Tensor.checkpoint() { #tensorcheckpoint }

<!-- typed-contract:start -->

无显式参数（实例方法的 `self` 省略）。

返回类型：`tg.Tensor`。

??? example "最小用法"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    result = x.checkpoint()
    ```

<!-- typed-contract:end -->

把一个非叶子值标记为显式的反向保存点（IR 中为
`gf_tensor.checkpoint`），对该值覆盖 checkpoint 规划器的选择。

### Tensor.realize() { #tensorrealize }

<!-- typed-contract:start -->

无显式参数（实例方法的 `self` 省略）。

返回类型：`tg.Tensor`。

??? example "最小用法"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    result = x.realize()
    ```

<!-- typed-contract:end -->

按所选执行策略物化延迟表达式，返回 Tensor。
`TIGA_TENSOR_BACKEND=native` 要求 JIT 编译；默认 `auto` 策略可能对小表达式
使用 Python oracle。

### Tensor.prepare() { #tensorprepare }

<!-- typed-contract:start -->

无显式参数（实例方法的 `self` 省略）。

返回类型：`Callable[[], tg.Tensor]`。

运行示例前，先按下文配置所需设备、文件或分布式环境。

??? example "最小用法"

    ```python
    # Requires a CUDA provider supporting prepared launches.
    import tiga as tg
    x = tg.tensor([1., 2.], device='cuda')
    prepared = (x * 2).prepare()
    result = prepared()
    ```

<!-- typed-contract:end -->

在首次 `realize()` 之后返回一个零参数的热 callable，以稳定的
输入/输出缓冲区重复提交已编译的 DAG。冷编译开销仍可通过
`Tensor.execution` 观察到。

backend 没有 prepared submission，或当前启用 `tg.execution(eviction="lru")` 时，
抛出 `RuntimeError`；固定地址不能参与自动换出。callable 复用存储，每次调用不是一份独立不可变的输出。

### Tensor.execution { #tensorexecution }

<!-- typed-contract:start -->

无显式参数（实例方法的 `self` 省略）。

返回类型：`dict[str, object] | None`。

??? example "最小用法"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    result = (x * 2).realize()
    print(result.execution)
    ```

<!-- typed-contract:end -->

属性，在物化后返回 codegen/启动诊断：语义哈希、编译/启动/物化耗时、
MLIR checkpoint 计划、已保存字节数以及所选 backend。执行前为 `None`；
已物化的输入，以及无需执行的常量或 view 结果，也可能保持 `None`。

### Tensor.generated_code(kind=None) { #tensorgenerated_code }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `kind` | <code>str &#124; None</code> | 产物名称，如 llvm 或 ptx；None 选择默认产物，是否存在取决于实际执行路径。 |

返回类型：`str | bytes | None`。

运行示例前，先按下文配置所需设备、文件或分布式环境。

??? example "最小用法"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    result = (x * 2).realize()
    print(result.generated_code())
    ```

<!-- typed-contract:end -->

返回真实的编译器阶段产物，如 `"gf_tensor"`、`"cpu_loop"`、`"llvm"` 或
`"ptx"`；缺省时选取已生成的最低层阶段。指南：
[编译器管线](compiler-pipeline.zh.md#inspecting-a-compiled-program)。

### Tensor.mlir(*, verify=False) { #tensormlir }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `verify` | <code>bool</code> | True 调用原生 IR verifier；默认 False。 |

返回类型：`str`。

??? example "最小用法"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    print((x * 2).mlir(verify=True))
    ```

<!-- typed-contract:end -->

返回规范化的、与目标无关的 `gf_tensor` IR；`verify=True` 会用原生
解析器与 C++ verifier 做一次往返校验。这是编译器契约；
`Tensor.expression()` 只是非正式的调试捕获。

### Tensor.tolist() / Tensor.to_numpy() / Tensor.to_torch(*, copy=False) { #tensor-host-interop }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `copy` | <code>bool</code> | 仅用于 to_torch：False 共享 Torch-owned 存储；True 允许拷贝原生连续存储。 |

返回类型：`Python scalar/list / numpy.ndarray / torch.Tensor`。

??? example "最小用法"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    values = x.tolist()
    array = x.to_numpy()
    torch_copy = x.to_torch(copy=True)
    ```

<!-- typed-contract:end -->

`tolist()` 返回嵌套 Python 值（shape 为 `()` 时返回标量），`to_numpy()` 返回主机拷贝；
两者都会观察/物化值，NumPy 是可选依赖。`to_torch(copy=False)` 仅接受
**Torch-owned 存储**，例如未改变存储的 `tg.from_torch(x)`；原生分配结果抛出 `RuntimeError`。
`copy=True` 允许同设备的连续原生存储拷贝。共享存储本身不会连接两套 autograd 图。

### tg.autograd.grad(output, inputs, *, grad_output=None, allow_unused=False, checkpoint="auto") { #gfautogradgrad }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `output` | <code>tg.Tensor</code> | 参与原生求导的输出，不接受 Torch Tensor。 |
| `inputs` | <code>tg.Tensor &#124; Sequence[tg.Tensor]</code> | requires_grad=True 的求导目标；传序列时梯度以 tuple 返回。 |
| `grad_output` | <code>tg.Tensor &#124; None</code> | 与输出 shape/dtype/device 一致的 cotangent；非标量或复数输出必填。 |
| `allow_unused` | <code>bool</code> | True 对未连接输入返回 None；否则报错。 |
| `checkpoint` | <code>Literal[&#x27;auto&#x27;, &#x27;save&#x27;, &#x27;recompute&#x27;]</code> | 反向所需前向值的存储策略。 |

返回类型：`tg.Tensor | None | tuple[tg.Tensor | None, ...]`。

??? example "最小用法"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    dx = tg.autograd.grad((x * x).sum(), x)
    assert dx.tolist() == [[2., 4.], [6., 8.]]
    ```

<!-- typed-contract:end -->

构建函数式的反向模式 [VJP](https://en.wikipedia.org/wiki/Automatic_differentiation) 表达式，不改动输入。
`checkpoint` 选择反向所需前向值的存储方式：`save` 把非叶子值标记为
保存，`recompute` 保留完整前向表达式，`auto` 生成
`gf_tensor.checkpoint_candidate` 交给与目标无关的 checkpoint 规划器。
非标量或复数输出必须显式给出 `grad_output`，其 shape、dtype、device 必须与输出一致。
仅接收原生 Tensor；每个求导输入都须 `requires_grad=True`。单个输入返回单个梯度，
输入序列返回 tuple；不连接到输出的输入默认报错，`allow_unused=True` 时返回 `None`。
不写入输入的 `.grad`。普通 Torch 求导使用 `torch.autograd.grad`。示例：
[矩阵乘法与自动推导的梯度](examples/programs-and-interop.zh.md#matrix-multiplication-with-derived-gradients)。

### tg.autograd.value_and_grad(function, *, argnums=0) { #gfautogradvalue_and_grad }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `function` | <code>Callable</code> | 指定参数可微的原生 Tensor 函数。 |
| `argnums` | <code>int &#124; Sequence[int]</code> | 求导的位置参数编号；默认 0。 |

返回类型：`Callable`。

??? example "最小用法"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    f = tg.autograd.value_and_grad(lambda a: (a * a).sum())
    value, dx = f(x)
    ```

<!-- typed-contract:end -->

返回 `function` 的一个变换：同时产出前向输出和 `argnums` 选定的
位置参数的梯度，不涉及 `.grad` 变更。

### tg.autograd.joint_plan(output, inputs, *, checkpoint="auto") { #gfautogradjoint_plan }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `output` | <code>tg.Tensor</code> | 参与原生求导的输出，不接受 Torch Tensor。 |
| `inputs` | <code>tg.Tensor &#124; Sequence[tg.Tensor]</code> | requires_grad=True 的求导目标；传序列时梯度以 tuple 返回。 |
| `checkpoint` | <code>Literal[&#x27;auto&#x27;, &#x27;save&#x27;, &#x27;recompute&#x27;]</code> | 反向所需前向值的存储策略。 |

返回类型：`JointAutogradPlan`。

??? example "最小用法"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    plan = tg.autograd.joint_plan((x * x).sum(), x)
    value, dx = plan.run()
    ```

<!-- typed-contract:end -->

构建一个可执行、可检查的联合前向/反向任务 DAG，使用同一份快照版本。
`plan.run()` 返回 `(output, gradients)`；`plan.explain()` 渲染该
bundle。示例：
[联合前向/反向 DAG](examples/programs-and-interop.zh.md#joint-forwardbackward-dag)。

### tg.autograd.grad_mlir(output, input, *, grad_output=None, lower=True) { #gfautogradgrad_mlir }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `output` | <code>tg.Tensor</code> | 参与原生求导的输出，不接受 Torch Tensor。 |
| `input` | <code>tg.Tensor</code> | 单个 requires_grad 求导输入。 |
| `grad_output` | <code>tg.Tensor &#124; None</code> | 与输出 shape/dtype/device 一致的 cotangent；非标量或复数输出必填。 |
| `lower` | <code>bool</code> | 是否执行 VJP lowering；False 保留显式请求。 |

返回类型：`str`。

??? example "最小用法"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    print(tg.autograd.grad_mlir((x * x).sum(), x))
    ```

<!-- typed-contract:end -->

以 [MLIR](https://en.wikipedia.org/wiki/MLIR_(software)) 文本返回显式 VJP 请求，或（`lower=True`）其
`gf-tensor-vjp` 结果，供检查。

## Graph { #graph }

连接源节点与目标节点的不可变逻辑关系，可接收 Torch 或原生索引 Tensor。物化 [CSR](https://en.wikipedia.org/wiki/Sparse_matrix) 拥有两个索引张量；
dense、triangular、radius 与 kNN 关系保持隐式或过程式，绝不会被静默
物化成 O(N²) 邻接。

### Graph.from_csr(row_ptr, col_idx, *, num_src=None, sorted_by_dst=True, validate="basic") { #graphfrom_csr }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `row_ptr` | <code>torch.Tensor &#124; tg.Tensor</code> | 一维 int32/int64 目标行边界，长度为 num_dst + 1。 |
| `col_idx` | <code>torch.Tensor &#124; tg.Tensor</code> | 按边顺序排列的一维源编号，与 row_ptr 同设备、同 dtype。 |
| `num_src` | <code>int &#124; None</code> | 源节点数；None 按索引推断。保留孤立源节点时显式提供。 |
| `sorted_by_dst` | <code>bool</code> | 目标分组声明，默认 True；CSR 行已经按目标分组。 |
| `validate` | <code>Literal[&#x27;basic&#x27;, &#x27;full&#x27;]</code> | 基础结构检查，或完整索引/边界验证。 |

返回类型：`tg.Graph`。

??? example "最小用法"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    assert graph.schema.num_dst == 3
    ```

<!-- typed-contract:end -->

创建目的行为单位的 CSR 关系。默认传入同设备上的一维整数 Torch Tensor，
也接受原生 Tensor。目标 `i` 接收 `row_ptr[i]:row_ptr[i+1]` 中的边，
对应源节点记录在 `col_idx`；边字段必须遵循该顺序。相邻 row pointer 相等表示空目标行。
`num_dst = len(row_ptr) - 1`。`num_src` 默认是最大源索引加一，空列时取 `num_dst`；
需要保留孤立源节点时显式指定。`validate="full"` 检查端点、单调性与索引范围，
非法 CSR 抛出 `ValueError`。例子见[默认 Torch 输入输出](api-examples.zh.md#torch-default)。

### Graph.from_coo(src, dst, *, num_src=None, num_dst=None) { #graphfrom_coo }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `src / dst` | <code>torch.Tensor &#124; tg.Tensor</code> | 等长一维整数源/目标编号，同设备、同 dtype。 |
| `num_src` | <code>int &#124; None</code> | 源节点数；None 按索引推断。保留孤立源节点时显式提供。 |
| `num_dst` | <code>int &#124; None</code> | 目标节点数；省略时的推断规则见本条说明。 |

返回类型：`tg.Graph`。

??? example "最小用法"

    ```python
    import torch
    import tiga as tg
    graph = tg.Graph.from_coo(torch.tensor([0, 1]), torch.tensor([1, 0]), num_src=3, num_dst=3)
    ```

<!-- typed-contract:end -->

从坐标列表创建同样的冻结关系，内部做一次按目标点的稳定排序，写成
规范 CSR。

### Graph.regular(num_nodes, degree, *, device="cpu") { #graphregular }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `num_nodes` | <code>int</code> | 节点数。 |
| `degree` | <code>int</code> | 每个节点固定的入邻居数。 |
| `device` | <code>str &#124; tg.Device &#124; None</code> | 执行设备；None 按本条构造器的默认规则处理。 |

返回类型：`tg.Graph`。

??? example "最小用法"

    ```python
    import tiga as tg
    graph = tg.Graph.regular(4, 2)
    ```

<!-- typed-contract:end -->

创建确定性的定度数关系——目标 `i` 从源 `(i * degree + k) % num_nodes`
收集——基准测试与冒烟测试不再手工组装 `arange`/`%` CSR 数组。

### Graph.dense(num_src, num_dst=None, *, device=None, index_dtype=tg.int64) { #graphdense }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `num_src` | <code>int</code> | 非负源节点数。 |
| `num_dst` | <code>int &#124; None</code> | 目标节点数；None 使用 num_src。 |
| `device` | <code>str &#124; tg.Device &#124; None</code> | 执行设备；None 按本条构造器的默认规则处理。 |
| `index_dtype` | <code>tg.DType</code> | tg.int32 或 tg.int64；默认 tg.int64。 |

返回类型：`tg.Graph`。

??? example "最小用法"

    ```python
    import tiga as tg
    graph = tg.Graph.dense(3, 2)
    assert graph.num_edges == 6
    ```

<!-- typed-contract:end -->

创建隐式笛卡尔关系；不分配 `N×N` 索引张量。`num_dst` 缺省等于
`num_src`。示例：
[全量注意力，无掩码](examples/attention.zh.md#full-attention-no-mask)。

### Graph.triangular(num_entities, *, device=None, index_dtype=tg.int64) { #graphtriangular }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `num_entities` | <code>int</code> | 非负节点数；目标 i 从 0 到 i 接收。 |
| `device` | <code>str &#124; tg.Device &#124; None</code> | 执行设备；None 按本条构造器的默认规则处理。 |
| `index_dtype` | <code>tg.DType</code> | tg.int32 或 tg.int64；默认 tg.int64。 |

返回类型：`tg.Graph`。

??? example "最小用法"

    ```python
    import tiga as tg
    graph = tg.Graph.triangular(3)
    assert graph.num_edges == 6
    ```

<!-- typed-contract:end -->

创建隐式下三角含对角关系（`src <= dst`）：在 IR 中仍是图拓扑，
任何 MessagePassing reducer 都可使用；dense lowering 使用有界
源 tile 加配对掩码，而不物化 CSR。示例：
[因果稠密关系](examples/attention.zh.md#causal-dense-relation)。

### Graph.cu_seqlens(cu_seqlens, *, causal=True) { #graphcu_seqlens }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `cu_seqlens` | <code>torch.Tensor &#124; tg.Tensor</code> | 从零开始、单调不减的一维序列累计边界。 |
| `causal` | <code>bool</code> | True 使用三角块；False 使用稠密块。 |

返回类型：`tg.Graph`。

??? example "最小用法"

    ```python
    import torch
    import tiga as tg
    graph = tg.Graph.cu_seqlens(torch.tensor([0, 2, 5]), causal=True)
    ```

<!-- typed-contract:end -->

为 flash-attn 风格的打包变长序列创建物化的块对角关系：序列 `k` 中的
位置 `i` 从源 `[s_k, i]`（`causal=True`）或从所属序列整体
（`causal=False`）收集。`cu_seqlens` 必须以 0 开头且单调。示例：
[带 cu_seqlens 的 varlen 因果注意力](examples/attention.zh.md#varlen-causal-attention-with-cu_seqlens)。

### Graph.cat(graphs) { #graphcat }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `graphs` | <code>list[tg.Graph] &#124; tuple[tg.Graph, ...]</code> | 非空、同设备子图；分别平移源/目标编号，不添加跨块边。 |

返回类型：`tg.Graph`。

??? example "最小用法"

    ```python
    import tiga as tg
    graph = tg.Graph.cat([tg.Graph.triangular(2), tg.Graph.triangular(3)])
    assert graph.num_edges == 9
    ```

<!-- typed-contract:end -->

把多个关系组合成一个块对角关系：块 `k` 的目标点只与块 `k` 的源点
相连，索引按源点累计数量偏移。`cat([Graph.triangular(n) for n in lengths])`
等价于把 `[0, *cumsum(lengths)]` 转为整数 Tensor 后传给 `Graph.cu_seqlens(..., causal=True)`；用 `Graph.dense`
块则等价于 `causal=False` 变体。组合结果是一份物化的 CSR 快照。示例：
[带 cu_seqlens 的 varlen 因果注意力](examples/attention.zh.md#varlen-causal-attention-with-cu_seqlens)。

### Graph.stencil(dims, offsets=None, *, periodic=False, device="cpu") { #graphstencil }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `dims` | <code>tuple[int, ...]</code> | 按行主序编号的正整数网格尺寸。 |
| `offsets` | <code>tuple[tuple[int, ...], ...] &#124; tg.stencil.Neighborhood &#124; None</code> | 整数源偏移或邻域宏；None 默认 von_neumann(radius=1, include_center=True)。 |
| `periodic` | <code>bool</code> | True 对越界坐标取模回绕；否则省略，不是零填充。 |
| `device` | <code>str &#124; tg.Device &#124; None</code> | 执行设备；None 按本条构造器的默认规则处理。 |

返回类型：`tg.Graph`。

??? example "最小用法"

    ```python
    import tiga as tg
    graph = tg.Graph.stencil((3, 3), tg.stencil.von_neumann())
    assert graph.num_edges == 33
    ```

<!-- typed-contract:end -->

在按行主序线性化的节点上创建物化的规则网格 stencil 关系；每个目标点
按 offsets 顺序从 `coord(dst) + offset` 收集。非周期边界截断网格外
源点，周期边界按取模回绕。None 默认使用半径为 1、含中心的 von Neumann 宏。
当前尚未支持 zero/constant padding。示例：
[邻域 MACRO 与五点均值](examples/dynamic-relations.zh.md#stencil-neighborhood-macros)、
[无矩阵 FEM 算子与求解循环](examples/solvers.zh.md#matrix-free-fem-operator-and-solver-loop)、
[磁盘上的超大图](examples/distributed-memory.zh.md#a-disk-resident-giant-graph)。

### stencil.von_neumann(radius=1, *, include_center=True) { #stencilvon_neumann }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `radius` | <code>int</code> | 正整数闭区间半径，默认 1，不接受 bool。 |
| `include_center` | <code>bool</code> | 是否包含零偏移中心点，默认 True。 |

返回类型：`tg.stencil.Neighborhood`。

??? example "最小用法"

    ```python
    import tiga as tg
    macro = tg.stencil.von_neumann(include_center=False)
    graph = tg.Graph.stencil((4, 4), macro)
    ```

<!-- typed-contract:end -->

返回与维数无关的 [von Neumann neighborhood](https://en.wikipedia.org/wiki/Von_Neumann_neighborhood)
宏：各偏移分量绝对值之和 ≤ radius。默认在二维为五点、三维为七点邻域。
`Graph.stencil` 从 `dims` 自动推断维数。

### stencil.moore(radius=1, *, include_center=True) { #stencilmoore }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `radius` | <code>int</code> | 正整数闭区间半径，默认 1，不接受 bool。 |
| `include_center` | <code>bool</code> | 是否包含零偏移中心点，默认 True。 |

返回类型：`tg.stencil.Neighborhood`。

??? example "最小用法"

    ```python
    import tiga as tg
    macro = tg.stencil.moore(include_center=False)
    graph = tg.Graph.stencil((4, 4), macro)
    ```

<!-- typed-contract:end -->

返回与维数无关的 [Moore neighborhood](https://en.wikipedia.org/wiki/Moore_neighborhood)
宏：偏移分量绝对值的最大值 ≤ radius。默认在二维为九点、三维为 27 点邻域。
设置 `include_center=False` 去掉零偏移中心点。

### stencil.Neighborhood.offsets(ndim) { #stenciloffsets }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `ndim` | <code>int</code> | 正整数网格维数，不接受 bool。 |

返回类型：`tuple[tuple[int, ...], ...]`。

??? example "最小用法"

    ```python
    import tiga as tg
    offsets = tg.stencil.von_neumann(include_center=False).offsets(2)
    assert offsets == ((-1, 0), (0, -1), (0, 1), (1, 0))
    ```

<!-- typed-contract:end -->

按字典序展开整数偏移，不分配设备数据，不导入 Torch。
不可变的宏对象包含 `kind`、`radius` 与 `include_center` 属性。
手写 offsets 元组则保持传入顺序。邻域大小随半径和维数增长；此接口会实际物化偏移。

### Graph.radius(positions, cutoff, *, exclude_self=True, fields=None, metric=None, select=None, periodic=None) { #graphradius }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `positions` | <code>torch.Tensor &#124; tg.Tensor</code> | 形状 (N, D) 的浮点坐标，位于图所在设备。 |
| `cutoff` | <code>float</code> | 正的距离阈值。 |
| `exclude_self` | <code>bool &#124; None</code> | 是否排除自环；默认值取决于本构造器。 |
| `fields` | <code>Mapping[str, Tensor] &#124; None</code> | 关系 builder 字段，与 MessagePassing 绑定不同。 |
| `metric` | <code>Callable &#124; None</code> | 可选距离 UDF；None 为欧氏距离。 |
| `select` | <code>Callable &#124; None</code> | 可选选边 UDF；不支持的编译组合会明确失败。 |
| `periodic` | <code>Tensor &#124; Sequence &#124; None</code> | 盒长 (D,) 或晶格 (D,D)；None 不使用周期边界。 |

返回类型：`tg.Graph`。

??? example "最小用法"

    ```python
    import torch
    import tiga as tg
    graph = tg.Graph.radius(torch.tensor([[0., 0.], [0.5, 0.]]), cutoff=1.)
    ```

<!-- typed-contract:end -->

从 rank-2 浮点位置创建可重建的过程式关系；邻接按当前快照重新计算。
`metric`/`select` 是自定义 builder UDF，`periodic` 提供盒长 `[D]` 或
晶格向量 `[D,D]`。示例：
[可微的半径关系](examples/dynamic-relations.zh.md#differentiable-radius-relation)。
回调的 `(P,D)` 输入、`(P,)` 返回值、cosine 示例与 all-pairs 性能限制见
[自定义距离](examples/dynamic-relations.zh.md#distance-metrics)。
`metric` 接受函数或 None，不接受距离名称字符串。

### Graph.knn(positions, k, *, candidates=None, exclude_self=None) { #graphknn }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `positions` | <code>torch.Tensor &#124; tg.Tensor</code> | 形状 (N, D) 的浮点坐标，位于图所在设备。 |
| `k` | <code>int</code> | 每个目标的最近候选数，受实体数量与 backend 限制。 |
| `candidates` | <code>torch.Tensor &#124; tg.Tensor &#124; None</code> | 源坐标 (M,D)；None 为 positions 自 kNN。 |
| `exclude_self` | <code>bool &#124; None</code> | 是否排除自环；默认值取决于本构造器。 |

返回类型：`tg.Graph`。

??? example "最小用法"

    ```python
    import torch
    import tiga as tg
    graph = tg.Graph.knn(torch.tensor([[0., 0.], [1., 0.], [3., 0.]]), k=1)
    ```

<!-- typed-contract:end -->

创建精确的过程式 [kNN](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) 关系；除非显式物化或编译器选定的使用方需要，
否则不分配邻接结构。省略 `candidates` 时为自 kNN 并默认
`exclude_self=True`；传入 candidates 时创建二分的 query→candidate
关系。示例：
[精确 kNN 接入 MessagePassing](examples/dynamic-relations.zh.md#exact-knn-feeding-messagepassing)。
目前仅支持 Euclidean，不能传 `metric=`。Cosine kNN 须先归一化非零查询与候选向量，
详见[自定义距离](examples/dynamic-relations.zh.md#distance-metrics)。

### Graph.open(path, *, device="cpu") { #graphopen }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `path` | <code>str &#124; os.PathLike</code> | 文件系统路径，不是 URL；父目录和覆盖规则见下文。 |
| `device` | <code>str &#124; tg.Device &#124; None</code> | 执行设备；None 按本条构造器的默认规则处理。 |

返回类型：`tg.Graph`。

运行示例前，先按下文配置所需设备、文件或分布式环境。

??? example "最小用法"

    ```python
    import tiga as tg
    graph = tg.Graph.open('saved-graph.gfg')
    ```

<!-- typed-contract:end -->

打开一个持久化图而不急于加载 CSR 数组；结果是由分页存储支撑的
普通 `Graph`，在规划器请求有界行范围之前只读取清单（manifest）。
`tg.load(path, *, device="cpu")` 是模块级别名。示例：
[磁盘上的超大图](examples/distributed-memory.zh.md#a-disk-resident-giant-graph)。

### tg.save(graph, path, *, fields=None) { #gfsave }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `graph` | <code>tg.Graph</code> | 本次调用的关系；设备、实体数须与字段匹配。 |
| `path` | <code>str &#124; os.PathLike</code> | 文件系统路径，不是 URL；父目录和覆盖规则见下文。 |
| `fields` | <code>Mapping[str, Mapping[str, tg.Tensor]] &#124; None</code> | 图快照中可选的 src/dst/edge 原生字段 payload。 |

返回类型：`None`。

运行示例前，先按下文配置所需设备、文件或分布式环境。

??? example "最小用法"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    tg.save(graph, 'new-graph.gfg')
    ```

<!-- typed-contract:end -->

以带版本号的 `.gfg` 格式持久化静态 CSR 图（manifest 加
`row_ptr.bin`/`col_idx.bin`）；`tg.load` 负责重新打开。
`fields={"src": {...}, "dst": {...}, "edge": {...}}` 会把节点/边字段
额外写成定长行，使其随拓扑一起从磁盘分页读取。

### Graph.fields(role, *, requires_grad=None) { #graphfields }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `role` | <code>Literal[&#x27;src&#x27;, &#x27;dst&#x27;, &#x27;edge&#x27;]</code> | 读取对应角色保存的字段，仅支持分页图。 |
| `requires_grad` | <code>bool &#124; None</code> | 原生叶子的求导标志；默认 None 按 False 处理。 |

返回类型：`dict[str, tg.Tensor]`。

运行示例前，先按下文配置所需设备、文件或分布式环境。

??? example "最小用法"

    ```python
    import tiga as tg
    graph = tg.load('saved-graph.gfg')
    source_fields = graph.fields('src', requires_grad=True)
    ```

<!-- typed-contract:end -->

返回句柄继承 Graph 的逻辑设备并持有底层 reader；attach 不分配完整字段。
CPU 分页支持 VJP，CUDA 分页目前仅支持前向。参见[内存边界](memory.zh.md#large-graphs-current-boundary)。

在用 `fields=` 保存的 `paged_csr` 图上，返回该角色下惰性读取的字段壳
（完整 shape、payload 在磁盘）映射，可直接传入 kernel 调用。
`requires_grad=True` 让壳可微；CPU 分页执行同时支持前向与
`tg.autograd.grad`。CUDA 分页执行会在开始计算前拒绝需要梯度的字段。

### graph.halo(mesh, *, partition=None, depth="auto") { #graphhalo }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `mesh` | <code>tg.DeviceMesh</code> | 逻辑设备网格；不会启动 worker 进程。 |
| `partition` | <code>tg.ByDestination &#124; None</code> | 目标分区策略；None 使用默认策略。 |
| `depth` | <code>int &#124; Literal[&#x27;auto&#x27;]</code> | 非负 halo 深度或交由编译器推断。 |

返回类型：`tg.Graph`。

??? example "最小用法"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    placed = graph.halo(tg.DeviceMesh('cpu', 2))
    ```

<!-- typed-contract:end -->

返回带有 owned/ghost 需求的同一种逻辑 `Graph` 类型；`partition` 缺省
为 `tg.ByDestination()`。这是声明式的：通信以类型化的 pack、
exchange、unpack、interior 和 boundary 任务插入到用户 kernel 之下。
示例：
[双进程 halo 交换](examples/distributed-memory.zh.md#two-process-halo-exchange)。

### graph.paged_rows(begin, end) { #graphpaged_rows }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `begin / end` | <code>int</code> | 分页图内左闭右开的目标行范围。 |

返回类型：`tuple[tuple[int, ...], tuple[int, ...]]`。

运行示例前，先按下文配置所需设备、文件或分布式环境。

??? example "最小用法"

    ```python
    import tiga as tg
    graph = tg.load('saved-graph.gfg')
    row_ptr, col_idx = graph.paged_rows(0, 2)
    ```

<!-- typed-contract:end -->

编译器/运行时钩子，返回一个有界的 CSR 目标行分页；只对用 `tg.load`
打开的 `paged_csr` 图有效。

### graph.resolve_csr() { #graphresolve_csr }

<!-- typed-contract:start -->

无显式参数（实例方法的 `self` 省略）。

返回类型：`tuple[torch.Tensor | tg.Tensor, torch.Tensor | tg.Tensor]`。

??? example "最小用法"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    result = graph.resolve_csr()
    ```

<!-- typed-contract:end -->

显式物化并返回 `(row_ptr, col_idx)`——这是检查或调试请求，不属于
正常的编译器消费路径。

### graph.transpose() { #graphtranspose }

<!-- typed-contract:start -->

无显式参数（实例方法的 `self` 省略）。

返回类型：`tg.Graph`。

??? example "最小用法"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    result = graph.transpose()
    ```

<!-- typed-contract:end -->

返回一个交换了端点角色的物化快照。

### graph.explain() { #graphexplain }

<!-- typed-contract:start -->

无显式参数（实例方法的 `self` 省略）。

返回类型：`str`。

??? example "最小用法"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    result = graph.explain()
    ```

<!-- typed-contract:end -->

渲染一行摘要：origin、lifecycle、realization、实体与边数量、设备，
以及存在的 halo/分页后端。示例：
[磁盘上的超大图](examples/distributed-memory.zh.md#a-disk-resident-giant-graph)。

### Graph 内省属性 { #graph-properties }

<!-- typed-contract:start -->

无显式参数（实例方法的 `self` 省略）。

返回类型：`GraphSchema / tg.Device / int | None / GraphPlacement | None / bool`。

??? example "最小用法"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    print(graph.schema, graph.device, graph.num_edges)
    print(graph.placement, graph.is_distributed)
    ```

<!-- typed-contract:end -->

`graph.schema`（`GraphSchema` dataclass）、`graph.device`、
`graph.num_edges`（过程式关系在实现前为 `None`）、`graph.placement`
与 `graph.is_distributed`。

### 规划器统计 { #graph-planner-statistics }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `samples` | <code>int</code> | 仅用于 source_index_span_ratio：采样行数，默认 4096。 |

返回类型：`Planner statistics (method-specific)`。

??? example "最小用法"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    print(graph.degree_bounds())
    print(graph.degree_statistics())
    print(graph.fixed_degree())
    ```

<!-- typed-contract:end -->

`graph.degree_bounds()`、`graph.degree_statistics()`、
`graph.degree_histogram()`、`graph.fixed_degree()` 与
`graph.source_index_span_ratio(samples=4096)` 暴露规划器所消费的
物理统计量；结果按快照缓存。

## MessagePassing { #messagepassing }

subclass `MessagePassing`，设置类属性 `reducer`，并实现
`edge(src, dst, edge, **params)`；`node(dst, aggregate, **params)`
可选，缺省为恒等。节点字段通过 `src={...}`/`dst={...}` 端点角色映射
绑定，或通过同构图上的单个 `ndata={...}` 映射。指南：
[消息传递](message-passing.zh.md)。

### program(graph=..., src=None, dst=None, edge=None, ndata=None, **params) { #messagepassing-call }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `graph` | <code>tg.Graph</code> | 本次调用的关系；设备、实体数须与字段匹配。 |
| `src` | <code>Mapping[str, torch.Tensor &#124; tg.Tensor] &#124; None</code> | 源字段，首维为 num_src；未使用该角色时传 {}。 |
| `dst` | <code>Mapping[str, torch.Tensor &#124; tg.Tensor] &#124; None</code> | 目标字段，首维为 num_dst；未使用该角色时传 {}。 |
| `edge` | <code>Mapping[str, torch.Tensor &#124; tg.Tensor] &#124; None</code> | 按图的边顺序绑定的字段；None 表示没有显式边字段。 |
| `ndata` | <code>Mapping[str, torch.Tensor &#124; tg.Tensor] &#124; None</code> | 同构图节点字段简写，不能与 src/dst 混用。 |
| `**params` | <code>object</code> | edge/node 签名声明的标量或可捕获参数；不代表支持任意 Python 对象。 |

返回类型：`torch.Tensor | tg.Tensor`。

??? example "最小用法"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    class Sum(tg.MessagePassing):
        reducer = tg.sum()
        def edge(self, src, dst, edge):
            return src.x
    program = Sum()
    x = torch.tensor([2., 3.], requires_grad=True)
    out = program(graph=graph, src={'x': x}, dst={})
    out.sum().backward()
    assert x.grad.tolist() == [1., 2.]
    ```

<!-- typed-contract:end -->

调用 kernel 实例选择执行路径。默认传入 Torch 字段，返回普通 `torch.Tensor`，
可直接参与 Torch 求导；全原生字段返回 `tg.Tensor`。原生 CSR/radius
调用捕获延迟 Tensor 表达式；读取结果时根据 backend 策略选择
[JIT](https://en.wikipedia.org/wiki/Just-in-time_compilation) 或 oracle 执行。

| 参数 | 契约 |
|---|---|
| `graph` | 受支持的关系；构造器可用不等于 backend 覆盖 |
| `src`、`dst` | 除非使用 `ndata`，否则两个映射都必须提供；未使用的角色传 `{}` |
| `edge` | 边字段映射；首维等于边数 |
| `ndata` | 仅用于同构图，不能与 `src/dst` 混用 |
| `**params` | 由 edge/node 方法签名声明的附加参数 |

节点字段须匹配角色实体数和图设备。角色映射缺失或 `ndata/src/dst` 混用
抛出 `TypeError`；尺寸/设备不符或二部图使用 `ndata` 抛出 `ValueError`。
`reducer` 必须是 `tg.sum()` 这样的实例，不能直接传工厂 `tg.sum`。
结果首维等于目标节点数。完整代码见 [CSR 例子](api-examples.zh.md#message-passing)。

在 `paged_csr` 图上，调用按有界目标行分页流式执行，并额外接受三个
调用时参数：`page_rows`（页高，缺省 100 000 或
`TIGA_PAGED_PAGE_ROWS`）、`prefetch=True`（让页读取与计算重叠）
与 `prefetch_depth`（并发预取的页数，缺省 2 或
`TIGA_PAGED_PREFETCH_DEPTH`）。示例：
[GCN 聚合](examples/message-passing.zh.md#gcn-aggregation)、
[磁盘上的超大图](examples/distributed-memory.zh.md#a-disk-resident-giant-graph)。

### tg.runtime.auto_offload(ram) { #gfruntimeautoffload }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `ram` | <code>int</code> | 图 CSR 自动卸载的非负字节阈值；IEC 字符串使用 tg.execution。 |

返回类型：`context manager`。

??? example "最小用法"

    ```python
    import tiga as tg
    with tg.runtime.auto_offload(1024 ** 3):
        graph = tg.Graph.regular(4, 2)
    ```

<!-- typed-contract:end -->

上下文管理器：生效期间，每个显式 CSR 构造器（`Graph.from_csr`、
`Graph.stencil`、`Graph.cat` 等）在 CSR 超过 `ram` 字节时把拓扑持久化
为 `.gfg` 并返回 `paged_csr` 图，kernel 调用随之自动分页并带 prefetch。
环境变量 `TIGA_GRAPH_RAM_BUDGET` 设定进程级默认预算；上下文管理器
优先。offload 产物落在进程级临时目录、退出时清理；设置
`TIGA_SPILL_DIR` 后可跨进程存活。仅 CPU 图；字段也可随拓扑落盘
（`tg.save(..., fields=...)`），前向与反向都能分页执行。
示例：
[RAM 预算下的自动 offload](examples/distributed-memory.zh.md#automatic-graph-offload)。

### program.reference(**kwargs) { #messagepassingreference }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `**kwargs` | <code>Mapping[str, object]</code> | 与普通调用相同的 graph、字段与 UDF 参数；使用 Torch。 |

返回类型：`torch.Tensor`。

??? example "最小用法"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    class Sum(tg.MessagePassing):
        reducer = tg.sum()
        def edge(self, src, dst, edge):
            return src.x
    program = Sum()
    x = torch.tensor([2., 3.], requires_grad=True)
    out = program.reference(graph=graph, src={'x': x}, dst={})
    ```

<!-- typed-contract:end -->

通过 Torch interop oracle 显式执行语义实现；这是正确性参照，
不是原生 backend。

### program.prepare(*, graph, src=None, dst=None, edge=None, ndata=None, **params) { #messagepassingprepare }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `graph` | <code>tg.Graph</code> | 本次调用的关系；设备、实体数须与字段匹配。 |
| `src` | <code>Mapping[str, torch.Tensor &#124; tg.Tensor] &#124; None</code> | 源字段，首维为 num_src；未使用该角色时传 {}。 |
| `dst` | <code>Mapping[str, torch.Tensor &#124; tg.Tensor] &#124; None</code> | 目标字段，首维为 num_dst；未使用该角色时传 {}。 |
| `edge` | <code>Mapping[str, torch.Tensor &#124; tg.Tensor] &#124; None</code> | 按图的边顺序绑定的字段；None 表示没有显式边字段。 |
| `ndata` | <code>Mapping[str, torch.Tensor &#124; tg.Tensor] &#124; None</code> | 同构图节点字段简写，不能与 src/dst 混用。 |
| `**params` | <code>object</code> | edge/node 签名声明的标量或可捕获参数；不代表支持任意 Python 对象。 |

返回类型：`Callable[[], Tensor]`。

运行示例前，先按下文配置所需设备、文件或分布式环境。

??? example "最小用法"

    ```python
    # CUDA scalar CSR only; graph and native fields must already be validated.
    prepared = program.prepare(graph=graph, src={'x': x}, dst={})
    out = prepared()
    ```

<!-- typed-contract:end -->

可选地将一个经过验证的标量 CSR CUDA 绑定冻结为零参数热提交。普通
执行仍是惰性 JIT，从不需要此调用。

### program.explain() { #messagepassingexplain }

<!-- typed-contract:start -->

无显式参数（实例方法的 `self` 省略）。

返回类型：`str`。

??? example "最小用法"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    class Sum(tg.MessagePassing):
        reducer = tg.sum()
        def edge(self, src, dst, edge):
            return src.x
    program = Sum()
    x = torch.tensor([2., 3.], requires_grad=True)
    out = program(graph=graph, src={'x': x}, dst={})
    print(program.explain())
    ```

<!-- typed-contract:end -->

以文本渲染最近一个变体的规划器/lowering/缓存信息。

### program.diagnostics { #messagepassingdiagnostics }

<!-- typed-contract:start -->

无显式参数（实例方法的 `self` 省略）。

返回类型：`tuple[AnalysisFinding, ...]`。

??? example "最小用法"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    class Sum(tg.MessagePassing):
        reducer = tg.sum()
        def edge(self, src, dst, edge):
            return src.x
    program = Sum()
    x = torch.tensor([2., 3.], requires_grad=True)
    out = program(graph=graph, src={'x': x}, dst={})
    print(program.diagnostics)
    ```

<!-- typed-contract:end -->

属性，返回类型化的 `AnalysisFinding` 记录——阶段、处置结论
（`accepted`/`rejected`/`warning`/`unknown`/`remark`）以及可选的
资源、目标契约、估计与建议动作。`explain()` 是这些记录的人类可读
形式。

### program.schedules { #messagepassingschedules }

<!-- typed-contract:start -->

无显式参数（实例方法的 `self` 省略）。

返回类型：`tuple[MachineSchedule, ...]`。

??? example "最小用法"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    class Sum(tg.MessagePassing):
        reducer = tg.sum()
        def edge(self, src, dst, edge):
            return src.x
    program = Sum()
    x = torch.tensor([2., 3.], requires_grad=True)
    out = program(graph=graph, src={'x': x}, dst={})
    print(program.schedules)
    ```

<!-- typed-contract:end -->

属性，返回由原生编译器从已验证的 `gf.kernel` IR 中提取的类型化
`MachineSchedule` 记录：调度族、行/邻居 tile、子组数量、流水线深度、
具名资源、执行角色、生产者/消费者交接以及获准的指令类别。
`pipeline_stages == 1` 是明确的否定结果——编译器未接纳由它控制的
异步流水线——因此不能把 provider 的指令重排说成已证明的重叠。

### program.ir(stage="domain") { #messagepassingir }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `stage` | <code>str</code> | IR 产物阶段；默认 domain。缺失产物抛出 KeyError。 |

返回类型：`str`。

??? example "最小用法"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    class Sum(tg.MessagePassing):
        reducer = tg.sum()
        def edge(self, src, dst, edge):
            return src.x
    program = Sum()
    x = torch.tensor([2., 3.], requires_grad=True)
    out = program(graph=graph, src={'x': x}, dst={})
    print(program.ir('domain'))
    ```

<!-- typed-contract:end -->

对 `domain` 返回语义计划，不一定是可解析的 MLIR；对 `iter`、`kernel`、
`task` 或 provider 阶段返回已有产物。`"gf.iter"` 等阶段别名是检查接口的
参数，不是操作名。产物不存在时抛出 `KeyError`。指南：
[检查边界](compiler-pipeline.md#inspecting-a-compiled-program)。

### program.code(kind="ptx") { #messagepassingcode }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `kind` | <code>str</code> | 生成代码类型，默认 ptx；要求已执行对应编译路径。 |

返回类型：`str`。

运行示例前，先按下文配置所需设备、文件或分布式环境。

??? example "最小用法"

    ```python
    # After a CUDA-compiled program call:
    print(program.code('ptx'))
    ```

<!-- typed-contract:end -->

返回最近变体的代码产物，如 `ttgir`、`llir` 或 `ptx`。

### program.cache_info { #messagepassingcache_info }

<!-- typed-contract:start -->

无显式参数（实例方法的 `self` 省略）。

返回类型：`dict[str, int]`。

??? example "最小用法"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    class Sum(tg.MessagePassing):
        reducer = tg.sum()
        def edge(self, src, dst, edge):
            return src.x
    program = Sum()
    x = torch.tensor([2., 3.], requires_grad=True)
    out = program(graph=graph, src={'x': x}, dst={})
    print(program.cache_info)
    ```

<!-- typed-contract:end -->

属性，返回 kernel 变体查询的命中/未命中/变体计数。原生 Tensor 路径统计
capture binding，物化前也会更新，不等于编译次数。物化后的
`Tensor.execution` 报告实际 backend，以及原生可执行文件的
`cache_hit`/`compile_ms`。

### program.last_variant 与 program.variants { #messagepassinglast_variant }

<!-- typed-contract:start -->

无显式参数（实例方法的 `self` 省略）。

返回类型：`CompiledVariant (variants: tuple[CompiledVariant, ...])`。

??? example "最小用法"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    class Sum(tg.MessagePassing):
        reducer = tg.sum()
        def edge(self, src, dst, edge):
            return src.x
    program = Sum()
    x = torch.tensor([2., 3.], requires_grad=True)
    out = program(graph=graph, src={'x': x}, dst={})
    print(program.last_variant)
    ```

<!-- typed-contract:end -->

最近一次调用（或每个已缓存特化）的 `CompiledVariant` 记录：backend、
provider、lowering、pass、产物、诊断与调度。首次调用前访问
`last_variant` 会抛出异常。

## Reducer { #reducers }

reducer 是可执行的编译器输入，不是 eager 张量运算。
[reducer 指南](reducers.zh.md)以文字形式解释代数契约、属性与
lowering 选择。

### tg.sum(*, identity=0, deterministic=False) { #gfsum }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `identity` | <code>int &#124; float</code> | 加法幺元；默认 0。 |
| `deterministic` | <code>bool</code> | 是否要求固定归约顺序；默认 False，受 backend 支持范围限制。 |

返回类型：`SumReducer`。

??? example "最小用法"

    ```python
    import tiga as tg
    reducer = tg.sum()
    ```

<!-- typed-contract:end -->

满足结合律与交换律的加法 reducer。示例：
[GCN 聚合](examples/message-passing.zh.md#gcn-aggregation)。

### tg.mean(*, deterministic=False) { #gfmean }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `deterministic` | <code>bool</code> | 是否要求固定归约顺序；默认 False，受 backend 支持范围限制。 |

返回类型：`MeanReducer`。

??? example "最小用法"

    ```python
    import tiga as tg
    reducer = tg.mean()
    ```

<!-- typed-contract:end -->

基于内置元组状态 `(sum, count)` 的邻居均值 reducer；度为 0 的行产生
NaN。指南：[tg.mean()](reducers.zh.md#gfmean)。

### tg.prod(*, deterministic=False) { #gfprod }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `deterministic` | <code>bool</code> | 是否要求固定归约顺序；默认 False，受 backend 支持范围限制。 |

返回类型：`ProductReducer`。

??? example "最小用法"

    ```python
    import tiga as tg
    reducer = tg.prod()
    ```

<!-- typed-contract:end -->

乘法 reducer，被提升为零安全乘积 IR；度为 0 的行产生幺元 1。示例：
[边 gate 的乘积](reducers.zh.md#product-of-edge-gates)。

### tg.online_softmax(*, accumulation_dtype=None, deterministic=False, block_prune_threshold=None) { #gfonline_softmax }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `accumulation_dtype` | <code>tg.DType &#124; torch.dtype &#124; None</code> | 累加精度提示；None 使用 provider 默认值。 |
| `deterministic` | <code>bool</code> | 是否要求固定归约顺序；默认 False，受 backend 支持范围限制。 |
| `block_prune_threshold` | <code>float &#124; None</code> | None 为精确计算；(0,1] 显式启用受支持 CUDA 前向的近似 tile 剪枝。 |

返回类型：`OnlineSoftmaxReducer`。

??? example "最小用法"

    ```python
    import tiga as tg
    reducer = tg.online_softmax()
    ```

<!-- typed-contract:end -->

稳定的多值流式 reducer；edge 区域返回 `reducer(score, value)`
（一个 `OnlineSoftmaxItem`）。位于 `(0, 1]` 的
`block_prune_threshold` 是面向 dense 流式 tile 的语义近似策略，不是
调度提示。示例：
[全量注意力，无掩码](examples/attention.zh.md#full-attention-no-mask)、
[基于 tile 剪枝的稀疏注意力](examples/attention.zh.md#tile-pruned-sparse-attention)。

### class tg.Reducer { #gfreducer }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `deterministic` | <code>bool</code> | 是否要求固定归约顺序；默认 False，受 backend 支持范围限制。 |

返回类型：`tg.Reducer subclass instance`。

??? example "最小用法"

    ```python
    import tiga as tg
    class Add(tg.Reducer):
        associative = True
        commutative = True
        def identity(self): return 0.
        def combine(self, left, right): return left + right
    reducer = Add()
    ```

<!-- typed-contract:end -->

用户自定义 reducer 基类：subclass 并实现 `identity()`、
`lift(*messages)`、`combine(left, right)` 与 `finalize(state)`；
状态可以是标量或元组。类属性 `associative`、`commutative` 与
`deterministic` 声明代数性质——并行 lowering 要求显式的结合律声明，
Tiga 不会从 Python 源码证明结合律。示例：
[用户自定义 reducer](examples/message-passing.zh.md#user-defined-reducer)；
指南：[自定义 reducer](reducers.zh.md#inventing-your-own)。

### reducer(*messages) { #reducer-call }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `*messages` | <code>Tensor expressions</code> | 交给 lift 的一个或多个边局部输入；这里只打包消息，不立即归约。 |

返回类型：`ReducerCall | OnlineSoftmaxItem`。

??? example "最小用法"

    ```python
    import tiga as tg
    class Attention(tg.MessagePassing):
        reducer = tg.online_softmax()
        def edge(self, src, dst, edge):
            return self.reducer(edge.score, src.value)
    ```

<!-- typed-contract:end -->

在 `edge()` 内调用 reducer，把它与一个或多个边局部消息绑定成一个
staged `ReducerCall`。

### reducer.mlir(*, message_dtypes=None, symbol=None) { #reducermlir }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `message_dtypes` | <code>Sequence[tg.DType] &#124; None</code> | 每个消息一个原生 dtype；None 表示单个 float32 消息。 |
| `symbol` | <code>str &#124; None</code> | IR 符号覆盖值；None 使用 reducer 名称。 |

返回类型：`str`。

??? example "最小用法"

    ```python
    import tiga as tg
    print(tg.sum().mlir(message_dtypes=(tg.float32,)))
    ```

<!-- typed-contract:end -->

把四个区域捕获为经过验证的原生 `gf.reducer` op 并返回 MLIR 文本；
`message_dtypes` 缺省为单个 `float32` 消息。指南：
[检查 reducer](reducers.zh.md#inspecting-a-reducer)。

??? info "自动 reducer VJP 覆盖范围"

    加法/稳定代数以及捕获的、满足结合律的标量 reducer，会通过 CPU LLVM
    与 CUDA TTIR lowering 为平衡树或确定性树。乘积被提升为零安全 CSR
    乘积/VJP IR。空行/参差行和有序非交换元组状态已被覆盖；未注册的
    reducer 形态仍只有正确性保证。

## nn { #nn }

### tg.nn.trace(module, *, block_e=None, num_warps=None) { #gfnntrace }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `module` | <code>torch.nn.Module</code> | 可 trace 的受支持模块；原始参数保留 Torch autograd 连接。 |
| `block_e` | <code>int &#124; None</code> | 可选 tile 大小，[16,1024] 内的 2 的幂。 |
| `num_warps` | <code>int &#124; None</code> | 可选 provider 启动 warp 数；None 使用 lowering 默认值。 |

返回类型：`TracedModule`。

??? example "最小用法"

    ```python
    import torch
    import tiga as tg
    module = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh())
    traced = tg.nn.trace(module)
    y = traced(torch.ones(2, 3))
    y.sum().backward()
    ```

<!-- typed-contract:end -->

包装一个 `torch.nn` 模块，使其可在 edge UDF 内调用：对 Torch tensor
走 eager（输入沿特征维拼接），或被捕获为可编译子图，进入融合的
edge-NN tile lowering。`block_e`（edge-tile 大小，[16, 1024] 内的
2 的幂）与 `num_warps` 只是 launch geometry，从不改变数值；缺省值
按 lowering 区分——edge-centric sum tile 为 128/4，row-centric 注意力
tile 为 16/1。示例：
[用 nn 打分的 GAT 边注意力](examples/attention.zh.md#gat-edge-attention-with-an-nn-score)、
[Edge nn 模块与融合 tile kernel](examples/programs-and-interop.zh.md#edge-nn-modules-with-a-fused-tile-kernel)；
指南：[Edge nn 模块（CUDA，torch interop）](message-passing.zh.md#edge-nn-modules-cuda-torch-interop)。

## Control 与 jit { #control }

Tiga 语义值之上的结构化编译器控制流：循环体只捕获一次，lowering
为 `gf_control.repeat` / `gf_control.while` op，而不是作为 Python
循环执行。

### @tg.jit 或 @tg.jit(max_iterations=k) { #gfjit }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `function` | <code>Callable &#124; None</code> | 被装饰函数；AST 捕获要求能读取源码。 |
| `max_iterations` | <code>int &#124; None</code> | while 的有限上界；无 while 时允许 None。 |

返回类型：`Callable`。

运行示例前，先按下文配置所需设备、文件或分布式环境。

??? example "最小用法"

    ```python
    import tiga as tg
    @tg.jit
    def decay(x):
        for i in range(3):
            x = x * 0.5
        return x
    print(decay(tg.tensor([8.])).tolist())
    ```

<!-- typed-contract:end -->

AST 路由：把 `for i in range(k)` 捕获为 `gf_control.repeat`，把裸
`while cond:` 捕获为由装饰器参数限定上界的 `gf_control.while`。
`for` 循环体开头的 `if cond: break` 是提前退出。循环携带变量是在
循环前赋值的 Tensor；`continue`、循环体中部的 `break`、`while True`、
非 range 迭代以及循环 `else` 均 fail closed。MessagePassing UDF 区域
永远不会被 AST 变换。`@tg.jit` 同时自动激活组合上下文：直线部分里
平级的 MessagePassing 调用被捕获为 `GraphProgram` SSA 叶子并在合法处
融合，叶子可直接参与张量算术；staged 循环体内的 kernel 调用保持逐
迭代语义（内联进循环体，绝不注册成顶层叶子）。捕获字段带
`requires_grad` 时叶子可微：反向逐叶子内联展开求 VJP（不融合），
共享输入的梯度贡献累加；不支持的叶子形态 fail closed。示例：
[固定迭代的 PageRank 探针](examples/message-passing.zh.md#fixed-iteration-pagerank-probe)、
[无矩阵 FEM 算子与求解循环](examples/solvers.zh.md#matrix-free-fem-operator-and-solver-loop)。

### tg.repeat(initial, body, *, iterations) { #gfrepeat }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `initial` | <code>tg.Tensor &#124; Sequence[tg.Tensor]</code> | 循环携带的原生值；shape、dtype 和设备须保持不变。 |
| `body` | <code>Callable[..., tg.Tensor &#124; Sequence[tg.Tensor]]</code> | 捕获一次；每个携带状态返回一个值。 |
| `iterations` | <code>int</code> | 非负固定迭代次数；不接受 bool。 |

返回类型：`tg.Tensor | tuple[tg.Tensor, ...]`。

??? example "最小用法"

    ```python
    import tiga as tg
    y = tg.repeat(tg.tensor([8.]), lambda x: x * 0.5, iterations=3)
    assert y.tolist() == [1.]
    ```

<!-- typed-contract:end -->

匿名简写，捕获一个携带一个或多个 Tensor 状态的定次循环；`body` 只
trace 一次，且必须为每个携带状态返回一个形状、dtype、设备均不变的
Tensor。CPU lowering 为每个携带值分配两个可复用缓冲区。

### tg.while_loop(initial, condition, body, *, max_iterations) { #gfwhile_loop }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `initial` | <code>tg.Tensor &#124; Sequence[tg.Tensor]</code> | 循环携带的原生值；shape、dtype 和设备须保持不变。 |
| `condition` | <code>Callable[..., tg.Tensor]</code> | 捕获的条件函数，返回标量布尔 Tensor。 |
| `body` | <code>Callable[..., tg.Tensor &#124; Sequence[tg.Tensor]]</code> | 捕获一次；每个携带状态返回一个值。 |
| `max_iterations` | <code>int</code> | 非负有限循环上界，允许零。 |

返回类型：`tg.Tensor | tuple[tg.Tensor, ...]`。

??? example "最小用法"

    ```python
    import tiga as tg
    y = tg.while_loop(tg.tensor(1.), lambda x: x < 4., lambda x: x + 1., max_iterations=8)
    assert y.tolist() == 4.
    ```

<!-- typed-contract:end -->

匿名简写，捕获无主机标量检查的有界数据依赖控制流；`condition` 必须
返回 rank-0 布尔 Tensor，`max_iterations` 作为有限资源守卫保留在
IR 中。CPU lowering 为 `scf.while`；在 provider 循环计划就绪之前 CUDA
fail closed。

### class tg.control.Repeat { #gfcontrolrepeat }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `initial` | <code>tg.Tensor &#124; Sequence[tg.Tensor]</code> | 循环携带的原生值；shape、dtype 和设备须保持不变。 |
| `iterations` | <code>int</code> | 非负固定迭代次数；不接受 bool。 |

返回类型：`tg.Tensor | tuple[tg.Tensor, ...]`。

??? example "最小用法"

    ```python
    import tiga as tg
    class Decay(tg.control.Repeat):
        def body(self, x): return x * 0.5
    assert Decay()(tg.tensor([8.]), iterations=3).tolist() == [1.]
    ```

<!-- typed-contract:end -->

同一个 `gf_control.repeat` op 的构建器形式：subclass 并重写 `body`；
捕获的常量是普通实例属性。以 `kernel(initial, iterations=k)` 调用
实例时，`body` 恰好 trace 一次。

### class tg.control.While { #gfcontrolwhile }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `initial` | <code>tg.Tensor &#124; Sequence[tg.Tensor]</code> | 循环携带的原生值；shape、dtype 和设备须保持不变。 |
| `max_iterations` | <code>int</code> | 非负有限循环上界，允许零。 |

返回类型：`tg.Tensor | tuple[tg.Tensor, ...]`。

??? example "最小用法"

    ```python
    import tiga as tg
    class Grow(tg.control.While):
        def condition(self, x): return x < 4.
        def body(self, x): return x + 1.
    assert Grow()(tg.tensor(1.), max_iterations=8).tolist() == 4.
    ```

<!-- typed-contract:end -->

`gf_control.while` 的构建器形式：subclass 并重写 `condition` 与
`body`；以 `kernel(initial, max_iterations=k)` 调用实例时，适用与
`tg.while_loop` 相同的上界。

定常求解器是这些原语之上的语法糖，不属于核心 API：
[examples/solvers.py](https://github.com/walkerchi/TIGA-lang/blob/main/examples/solvers.py)
提供 `dot`、`vector_norm`、`richardson` 和 `cg`；参见
[线性求解器与控制流](examples/solvers.zh.md)。

## 分布式与 halo { #distributed-and-halo }

### tg.DeviceMesh(device_type, shape, *, names=()) { #gfdevicemesh }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `device_type` | <code>str</code> | 逻辑设备族，如 cpu 或 cuda；不会初始化设备。 |
| `shape` | <code>int &#124; tuple[int, ...]</code> | 正的网格尺寸；乘积为 rank 数。 |
| `names` | <code>tuple[str, ...]</code> | 可选网格轴名，每个维度一个。 |

返回类型：`tg.DeviceMesh`。

??? example "最小用法"

    ```python
    import tiga as tg
    mesh = tg.DeviceMesh('cpu', (2,), names=('workers',))
    ```

<!-- typed-contract:end -->

描述逻辑设备（`shape` 为 int 或元组），不初始化进程组。部署时
绑定到 Torch `DeviceMesh`、MPI、NCCL/RCCL 或厂商通信器。示例：
[双进程 halo 交换](examples/distributed-memory.zh.md#two-process-halo-exchange)。

### tg.ByDestination(mesh_axis=0, balance="auto") { #gfbydestination }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `mesh_axis` | <code>int &#124; str</code> | 网格轴编号或名称。 |
| `balance` | <code>Literal[&#x27;auto&#x27;, &#x27;edges&#x27;, &#x27;entities&#x27;]</code> | 分区负载平衡目标。 |

返回类型：`tg.ByDestination`。

??? example "最小用法"

    ```python
    import tiga as tg
    partition = tg.ByDestination(balance='edges')
    ```

<!-- typed-contract:end -->

把目标实体与最终 reducer 状态分配到 mesh 分片的分区策略；
`balance` 取 `"edges"`、`"entities"` 或 `"auto"`。

### tg.GraphPlacement(mesh, partition, halo_depth) { #gfgraphplacement }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `mesh` | <code>tg.DeviceMesh</code> | 逻辑设备拓扑，不是运行中的 communicator。 |
| `partition` | <code>tg.ByDestination</code> | 目标实体的所有权策略。 |
| `halo_depth` | <code>int &#124; Literal[&#x27;auto&#x27;]</code> | 非负深度或由编译器推断。 |

返回类型：`tg.GraphPlacement`。

??? example "最小用法"

    ```python
    import tiga as tg
    placement = tg.GraphPlacement(tg.DeviceMesh('cpu', 2), tg.ByDestination(), 'auto')
    ```

<!-- typed-contract:end -->

由 `graph.halo(...)` 附加到 Graph 快照上的逻辑所有权与 ghost 需求；
`halo_depth` 为非负整数或 `"auto"`。

### tg.HaloMap { #gfhalomap }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `rank` | <code>int</code> | 本地 rank，范围 [0, world_size)。 |
| `world_size` | <code>int</code> | 正的 rank 数。 |
| `owned_begin / owned_end` | <code>int</code> | 拥有实体的全局编号左闭右开区间。 |
| `ghost_ids` | <code>tuple[int, ...]</code> | 拥有行需要的外部源编号，按序排列。 |
| `receive_from / send_to` | <code>tuple[tuple[int, tuple[int, ...]], ...]</code> | 对端 rank 与有序全局实体编号。 |
| `bytes_for: itemsize / trailing_elements` | <code>int</code> | 每标量字节数、每实体标量数（默认 1），均为正数。 |

返回类型：`tg.HaloMap; bytes_for → int`。

??? example "最小用法"

    ```python
    import tiga as tg
    halos = tg.collective_halo_maps([0, 1, 2], [1, 0], num_entities=2, world_size=2)
    print(halos[0].ghost_ids, halos[0].bytes_for(4))
    ```

<!-- typed-contract:end -->

具体的单 rank 拥有者/ghost 映射：`owned_begin`/`owned_end`、
`ghost_ids`、`receive_from` 与 `send_to`，以及 `owned_entities`、
`ghost_entities`、`bytes_for(itemsize, trailing_elements=1)` 辅助
方法。

### tg.derive_halo_map(row_ptr, col_idx, *, num_entities, world_size, rank, peer_requests=None) { #gfderive_halo_map }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `row_ptr / col_idx` | <code>Sequence[int]</code> | 全局 CSR 行边界与源编号，使用 host 整数序列。 |
| `num_entities` | <code>int</code> | 同构 CSR 关系的全局实体数。 |
| `world_size` | <code>int</code> | 正的 rank 数。 |
| `rank` | <code>int</code> | 本地 rank，范围 [0, world_size)。 |
| `peer_requests` | <code>Sequence[Sequence[int]] &#124; None</code> | 每个 rank 请求的编号；None 使 send_to 为空。 |

返回类型：`tg.HaloMap`。

??? example "最小用法"

    ```python
    import tiga as tg
    halo = tg.derive_halo_map([0, 1, 2], [1, 0], num_entities=2, world_size=2, rank=0)
    ```

<!-- typed-contract:end -->

从本 rank 拥有的 CSR 行推导精确的 ghost，不依赖任何框架；
`peer_requests[p]` 列出 peer `p` 请求的本地实体 ID，用于填充
`send_to`。指南：
[精确说明所有权与 ghost](memory-and-distributed.zh.md#ownership-and-ghosts-exactly)。

### tg.collective_halo_maps(row_ptr, col_idx, *, num_entities, world_size) { #gfcollective_halo_maps }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `row_ptr / col_idx` | <code>Sequence[int]</code> | 全局 CSR 行边界与源编号，使用 host 整数序列。 |
| `num_entities` | <code>int</code> | 同构 CSR 关系的全局实体数。 |
| `world_size` | <code>int</code> | 正的 rank 数。 |

返回类型：`tuple[tg.HaloMap, ...]`。

??? example "最小用法"

    ```python
    import tiga as tg
    halos = tg.collective_halo_maps([0, 1, 2], [1, 0], num_entities=2, world_size=2)
    ```

<!-- typed-contract:end -->

为所有 rank 创建互相一致的接收/发送映射。

### tg.exchange_halo(halo, owned_data, *, element_bytes, transport) { #gfexchange_halo }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `halo` | <code>tg.HaloMap</code> | 本 rank 的所有权与通信映射。 |
| `owned_data` | <code>bytes &#124; bytearray &#124; memoryview</code> | 从 owned_begin 开始、按全局编号排列的拥有实体字节。 |
| `element_bytes` | <code>int</code> | 每实体总字节数，包含特征维，必须为正。 |
| `transport` | <code>NeighborTransport</code> | 实现 rank、world_size 与 exchange 的 transport；须与 halo 拓扑匹配。 |

返回类型：`HaloBuffer`。

运行示例前，先按下文配置所需设备、文件或分布式环境。

??? example "最小用法"

    ```python
    # Inside an initialized rank; peers must execute the matching exchange.
    received = tg.exchange_halo(halo, owned_data, element_bytes=4, transport=transport)
    ```

<!-- typed-contract:end -->

在一条 transport 上对编译器推导的邻居 [halo](https://en.wikipedia.org/wiki/Halo_(computer_science)) 完成 pack、
exchange 与 unpack，返回 `HaloBuffer`。`owned_data` 是从
`halo.owned_begin` 起按目标拥有者排列的本地字节存储；不涉及任何
框架张量类型。

### DistributedRuntime(transport, *, progress_threads=1) { #distributedruntime }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `transport` | <code>NeighborTransport</code> | 实现 rank、world_size 与 exchange 的 transport；须与 halo 拓扑匹配。 |
| `progress_threads` | <code>int</code> | 后台 progress worker 正整数数量；默认 1。 |

返回类型：`tg.DistributedRuntime`。

运行示例前，先按下文配置所需设备、文件或分布式环境。

??? example "最小用法"

    ```python
    # transport is a deployment-bound NeighborTransport.
    import tiga as tg
    with tg.DistributedRuntime(transport) as runtime:
        output = program(graph=placed_graph, src={'x': local_x}, dst={})
    ```

<!-- typed-contract:end -->

在图 kernel 之下管理异步通信推进。作为上下文管理器使用时成为活跃
runtime；没有活跃 runtime 时调用分布式图会 fail closed。
`runtime.exchange_halo(halo, owned_data, *, element_bytes)` 启动 halo
推进并返回一个与 provider 无关的 completion；
`runtime.last_execution_trace` 报告最近一次自动分片执行的耗时。

分布式执行只使用一种策略：先完成 halo 交换，再计算全部 owned 行，无需选择调度模式。TCP 使用 host staging，NCCL 使用 device buffer。trace 的调度名为 `host-staged-serialized` 或 `device-direct-serialized`，`measured_overlap_ms=0`。TCP 的 `timeout` 要求有限正数，同时约束握手与数据等待；冷 JIT 较慢时应增大。参见[两台机器的完整验证命令](memory-and-distributed.zh.md#two-host-cuda-sanity)。

### DistributedRuntime.from_provider(name, *, progress_threads=1, **options) { #distributedruntimefrom_provider }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `name` | <code>str</code> | 已注册的 tcp、mpi、nccl 或已安装插件。 |
| `progress_threads` | <code>int</code> | 后台 progress worker 正整数数量；默认 1。 |
| `**options` | <code>provider-specific keyword arguments</code> | TCP：rank、world_size、host、port、bind、timeout。MPI：communicator、tag。NCCL 需要 communicator/部署配置，见分布式指南。 |
| `tcp/nccl: rank / world_size` | <code>int</code> | 必填 rank 与正的组规模；0 <= rank < world_size。 |
| `tcp: port` | <code>int</code> | 必填 rendezvous 端口；所有对端须一致且 peer 监听端口可达。 |
| `tcp: host / bind` | <code>str &#124; None / str</code> | host 指定 rendezvous 对端；None 在本地监听。bind 默认为空监听地址。 |
| `tcp: timeout` | <code>float</code> | 连接与数据等待的有限正秒数；默认 30。 |
| `mpi: communicator` | <code>mpi4py.MPI.Comm &#124; None</code> | None 使用 MPI.COMM_WORLD；需要 mpi4py 与 MPI runtime。 |
| `mpi: tag` | <code>int</code> | 指定 communicator 上的消息 tag；默认 0。 |
| `nccl: communicator_id` | <code>bytes &#124; None</code> | launcher 分发的共享 128 字节 NCCL unique ID；None 仅适用于单 rank。 |
| `nccl: device` | <code>str &#124; tg.Device</code> | 本地 CUDA 设备，默认 cuda:0；不是全局 rank 编号。 |
| `nccl: library` | <code>str &#124; os.PathLike &#124; None</code> | 显式 libnccl 路径，或自动查找。 |

返回类型：`tg.DistributedRuntime`。

运行示例前，先按下文配置所需设备、文件或分布式环境。

??? example "最小用法"

    ```python
    # Run collectively with an MPI launcher and the mpi extra installed.
    import tiga as tg
    with tg.DistributedRuntime.from_provider('mpi') as runtime:
        print(runtime.transport.rank)
    ```

<!-- typed-contract:end -->

在不变的图 API 之下绑定部署用 transport 插件：默认注册 `"mpi"`
（mpi4py 包装）、`"tcp"`（零依赖的跨机器套接字）与 `"nccl"`
（基于 `libnccl` 的分组 send/recv）；第三方通过
`tiga.transport` entry-point 组注册。示例：
[双进程 halo 交换](examples/distributed-memory.zh.md#two-process-halo-exchange)、
[跨进程与跨机器运行](examples/distributed-memory.zh.md#running-across-processes-and-machines)；
指南：[runtime 与 transport](memory-and-distributed.zh.md#runtimes-and-transports)。

??? info "当前分布式执行边界"

    CPU rank 本地执行、反向 halo VJP、分页 CSR 分片以及真实的双进程
    [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) 传输均已可执行。[CUDA](https://en.wikipedia.org/wiki/CUDA) 原生缓冲区使用独立的
    通信/编译器流和事件排序。双机 CUDA TCP/NCCL 支持前向与反向 halo VJP，
    执行顺序为先通信、再计算；暂不支持按 GPU 处理速度自动重分区及 RCCL 执行。参见
    [内存层次与分布式执行](memory-and-distributed.zh.md)。

## 执行与存储 { #spill-and-disk }

完整且有测试的例子见[内存与存储](memory.zh.md)和 [API 例子](api-examples.zh.md#storage)。

### tg.execution(*, device="cpu", memory=None, spill_dir=None, page_rows=100000, prefetch_depth=2, eviction="error") { #tgexecution }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `device` | <code>str &#124; tg.Device &#124; None</code> | 执行设备；None 按本条构造器的默认规则处理。 |
| `memory` | <code>Mapping[str, int &#124; str] &#124; None</code> | 各层字节数或 IEC 大小预算；省略表示不设限，不预留物理内存。 |
| `spill_dir` | <code>str &#124; os.PathLike &#124; None</code> | 临时 spill 目录；显式 LRU 策略需要提供。 |
| `page_rows` | <code>int</code> | CPU CSR 每页目标行数，正整数；默认 100000。 |
| `prefetch_depth` | <code>int</code> | 预取页数，正整数；默认 2。 |
| `eviction` | <code>Literal[&#x27;error&#x27;, &#x27;lru&#x27;]</code> | 超预算时报错，或淘汰空闲的完整原生 Tensor。 |

返回类型：`ExecutionScope (context manager)`。

??? example "最小用法"

    ```python
    import tempfile
    import tiga as tg
    with tempfile.TemporaryDirectory() as directory:
        with tg.execution(memory={'ram': '1 MiB'}, spill_dir=directory) as run:
            x = tg.tensor([1., 2.])
            print(run.memory_report())
    ```

<!-- typed-contract:end -->

返回 context manager，设置 `tg.tensor` / `tg.empty` 的默认设备、分配限额和 CPU 分页默认值。不迁移已有值；显式设备参数或分页调用参数优先。

| 参数 | 取值与默认行为 | 错误 |
|---|---|---|
| `device` | Runtime 设备，如 `"cpu"`、`"cuda:0"`；默认 CPU | 不支持的设备在使用时失败 |
| `memory` | `ram/host`、`device/hbm`、`host-pinned`、`nvme/ssd` 到非负整数字节或整数 IEC 容量的映射；省略的层不限额 | 无效容量、重复别名、不可分配层抛出 `ValueError` |
| `spill_dir` | 临时 spill 和图自动 offload 的路径；默认受管临时目录 | 文件系统错误直接传播 |
| `page_rows` | 正整数，默认 100000 | 零、负数、bool、非整数抛出 `ValueError` |
| `prefetch_depth` | 正整数，默认 2 | 同上 |
| `eviction` | `"error"` 默认超额报错；`"lru"` 自动换出闲置原生 Tensor | 未知策略或 LRU 缺少 `spill_dir` 抛出 `ValueError` |

`with tg.execution(...) as run:` 不支持嵌套，嵌套抛出 `RuntimeError`。退出只恢复默认值，不销毁返回的值。`run.memory_report()` 返回 `scope`、`excludes`、`live_bytes`、`peak_bytes`、`budget_bytes`、`automatic_tensor_eviction`、`eviction_policy`、`evictions` 和 `restores`；层名使用规范名称，容量均为字节。只统计作用域内新建的分配，包含其分页预取线程；不是 RSS 统计。LRU 只同步换出闲置的完整 Tensor，无法切分超预算的 kernel 工作集，且不支持固定地址的 `prepare()`。参见[可运行例子和边界](memory.zh.md#automatic-eviction)。

### Tensor.spill() { #tensorspill }

<!-- typed-contract:start -->

无显式参数（实例方法的 `self` 省略）。

返回类型：`tg.Tensor`。

??? example "最小用法"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    x.spill()
    assert not x.residency['resident']
    assert x.tolist() == [[1., 2.], [3., 4.]]
    ```

<!-- typed-contract:end -->

写出临时 payload 并放弃自身的驻留 buffer 引用，返回自身。保留 shape、dtype、逻辑设备、version 和 autograd 历史；读取恢复到原设备。已有 view/launch 可能仍持有旧分配。临时文件在恢复或回收时删除；NVMe 临时预算或恢复目标层超额抛出 `MemoryError`；未恢复就再次 spill 抛出 `RuntimeError`。

### Tensor.to(device) { #tensorto }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `device` | <code>str &#124; tg.Device</code> | 目标设备；同设备调用会 realize 并返回自身。 |

返回类型：`tg.Tensor`。

??? example "最小用法"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    host = x.to('cpu')
    ```

<!-- typed-contract:end -->

跨设备返回已 realize 的可微拷贝，不修改原对象；同设备 realize 并返回自身。目标分配参与预算检查，设备/驱动错误直接传播。当前拷贝是 eager 操作，不是调度标注。

### Tensor.cpu() { #tensorcpu }

<!-- typed-contract:start -->

无显式参数（实例方法的 `self` 省略）。

返回类型：`tg.Tensor`。

??? example "最小用法"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    host = x.cpu()
    ```

<!-- typed-contract:end -->

等价于 `x.to("cpu")`。CUDA 输入仍是 CUDA，返回值是 CPU；CPU 输入原地 realize。

### Tensor.residency { #tensorresidency }

<!-- typed-contract:start -->

无显式参数（实例方法的 `self` 省略）。

返回类型：`dict[str, object]`。

??? example "最小用法"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    print(x.residency)
    ```

<!-- typed-contract:end -->

只读状态字典：`device` 是逻辑设备字符串，`resident` 表示是否持有 buffer，`backing` 为 `"snapshot"`、`"temporary-spill"` 或 `None`，`bytes/version` 描述数值。这里的字节数不是可唯一释放的字节数。

### Tensor.save(path, *, overwrite=False) / tg.save(tensor, path, *, overwrite=False) { #tensorsave }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `path` | <code>str &#124; os.PathLike</code> | 文件系统路径，不是 URL；父目录和覆盖规则见下文。 |
| `overwrite` | <code>bool</code> | 是否允许替换已有 Tensor 快照；默认 False。 |

返回类型：`pathlib.Path (method); None (tg.save)`。

??? example "最小用法"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'value.tga'
        x.save(path)
        restored = tg.load(path)
        assert restored.tolist() == x.tolist()
    ```

<!-- typed-contract:end -->

按连续逻辑顺序保存数值快照，也支持非连续 view，不换出原值。Tensor 方法返回路径，模块级入口返回 `None`。已有路径默认抛出 `FileExistsError`；`overwrite=True` 使用原子替换，失败会清理自身的临时输出。持久化文件不计入临时 NVMe 预算。不序列化 autograd 历史，但不破坏当前进程中输入的历史。

### tg.load(path, *, device="cpu") { #tgload }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `path` | <code>str &#124; os.PathLike</code> | 文件系统路径，不是 URL；父目录和覆盖规则见下文。 |
| `device` | <code>str &#124; tg.Device &#124; None</code> | 执行设备；None 按本条构造器的默认规则处理。 |

返回类型：`tg.Tensor | tg.Graph`。

??? example "最小用法"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'value.tga'
        x.save(path)
        restored = tg.load(path)
        assert restored.tolist() == x.tolist()
    ```

<!-- typed-contract:end -->

文件按 Tensor 快照打开，目录按版本化 Graph 打开。Tensor attach 只校验有界元数据和 payload 长度，不分配完整 payload；首次观察时在 `device` 上分配。返回 `requires_grad=False` 的叶子 Tensor，保留 shape、dtype 和 version。路径不存在时传播文件系统错误，快照损坏抛出 `ValueError`。Tensor 快照仍使用旧的 native-endian 格式，不保证跨字节序可移植。

### Tensor.disk(*, name=None) { #tensordisk }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `name` | <code>str &#124; None</code> | None 临时 spill；合法 basename 保存至旧式命名存储。 |

返回类型：`tg.Tensor`。

??? example "最小用法"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    x.disk()
    print(x.tolist())
    ```

<!-- typed-contract:end -->

兼容入口。无名称时临时 spill；有名称时持久保存在 `TIGA_SPILL_DIR` 或 `~/.cache/tiga/spill`。名称不能以点开头或包含路径分隔符，重名抛出 `FileExistsError`。新代码使用显式 `save/load` 路径。Spill 保留 autograd，重新 attach 的快照不会恢复计算图。

### tg.from_disk(name) { #gffrom_disk }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `name` | <code>str</code> | 已有快照 basename；不能以点开头或含路径分隔符。 |

返回类型：`tg.Tensor`。

运行示例前，先按下文配置所需设备、文件或分布式环境。

??? example "最小用法"

    ```python
    import tiga as tg
    # Requires an existing named snapshot; prefer explicit tg.save/tg.load paths.
    x = tg.from_disk('existing-snapshot')
    ```

<!-- typed-contract:end -->

在 CPU 上惰性 attach 旧的命名快照；名称无效抛出 `ValueError`，不存在抛出 `FileNotFoundError`。加载后文件仍保留；新代码优先使用 `tg.load(path)`。

## GraphProgram { #program-and-visualize }

### @tg.program { #gfprogram }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `function` | <code>Callable</code> | 包含 MessagePassing 调用的直线 Python 函数。 |

返回类型：`Callable`。

运行示例前，先按下文配置所需设备、文件或分布式环境。

??? example "最小用法"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    class Sum(tg.MessagePassing):
        reducer = tg.sum()
        def edge(self, src, dst, edge):
            return src.x
    program = Sum()
    x = torch.tensor([2., 3.], requires_grad=True)
    @tg.program
    def composed(x):
        return program(graph=graph, src={'x': x}, dst={})
    value = composed(x)
    print(value.materialize())
    ```

<!-- typed-contract:end -->

为无法提供源码的直线代码保留的兼容捕获边界：函数内的 MessagePassing
调用成为同一个 `GraphProgram`（类型化无环 SSA 组合）的 apply。
`@tg.jit` 会自动激活同一上下文（并额外捕获循环），新代码应直接使用
`@tg.jit`；两者叠加合法且幂等。装饰器之外的普通叶子调用仍是自动
JIT。示例：
[跨 kernel SSA 捕获](examples/programs-and-interop.zh.md#gfprogram-ssa-capture)。

### GraphProgram.apply(kernel, *, graph, src, dst, edge=None, **params) { #graphprogramapply }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `kernel` | <code>tg.MessagePassing</code> | 待捕获的叶程序。 |
| `graph` | <code>tg.Graph</code> | 本次调用的关系；设备、实体数须与字段匹配。 |
| `src` | <code>Mapping[str, torch.Tensor &#124; tg.Tensor] &#124; None</code> | 源字段，首维为 num_src；未使用该角色时传 {}。 |
| `dst` | <code>Mapping[str, torch.Tensor &#124; tg.Tensor] &#124; None</code> | 目标字段，首维为 num_dst；未使用该角色时传 {}。 |
| `edge` | <code>Mapping[str, torch.Tensor &#124; tg.Tensor] &#124; None</code> | 按图的边顺序绑定的字段；None 表示没有显式边字段。 |
| `**params` | <code>object</code> | edge/node 签名声明的标量或可捕获参数；不代表支持任意 Python 对象。 |

返回类型：`ProgramValue`。

运行示例前，先按下文配置所需设备、文件或分布式环境。

??? example "最小用法"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    class Sum(tg.MessagePassing):
        reducer = tg.sum()
        def edge(self, src, dst, edge):
            return src.x
    program = Sum()
    x = torch.tensor([2., 3.], requires_grad=True)
    composition = tg.GraphProgram()
    value = composition.apply(program, graph=graph, src={'x': x}, dst={})
    ```

<!-- typed-contract:end -->

向组合中加入一个 MessagePassing 叶子，返回其类型化 `ProgramValue`。

### GraphProgram.outputs(*values) { #graphprogramoutputs }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `*values` | <code>ProgramValue &#124; tg.Tensor</code> | 属于此组合的输出；不接受其他程序的值。 |

返回类型：`GraphProgram (self)`。

运行示例前，先按下文配置所需设备、文件或分布式环境。

??? example "最小用法"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    class Sum(tg.MessagePassing):
        reducer = tg.sum()
        def edge(self, src, dst, edge):
            return src.x
    program = Sum()
    x = torch.tensor([2., 3.], requires_grad=True)
    composition = tg.GraphProgram()
    value = composition.apply(program, graph=graph, src={'x': x}, dst={})
    composition.outputs(value)
    ```

<!-- typed-contract:end -->

选定在组合边界上被观察的值。裸 `ProgramValue` 叶子直接选定其
producer；Tensor 表达式则贡献其传递引用的全部 program 叶子——
因此在叶子之上叠加算术是合法的输出写法，而不会谎称并入了叶子
kernel 的融合。

### GraphProgram.run() { #graphprogramrun }

<!-- typed-contract:start -->

无显式参数（实例方法的 `self` 省略）。

返回类型：`Tensor | tuple[Tensor, ...]`。

运行示例前，先按下文配置所需设备、文件或分布式环境。

??? example "最小用法"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    class Sum(tg.MessagePassing):
        reducer = tg.sum()
        def edge(self, src, dst, edge):
            return src.x
    program = Sum()
    x = torch.tensor([2., 3.], requires_grad=True)
    composition = tg.GraphProgram()
    value = composition.apply(program, graph=graph, src={'x': x}, dst={})
    result = composition.run()
    ```

<!-- typed-contract:end -->

对选定输出做 JIT 与执行；相互独立的叶子在合法处水平融合，有依赖的
apply 通过类型化的运行时依赖 DAG 执行。

### GraphProgram.ir(stage="domain") { #graphprogramir }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `stage` | <code>str</code> | domain、fused、iteration/iter 或 kernel。 |

返回类型：`str`。

运行示例前，先按下文配置所需设备、文件或分布式环境。

??? example "最小用法"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    class Sum(tg.MessagePassing):
        reducer = tg.sum()
        def edge(self, src, dst, edge):
            return src.x
    program = Sum()
    x = torch.tensor([2., 3.], requires_grad=True)
    composition = tg.GraphProgram()
    value = composition.apply(program, graph=graph, src={'x': x}, dst={})
    print(composition.ir('domain'))
    ```

<!-- typed-contract:end -->

返回 `domain`、`fused`、`iteration`/`iter` 或 `kernel` IR 文本。调用
`ir` 即触发组合、验证与 lowering 的观察边界。

### GraphProgram.code(kind="ptx") { #graphprogramcode }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `kind` | <code>str</code> | 如 ptx 或 ttir 的 provider 产物；不存在时抛出 KeyError。 |

返回类型：`str | bytes | dict`。

运行示例前，先按下文配置所需设备、文件或分布式环境。

??? example "最小用法"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    class Sum(tg.MessagePassing):
        reducer = tg.sum()
        def edge(self, src, dst, edge):
            return src.x
    program = Sum()
    x = torch.tensor([2., 3.], requires_grad=True)
    composition = tg.GraphProgram()
    value = composition.apply(program, graph=graph, src={'x': x}, dst={})
    # Requires a matching provider; CPU paths do not produce PTX.
    print(composition.code('ptx'))
    ```

<!-- typed-contract:end -->

返回 provider 产物，如 `ttir` 或 `ptx`；对有依赖的 apply 返回按
apply 组织的产物映射。

### GraphProgram.explain() 与 GraphProgram.semantic_hash { #graphprogramexplain }

<!-- typed-contract:start -->

无显式参数（实例方法的 `self` 省略）。

返回类型：`str (semantic_hash: str)`。

运行示例前，先按下文配置所需设备、文件或分布式环境。

??? example "最小用法"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    class Sum(tg.MessagePassing):
        reducer = tg.sum()
        def edge(self, src, dst, edge):
            return src.x
    program = Sum()
    x = torch.tensor([2., 3.], requires_grad=True)
    composition = tg.GraphProgram()
    value = composition.apply(program, graph=graph, src={'x': x}, dst={})
    result = composition.explain()
    ```

<!-- typed-contract:end -->

渲染 apply/融合计数以及组合后 domain IR 的内容哈希。

### ProgramValue.materialize() { #programvaluematerialize }

<!-- typed-contract:start -->

无显式参数（实例方法的 `self` 省略）。

返回类型：`torch.Tensor | tg.Tensor`。

运行示例前，先按下文配置所需设备、文件或分布式环境。

??? example "最小用法"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    class Sum(tg.MessagePassing):
        reducer = tg.sum()
        def edge(self, src, dst, edge):
            return src.x
    program = Sum()
    x = torch.tensor([2., 3.], requires_grad=True)
    composition = tg.GraphProgram()
    value = composition.apply(program, graph=graph, src={'x': x}, dst={})
    result = value.materialize()
    ```

<!-- typed-contract:end -->

在此观察边界对所属程序做 JIT 与执行，返回具体值。`value.ir(stage)`
检查所属程序。

## visualize { #visualize }

### tg.visualize.heatmap(values, *, vmin=0.0, vmax=1.0, low=None, high=None, cmap=None) { #gfvisualizeheatmap }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `values` | <code>tg.Tensor</code> | 连续 rank-2 原生浮点 Tensor。Torch 输入当前须经 tg.from_torch；此转换不连接 autograd。 |
| `vmin / vmax` | <code>float &#124; None</code> | 标量颜色范围；几何渲染器的 None 从数据推断。heatmap 默认 0 和 1。 |
| `low / high` | <code>tuple[float, float, float] &#124; None</code> | 可选的双色渐变 RGB 端点，各通道在 [0,1] 内。 |
| `cmap` | <code>str &#124; Sequence[RGB] &#124; Sequence[tuple[float, RGB]] &#124; None</code> | 内置色图名或 [0,1] 内的颜色节点；显式 cmap 覆盖 low/high。 |

返回类型：`tg.visualize.Raster`。

??? example "最小用法"

    ```python
    import tiga as tg
    raster = tg.visualize.heatmap(tg.tensor([[0., 0.5], [0.75, 1.]]))
    assert raster.to_numpy().shape == (2, 2, 3)
    ```

<!-- typed-contract:end -->

把一个连续 rank-2 浮点 Tensor 的热力图编译为一个融合 kernel，返回惰性
RGB `Raster`。默认 colormap 为 `viridis`；`cmap` 选择其他多停靠点
[colormap](https://zh.wikipedia.org/wiki/%E9%A2%9C%E8%89%B2%E6%98%A0%E5%B0%84)
——取自 `tg.visualize.colormaps()` 的名称、等间距 RGB 颜色列表，或位置
在 [0, 1] 内严格递增的 `(position, RGB)` 停靠点列表——`low`/`high` 则
选择朴素的双色 ramp（缺省的一端回落到对应的 viridis 端点）。双色 ramp
下超出 `[vmin,vmax]` 的值在惰性 Tensor 中不截断，保持表达式可微；多
停靠点 colormap 与 Matplotlib 一样钳制到端点颜色。颜色变换就是普通的
Tensor 广播/算术（多停靠点 ramp 用 `sqrt(x²)` 构造 tent 函数），编译器
核心中不存在可视化操作。示例：
[GPU 热力图可视化准备](examples/visualization.zh.md#gpu-heatmap-visualization-prep)。

### tg.visualize.colormaps() { #gfvisualizecolormaps }

<!-- typed-contract:start -->

无显式参数（实例方法的 `self` 省略）。

返回类型：`tuple[str, ...]`。

??? example "最小用法"

    ```python
    import tiga as tg
    print(tg.visualize.colormaps())
    ```

<!-- typed-contract:end -->

返回内置热力图 colormap 的名称：`viridis`、`magma`、`plasma`、
`inferno`、`jet`、`coolwarm`、`gray`——同名 Matplotlib colormap 的
八停靠点采样，在融合 kernel 内做分段线性插值。

### Raster { #raster }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `pixels` | <code>tg.Tensor</code> | 惰性 RGB 像素值；优先使用渲染工厂而非手动构造。 |
| `width / height` | <code>int</code> | 正的图像像素宽高，默认各 512。 |
| `to_numpy: clip` | <code>bool</code> | 是否将导出 RGB 裁剪至 [0,1]；默认 True。 |
| `save: path` | <code>str &#124; os.PathLike</code> | 图像目标；Pillow 编码，已有文件会被替换。 |
| `mlir: verify` | <code>bool</code> | 是否执行 IR verifier；默认 False。 |
| `generated_code: kind` | <code>str &#124; None</code> | 产物名称；与 Tensor.generated_code 一致。 |
| `show: **imshow_options` | <code>object</code> | 转交 Matplotlib imshow 的关键字参数。 |

返回类型：`Raster; to_numpy → numpy.ndarray; save → pathlib.Path; show → None`。

??? example "最小用法"

    ```python
    import tiga as tg
    raster = tg.visualize.heatmap(tg.tensor([[0., 1.]]))
    print(raster.realize().execution)
    print(raster.to_numpy())
    ```

<!-- typed-contract:end -->

惰性 RGB 光栅，其像素仍是 Tiga Tensor：`realize()`、
`prepare()`、`execution`、`mlir(*, verify=False)` 与
`generated_code(kind)` 暴露与 `Tensor` 相同的编译器/运行时路径；
`to_numpy(*, clip=True)`、`save(path)`（Pillow）与
`show(**imshow_options)`（Matplotlib）是可选的主机互操作/编码辅助
方法。

### tg.visualize.Camera { #gfvisualizecamera }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `position / target / up` | <code>tuple[float, float, float]</code> | 世界坐标下相机位置、观察目标与上方向；up 默认 (0,0,1)。 |
| `fov` | <code>float</code> | 角度制垂直视场角；默认 45。 |
| `auto / from_angles: positions` | <code>array-like &#124; None</code> | 用于取景的点云；auto 必填。 |
| `auto: margin` | <code>float</code> | 包围球距离倍率；默认 1.2。 |
| `from_angles: elevation / azimuth` | <code>float</code> | 角度制相机角；默认 30 和 -60。 |
| `from_angles: distance` | <code>float &#124; None</code> | 相机到目标的距离；None 按 positions 或默认值计算。 |
| `world_to_ndc: points_xyz` | <code>numpy.ndarray</code> | 世界坐标 (N,3)。 |

返回类型：`Camera; world_to_ndc → tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]`。

??? example "最小用法"

    ```python
    import tiga as tg
    positions = [[0., 0.], [1., 0.], [0., 1.]]
    camera = tg.visualize.Camera.auto(positions)
    ```

<!-- typed-contract:end -->

主机端透视 camera，用于点云渲染：
`Camera(position, target, fov=45.0, up=(0.0, 0.0, 1.0))`；
`Camera.auto(positions, *, fov=45.0, margin=1.2)` 用包围球拟合 target
与距离——(N, 2) 平面输入按俯视处理，3-D 输入沿最小方差主轴取景，
各向同性点云回退到等轴测方向；
`Camera.from_angles(elevation=30.0, azimuth=-60.0, *, positions=None, distance=None, fov=45.0)`
把 camera 放在围绕 target 的球面上。
`camera.world_to_ndc(points_xyz)` 把 (N, 3) 点投影为
`(ndc, depth, visible)`。示例：
[粒子/视频](examples/visualization.zh.md#particles-and-video)。

### tg.visualize.particles(positions, values=None, *, camera="auto", width=512, height=512, point_radius=2.0, vmin=None, vmax=None, low=None, high=None, cmap=None) { #gfvisualizeparticles }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `positions` | <code>torch.Tensor &#124; tg.Tensor &#124; numpy.ndarray &#124; Sequence</code> | (N,2) 或 (N,3) 坐标，几何渲染/导出会复制到 host。 |
| `values` | <code>torch.Tensor &#124; tg.Tensor &#124; numpy.ndarray &#124; Sequence &#124; None</code> | 每顶点一个标量；None 使用渲染器的均匀值/密度默认值。 |
| `camera` | <code>Camera &#124; Literal[&#x27;auto&#x27;] &#124; tuple[float, float]</code> | 显式相机、自动取景或角度制 elevation/azimuth。 |
| `width / height` | <code>int</code> | 正的图像像素宽高，默认各 512。 |
| `point_radius` | <code>float</code> | 圆盘像素半径；默认 2。 |
| `vmin / vmax` | <code>float &#124; None</code> | 标量颜色范围；几何渲染器的 None 从数据推断。heatmap 默认 0 和 1。 |
| `low / high` | <code>tuple[float, float, float] &#124; None</code> | 可选的双色渐变 RGB 端点，各通道在 [0,1] 内。 |
| `cmap` | <code>str &#124; Sequence[RGB] &#124; Sequence[tuple[float, RGB]] &#124; None</code> | 内置色图名或 [0,1] 内的颜色节点；显式 cmap 覆盖 low/high。 |

返回类型：`Raster`。

??? example "最小用法"

    ```python
    import tiga as tg
    positions = [[0., 0.], [1., 0.], [0., 1.]]
    raster = tg.visualize.particles(positions, width=32, height=32)
    ```

<!-- typed-contract:end -->

把每个点以圆盘溅射到主机端标量场——取点的值，`values=None` 时累加
单位密度——然后复用 `heatmap` 的 Tensor 表达式为该标量场上色，返回
惰性 `Raster`。`camera` 接受 `"auto"`、`Camera` 实例或
`(elevation, azimuth)` 元组；`vmin`/`vmax` 默认取标量场的数据范围；
`cmap` 选用多停靠点 colormap（见 [`heatmap`](#gfvisualizeheatmap)），
覆盖 `low`/`high`。示例：
[从散点生成 Delaunay mesh](examples/visualization.zh.md#simulation-meshes-load-shade-orbit)。

### tg.visualize.delaunay(positions, values=None, *, camera="auto", width=512, height=512, vmin=None, vmax=None, low=None, high=None, cmap=None, wireframe=False) { #gfvisualizedelaunay }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `positions` | <code>torch.Tensor &#124; tg.Tensor &#124; numpy.ndarray &#124; Sequence</code> | (N,2) 或 (N,3) 坐标，几何渲染/导出会复制到 host。 |
| `values` | <code>torch.Tensor &#124; tg.Tensor &#124; numpy.ndarray &#124; Sequence &#124; None</code> | 每顶点一个标量；None 使用渲染器的均匀值/密度默认值。 |
| `camera` | <code>Camera &#124; Literal[&#x27;auto&#x27;] &#124; tuple[float, float]</code> | 显式相机、自动取景或角度制 elevation/azimuth。 |
| `width / height` | <code>int</code> | 正的图像像素宽高，默认各 512。 |
| `vmin / vmax` | <code>float &#124; None</code> | 标量颜色范围；几何渲染器的 None 从数据推断。heatmap 默认 0 和 1。 |
| `low / high` | <code>tuple[float, float, float] &#124; None</code> | 可选的双色渐变 RGB 端点，各通道在 [0,1] 内。 |
| `cmap` | <code>str &#124; Sequence[RGB] &#124; Sequence[tuple[float, RGB]] &#124; None</code> | 内置色图名或 [0,1] 内的颜色节点；显式 cmap 覆盖 low/high。 |
| `wireframe` | <code>bool</code> | 是否仅绘制边框；默认 False。 |

返回类型：`Raster`。

运行示例前，先按下文配置所需设备、文件或分布式环境。

??? example "最小用法"

    ```python
    import tiga as tg
    positions = [[0., 0.], [1., 0.], [0., 1.]]
    # Requires Matplotlib for triangulation.
    raster = tg.visualize.delaunay(positions, width=32, height=32)
    ```

<!-- typed-contract:end -->

对 positions 做三角剖分——(N, 2) 平面输入直接剖分，3-D 输入先经
camera 投影——每个三角形填充顶点值均值（`values=None` 时用均匀值；
`wireframe=True` 改为绘制三角形边线），再经 `heatmap` 为标量场上色
（`cmap` 覆盖 `low`/`high`），返回惰性 `Raster`。示例：
[mesh 渲染](examples/visualization.zh.md#simulation-meshes-load-shade-orbit)。

### tg.visualize.mesh(positions, faces, values=None, *, camera="auto", width=512, height=512, wireframe=False, vmin=None, vmax=None, low=None, high=None, cmap=None) { #gfvisualizemesh }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `positions` | <code>torch.Tensor &#124; tg.Tensor &#124; numpy.ndarray &#124; Sequence</code> | (N,2) 或 (N,3) 坐标，几何渲染/导出会复制到 host。 |
| `faces` | <code>numpy.ndarray &#124; Sequence[Sequence[int]]</code> | (M,3) 三角形顶点编号，均须在有效范围内。 |
| `values` | <code>torch.Tensor &#124; tg.Tensor &#124; numpy.ndarray &#124; Sequence &#124; None</code> | 每顶点一个标量；None 使用渲染器的均匀值/密度默认值。 |
| `camera` | <code>Camera &#124; Literal[&#x27;auto&#x27;] &#124; tuple[float, float]</code> | 显式相机、自动取景或角度制 elevation/azimuth。 |
| `width / height` | <code>int</code> | 正的图像像素宽高，默认各 512。 |
| `wireframe` | <code>bool</code> | 是否仅绘制边框；默认 False。 |
| `vmin / vmax` | <code>float &#124; None</code> | 标量颜色范围；几何渲染器的 None 从数据推断。heatmap 默认 0 和 1。 |
| `low / high` | <code>tuple[float, float, float] &#124; None</code> | 可选的双色渐变 RGB 端点，各通道在 [0,1] 内。 |
| `cmap` | <code>str &#124; Sequence[RGB] &#124; Sequence[tuple[float, RGB]] &#124; None</code> | 内置色图名或 [0,1] 内的颜色节点；显式 cmap 覆盖 low/high。 |

返回类型：`Raster`。

??? example "最小用法"

    ```python
    import tiga as tg
    positions = [[0., 0.], [1., 0.], [0., 1.]]
    raster = tg.visualize.mesh(positions, [[0, 1, 2]], width=32, height=32)
    ```

<!-- typed-contract:end -->

渲染给定三角网格——`delaunay` 是从点现算三角剖分，`mesh` 接受显式的
`faces`（(M, 3) 索引数组，例如仿真网格或经 `load_obj` 加载的模型）。
三角形按顶点深度均值从远到近绘制（painter's algorithm），近处三角形
遮挡远处者；每个三角形填充其有限顶点值的均值（`values=None` 时用均匀
值；顶点全非有限的三角形跳过），`wireframe=True` 改为画边线。camera
处理、平面输入与相机后方裁剪策略与 `delaunay` 一致。示例：
[仿真网格：加载、着色、环绕](examples/visualization.zh.md#simulation-meshes-load-shade-orbit)。

<span id="gfvisualizesplats"></span>

### tg.visualize.gaussians(positions, colors, scales, *, rotations=None, opacities=None, camera="auto", width=512, height=512, cutoff=3.0, background=(0,0,0)) { #tgvisualizegaussians }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `positions` | <code>array-like</code> | 三维均值 (N,3)；几何部分在 CPU 渲染，不可微。 |
| `colors` | <code>numpy.ndarray &#124; Sequence</code> | RGB (N,3)，值在 [0,1] 内。 |
| `scales` | <code>numpy.ndarray &#124; Sequence</code> | 正的标准差 (N,) 或 (N,3)。 |
| `rotations` | <code>numpy.ndarray &#124; Sequence &#124; None</code> | 单位四元数 (N,4)，顺序 w,x,y,z；None 为单位旋转。 |
| `opacities` | <code>numpy.ndarray &#124; Sequence &#124; None</code> | [0,1] 内的不透明度 (N,)；None 为完全不透明。 |
| `camera` | <code>Camera &#124; Literal[&#x27;auto&#x27;] &#124; tuple[float, float]</code> | 显式相机、自动取景或角度制 elevation/azimuth。 |
| `width / height` | <code>int</code> | 正的图像像素宽高，默认各 512。 |
| `cutoff` | <code>float</code> | 以标准差计的椭圆截断半径；默认 3。 |
| `background` | <code>tuple[float, float, float]</code> | [0,1] 内的背景 RGB；默认黑色。 |

返回类型：`Raster`。

??? example "最小用法"

    ```python
    import tiga as tg
    raster = tg.visualize.gaussians([[0., 0., 0.]], [[1., 0., 0.]], [0.1], width=32, height=32)
    ```

<!-- typed-contract:end -->

`tg.visualize.splats` 保留为兼容别名，新代码使用 `gaussians`。

把各向异性 3-D 高斯渲染为惰性 `Raster`（EWA splatting，与 3-D Gaussian
Splatting 相同）：协方差 `R·diag(scales²)·Rᵀ` 经透视投影的 Jacobian 投成
2-D 椭圆，在 `cutoff` 个标准差内求值，颜色按 front-to-back transmittance
从近到远做 alpha 合成。`colors` (N, 3) 为 [0, 1] 内的 RGB，`scales`
(N, 3) 或 (N,) 为正的标准差，`rotations` (N, 4) 为 `(w, x, y, z)`
quaternion，`opacities` (N,) 在 [0, 1] 内。相机后方、亚像素或衰减到
1/255 以下的高斯被跳过。示例：
[Gaussian splats 与体渲染](examples/visualization.zh.md#gaussian-splats-and-volumes)。

### tg.visualize.volume(density, *, camera="auto", width=512, height=512, steps=128, cmap="viridis", vmin=None, vmax=None, scale=8.0, background=(0,0,0)) { #gfvisualizevolume }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `density` | <code>torch.Tensor &#124; tg.Tensor &#124; numpy.ndarray</code> | 三维密度网格；host ray marching 不可微。 |
| `camera` | <code>Camera &#124; Literal[&#x27;auto&#x27;] &#124; tuple[float, float]</code> | 显式相机、自动取景或角度制 elevation/azimuth。 |
| `width / height` | <code>int</code> | 正的图像像素宽高，默认各 512。 |
| `steps` | <code>int</code> | 每条射线采样次数，正整数；默认 128。 |
| `cmap` | <code>str &#124; Sequence[RGB] &#124; Sequence[tuple[float, RGB]] &#124; None</code> | 内置色图名或 [0,1] 内的颜色节点；显式 cmap 覆盖 low/high。 |
| `vmin / vmax` | <code>float &#124; None</code> | 标量颜色范围；几何渲染器的 None 从数据推断。heatmap 默认 0 和 1。 |
| `scale` | <code>float</code> | 吸收倍率；默认 8。 |
| `background` | <code>tuple[float, float, float]</code> | [0,1] 内的背景 RGB；默认黑色。 |

返回类型：`Raster`。

??? example "最小用法"

    ```python
    import numpy as np
    import tiga as tg
    raster = tg.visualize.volume(np.ones((3, 3, 3)), width=8, height=8, steps=8)
    ```

<!-- typed-contract:end -->

对密度网格 (X, Y, Z)（映射到单位立方体）做 ray marching，返回惰性
`Raster`：每条像素光线上的三线性采样发出归一化密度的 `cmap` 颜色，
并按 `alpha = 1 - exp(-density·scale·Δt)` 吸收（emission-absorption
模型）。`vmin`/`vmax` 默认取数据范围；`camera` 为 `"auto"` 时取景
整个立方体。示例：
[Gaussian splats 与体渲染](examples/visualization.zh.md#gaussian-splats-and-volumes)。

### tg.visualize.load_ply(path) { #gfvisualizeloadply }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `path` | <code>str &#124; os.PathLike</code> | 文件系统路径，不是 URL；父目录和覆盖规则见下文。 |

返回类型：`tuple[numpy.ndarray, numpy.ndarray | None, dict[str, numpy.ndarray]]`。

??? example "最小用法"

    ```python
    import tiga as tg
    positions = [[0., 0.], [1., 0.], [0., 1.]]
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'triangle.ply'
        tg.visualize.export_ply(path, positions, faces=[[0, 1, 2]])
        points, faces, extras = tg.visualize.load_ply(path)
    ```

<!-- typed-contract:end -->

解析 PLY 文件（ASCII 或 binary_little_endian），返回
`(positions, faces, extras)`：由 vertex 的 `x y z` 属性构成的 (N, 3)
float64 数组、face 的列表属性构成的 (M, K) int64 数组（无 face 时为
`None`），以及其余每个 vertex 属性组成的 dict——3-D Gaussian Splatting
载荷（`f_dc_*`、`opacity`、`scale_*`、`rot_*`）的入口。示例：
[Gaussian splats 与体渲染](examples/visualization.zh.md#gaussian-splats-and-volumes)。

### tg.visualize.load_obj(path) { #gfvisualizeloadobj }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `path` | <code>str &#124; os.PathLike</code> | 文件系统路径，不是 URL；父目录和覆盖规则见下文。 |

返回类型：`tuple[numpy.ndarray, numpy.ndarray]`。

??? example "最小用法"

    ```python
    import tiga as tg
    positions = [[0., 0.], [1., 0.], [0., 1.]]
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'triangle.obj'
        tg.visualize.export_obj(path, positions, faces=[[0, 1, 2]])
        points, faces = tg.visualize.load_obj(path)
    ```

<!-- typed-contract:end -->

把极简 Wavefront OBJ 文件解析为 `(positions, faces)`：`v x y z` 顶点
行变成 (N, 3) float64 数组，`f` 面行（含 `f a/b/c` 形式，只取顶点
索引）变成 (M, 3) 整数数组，1-based 索引就地解析、多边形扇形三角化。
空文件、缺少顶点/面、索引越界都会抛出 `ValueError`。示例：
[仿真网格：加载、着色、环绕](examples/visualization.zh.md#simulation-meshes-load-shade-orbit)。

### tg.visualize.export_ply(path, positions, *, faces=None, values=None, low=None, high=None, cmap=None) { #gfvisualizeexportply }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `path` | <code>str &#124; os.PathLike</code> | 文件系统路径，不是 URL；父目录和覆盖规则见下文。 |
| `positions` | <code>torch.Tensor &#124; tg.Tensor &#124; numpy.ndarray &#124; Sequence</code> | (N,2) 或 (N,3) 坐标，几何渲染/导出会复制到 host。 |
| `faces` | <code>array-like &#124; None</code> | 可选三角形顶点编号 (M,3)。 |
| `values` | <code>torch.Tensor &#124; tg.Tensor &#124; numpy.ndarray &#124; Sequence &#124; None</code> | 每顶点一个标量；None 使用渲染器的均匀值/密度默认值。 |
| `low / high` | <code>tuple[float, float, float] &#124; None</code> | 可选的双色渐变 RGB 端点，各通道在 [0,1] 内。 |
| `cmap` | <code>str &#124; Sequence[RGB] &#124; Sequence[tuple[float, RGB]] &#124; None</code> | 内置色图名或 [0,1] 内的颜色节点；显式 cmap 覆盖 low/high。 |

返回类型：`pathlib.Path`。

??? example "最小用法"

    ```python
    import tiga as tg
    positions = [[0., 0.], [1., 0.], [0., 1.]]
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'triangle.ply'
        tg.visualize.export_ply(path, positions, faces=[[0, 1, 2]])
        points, faces, extras = tg.visualize.load_ply(path)
    ```

<!-- typed-contract:end -->

把几何体导出为 binary_little_endian [PLY](https://en.wikipedia.org/wiki/PLY_(file_format))
供 [Blender](https://en.wikipedia.org/wiki/Blender_(software)) 导入：顶点
携带 `x y z` float32（(N, 2) 平面输入补 `z = 0`），可选 `faces` (M, 3)
写成 `vertex_indices` 列表元素，`values` 额外携带 `red green blue`
uint8（与 `heatmap` 同一 colormap——默认 `viridis`，可用 `cmap` 或
`low`/`high` 覆盖——按数据范围归一化）与保存原始场值的
`scalar_value` float32。positions 与 values 经一次主机拷贝（GPU tensor
同样）进入单个结构化缓冲区一次落盘；耗时取决于输入规模、设备传输和文件系统。示例：
[仿真网格：加载、着色、环绕](examples/visualization.zh.md#simulation-meshes-load-shade-orbit)。

### tg.visualize.export_obj(path, positions, faces) { #gfvisualizeexportobj }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `path` | <code>str &#124; os.PathLike</code> | 文件系统路径，不是 URL；父目录和覆盖规则见下文。 |
| `positions` | <code>torch.Tensor &#124; tg.Tensor &#124; numpy.ndarray &#124; Sequence</code> | (N,2) 或 (N,3) 坐标，几何渲染/导出会复制到 host。 |
| `faces` | <code>numpy.ndarray &#124; Sequence[Sequence[int]]</code> | (M,3) 三角形顶点编号，均须在有效范围内。 |

返回类型：`pathlib.Path`。

??? example "最小用法"

    ```python
    import tiga as tg
    positions = [[0., 0.], [1., 0.], [0., 1.]]
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'triangle.obj'
        tg.visualize.export_obj(path, positions, faces=[[0, 1, 2]])
        points, faces = tg.visualize.load_obj(path)
    ```

<!-- typed-contract:end -->

写出文本 Wavefront OBJ——`load_obj` 的精确逆操作：`v` 行按 float64
round-trip 精度写出、`f` 行为 1-based，因此 `load_obj(export_obj(...))`
恒等恢复坐标与面。示例：
[仿真网格：加载、着色、环绕](examples/visualization.zh.md#simulation-meshes-load-shade-orbit)。

### tg.visualize.export_vdb(path, positions, values, *, voxel_size=0.05) { #gfvisualizeexportvdb }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `path` | <code>str &#124; os.PathLike</code> | 文件系统路径，不是 URL；父目录和覆盖规则见下文。 |
| `positions` | <code>torch.Tensor &#124; tg.Tensor &#124; numpy.ndarray &#124; Sequence</code> | (N,2) 或 (N,3) 坐标，几何渲染/导出会复制到 host。 |
| `values` | <code>torch.Tensor &#124; tg.Tensor &#124; numpy.ndarray &#124; Sequence &#124; None</code> | 每顶点一个标量；None 使用渲染器的均匀值/密度默认值。 |
| `voxel_size` | <code>float</code> | 世界坐标体素边长，正数；默认 0.05。 |

返回类型：`pathlib.Path`。

运行示例前，先按下文配置所需设备、文件或分布式环境。

??? example "最小用法"

    ```python
    import tiga as tg
    positions = [[0., 0.], [1., 0.], [0., 1.]]
    # Requires pyopenvdb installed separately.
    tg.visualize.export_vdb('cloud.vdb', positions, [1., 2., 3.])
    ```

<!-- typed-contract:end -->

把点云与标量场栅格化为 [OpenVDB](https://en.wikipedia.org/wiki/OpenVDB)
dense grid。需要可选依赖 `pyopenvdb`——无依赖的 PLY 导出已覆盖 Blender
互导路径，仅在明确需要体素网格时才引入 OpenVDB；缺失时抛出
`ModuleNotFoundError`。

### tg.visualize.save_video(frames, path, *, fps=30) { #gfvisualizesavevideo }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `frames` | <code>Iterable[Raster &#124; numpy.ndarray]</code> | 尺寸一致的 RGB 帧；数组为 (H,W,3)，值在 [0,1] 内。 |
| `path` | <code>str &#124; os.PathLike</code> | 文件系统路径，不是 URL；父目录和覆盖规则见下文。 |
| `fps` | <code>float</code> | 有限的正播放帧率；拒绝 bool 与非有限值，默认 30。 |

返回类型：`pathlib.Path`。

??? example "最小用法"

    ```python
    import tempfile
    from pathlib import Path
    import tiga as tg
    frame = tg.visualize.heatmap(tg.tensor([[0., 1.]]))
    with tempfile.TemporaryDirectory() as directory:
        tg.visualize.save_video([frame, frame], Path(directory) / 'demo.gif', fps=10)
    ```

<!-- typed-contract:end -->

把任意帧迭代对象——`Raster` 或 [0, 1] 范围内的浮点 (H, W, 3) 数组——
编码为 `.gif`（Pillow）或 `.mp4`（原始 RGB 逐帧管道送入 ffmpeg 子
进程）。MP4 逐帧消费生成器；GIF 当前先把全部帧缓存在 host 内存再编码。
已有目标文件会被替换。示例：
[粒子/视频](examples/visualization.zh.md#particles-and-video)。

## 编译器类型 { #compiler-types }

### tg.compiler.Dim / TensorSpec / ShapeSpecializer { #gfcompiler-symbolic-shapes }

<!-- typed-contract:start -->

| 参数 | 类型 | 含义 |
|---|---|---|
| `Dim: name` | <code>str</code> | 跨参数 shape 共享的符号标识符。 |
| `Dim: minimum / maximum / multiple_of` | <code>int / int &#124; None / int</code> | 含端点上下界与正的整除约束；默认 1、None、1。 |
| `TensorSpec: shape` | <code>Sequence[int &#124; Dim]</code> | 非负具体尺寸或受约束符号维。 |
| `dtype` | <code>tg.DType &#124; None</code> | 原生数值类型；允许 None 的构造器按输入推断。 |
| `device` | <code>str &#124; tg.Device &#124; None</code> | 执行设备；None 按本条构造器的默认规则处理。 |
| `ShapeSpecializer: specs` | <code>Sequence[TensorSpec]</code> | 每个位置输入一个规格。 |
| `ShapeSpecializer: compiler` | <code>Callable[[ShapeBinding], Callable]</code> | 为验证通过的具体 shape 绑定构建可调用对象。 |
| `ShapeSpecializer.__call__: *values` | <code>tg.Tensor</code> | 在查找/编译特化前验证的输入。 |

返回类型：`Dim / TensorSpec / ShapeSpecializer; calling specializer → compiler-defined result`。

??? example "最小用法"

    ```python
    import tiga as tg
    n = tg.compiler.Dim('N', maximum=16)
    spec = tg.compiler.TensorSpec((n,))
    run = tg.compiler.ShapeSpecializer([spec], lambda binding: lambda x: x * 2)
    assert run(tg.tensor([1., 2.])).tolist() == [2., 4.]
    ```

<!-- typed-contract:end -->

有界符号签名：`Dim(name, minimum=1, maximum=None, multiple_of=1)`、
`TensorSpec(shape, *, dtype=tg.float32, device=None)` 与
`ShapeSpecializer(specs, compiler)`。同一个符号在多个参数中重复出现
时必须绑定到同一长度；最小值/最大值/可整除性守卫在编译前检查，每个
合法绑定形成一个具体的 MLIR 特化/缓存键。指南：
[Tensor 运行时与 autograd](runtime-and-autograd.zh.md#current-alpha-slice)。
