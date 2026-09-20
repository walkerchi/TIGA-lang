"""Maintain typed parameter tables and usage snippets in both API references.

Run with --check in CI; no Torch, compiler or documentation dependency is needed.
Existing prose and stable anchors are preserved. Contracts are curated, not
inferred from parameter names or from an untyped Python signature.
"""
from __future__ import annotations

import argparse
import html
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
START = "<!-- typed-contract:start -->"
END = "<!-- typed-contract:end -->"
CONTRACTS = {}


def p(name, kind, en, zh):
    return name, kind, en, zh


DEVICE = p("device", "str | tg.Device | None", "Execution device; None uses the constructor's documented default.", "执行设备；None 按本条构造器的默认规则处理。")
DTYPE = p("dtype", "tg.DType | None", "Native value dtype; None infers from input data where supported.", "原生数值类型；允许 None 的构造器按输入推断。")
GRAD = p("requires_grad", "bool", "Record differentiable floating/complex inputs; default False.", "是否记录浮点/复数输入的求导关系；默认 False。")
PATH = p("path", "str | os.PathLike", "Filesystem path, not a URL; parent and overwrite rules are described below.", "文件系统路径，不是 URL；父目录和覆盖规则见下文。")
OVERWRITE = p("overwrite", "bool", "Allow replacing an existing Tensor snapshot; default False.", "是否允许替换已有 Tensor 快照；默认 False。")
NATIVE = "tg.Tensor"
INDEX = "torch.Tensor | tg.Tensor"
FIELD = "Mapping[str, torch.Tensor | tg.Tensor]"
ROW = p("row_ptr", INDEX, "1-D int32/int64 destination-row boundaries, length num_dst + 1.", "一维 int32/int64 目标行边界，长度为 num_dst + 1。")
COL = p("col_idx", INDEX, "1-D source IDs in edge order, same device/dtype as row_ptr.", "按边顺序排列的一维源编号，与 row_ptr 同设备、同 dtype。")
NUM_SRC = p("num_src", "int | None", "Source-node count; None infers from indices. Supply it to retain isolated sources.", "源节点数；None 按索引推断。保留孤立源节点时显式提供。")
NUM_DST = p("num_dst", "int | None", "Destination-node count; see the constructor's inference rule.", "目标节点数；省略时的推断规则见本条说明。")
INDEX_DTYPE = p("index_dtype", "tg.DType", "tg.int32 or tg.int64; default tg.int64.", "tg.int32 或 tg.int64；默认 tg.int64。")
GRAPH = p("graph", "tg.Graph", "Relation consumed by this call; device and entity counts must match fields.", "本次调用的关系；设备、实体数须与字段匹配。")
SRC = p("src", FIELD + " | None", "Source fields, leading dimension num_src; use {} for an unused role.", "源字段，首维为 num_src；未使用该角色时传 {}。")
DST = p("dst", FIELD + " | None", "Destination fields, leading dimension num_dst; use {} for an unused role.", "目标字段，首维为 num_dst；未使用该角色时传 {}。")
EDGE = p("edge", FIELD + " | None", "Edge fields in graph edge order; None means no explicit edge fields.", "按图的边顺序绑定的字段；None 表示没有显式边字段。")
NDATA = p("ndata", FIELD + " | None", "Homogeneous-node shorthand, mutually exclusive with src/dst.", "同构图节点字段简写，不能与 src/dst 混用。")
PARAMS = p("**params", "object", "Named scalar/captured arguments declared by edge/node; not arbitrary Python objects.", "edge/node 签名声明的标量或可捕获参数；不代表支持任意 Python 对象。")
DETERMINISTIC = p("deterministic", "bool", "Request fixed reduction ordering; default False, subject to backend support.", "是否要求固定归约顺序；默认 False，受 backend 支持范围限制。")
CHECKPOINT = p("checkpoint", "Literal['auto', 'save', 'recompute']", "Forward-value storage policy for backward.", "反向所需前向值的存储策略。")
OUTPUT = p("output", NATIVE, "Native differentiable output, not a Torch Tensor.", "参与原生求导的输出，不接受 Torch Tensor。")
INPUTS = p("inputs", "tg.Tensor | Sequence[tg.Tensor]", "Differentiation targets with requires_grad=True; sequence returns a tuple.", "requires_grad=True 的求导目标；传序列时梯度以 tuple 返回。")
COTANGENT = p("grad_output", "tg.Tensor | None", "Output cotangent with matching shape/dtype/device; required for non-scalar or complex output.", "与输出 shape/dtype/device 一致的 cotangent；非标量或复数输出必填。")
NATIVE_SETUP = "import tiga as tg\nx = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)\n"
GRAPH_SETUP = "import torch\nimport tiga as tg\nrow_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)\ncol_idx = torch.tensor([0, 1, 1], dtype=torch.int64)\ngraph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)\n"
MP_SETUP = GRAPH_SETUP + "class Sum(tg.MessagePassing):\n    reducer = tg.sum()\n    def edge(self, src, dst, edge):\n        return src.x\nprogram = Sum()\nx = torch.tensor([2., 3.], requires_grad=True)\n"


def add(anchor, params, result, example, *, runnable=False):
    CONTRACTS[anchor] = dict(params=params, result=result, example=example, runnable=runnable)


add("gftensor", [p("data", "bool | int | float | complex | Sequence", "Scalar or rectangular nested values.", "标量或规则嵌套数值序列。"), DTYPE, DEVICE, GRAD], NATIVE, "import tiga as tg\nx = tg.tensor([1., 2.], requires_grad=True)", runnable=True)
add("gfempty", [p("shape", "int | Sequence[int]", "Non-negative dimension sizes; elements are uninitialized.", "非负维度长度；元素值未初始化。"), p("dtype", "tg.DType", "Native element dtype, default tg.float32; None is not accepted.", "原生元素类型，默认 tg.float32；不接受 None。"), DEVICE, GRAD], NATIVE, "import tiga as tg\nbuffer = tg.empty((2, 3), dtype=tg.float32)", runnable=True)
for anchor, name in (("gfzeros_like", "zeros_like"), ("gfones_like", "ones_like")):
    add(anchor, [p("value", NATIVE, "Template for shape, dtype and device.", "shape、dtype 和设备的模板。"), GRAD], NATIVE, NATIVE_SETUP + f"result = tg.{name}(x)", runnable=True)
add("gffrom_torch", [p("value", "torch.Tensor", "Dense Torch-owned Tensor; wrapping shares storage, not autograd history.", "稠密 Torch Tensor；包装共享存储，不连接两套 autograd 历史。"), p("requires_grad", "bool | None", "None inherits the source flag; otherwise set the native leaf flag.", "None 继承源标志；否则设置原生叶子的求导标志。")], NATIVE, "import torch\nimport tiga as tg\nnative = tg.from_torch(torch.tensor([1., 2.]))", runnable=True)
for anchor, name, params, call in (
    ("tensorreshape", "reshape", [p("*shape", "int | Sequence[int]", "New extents; at most one -1, with unchanged element count.", "新形状；最多一个 -1，总元素数不变。")], "reshape(4)"),
    ("tensorpermute", "permute", [p("*axes", "int | Sequence[int]", "A permutation containing each axis exactly once.", "包含每个轴且不重复的排列。")], "permute(1, 0)"),
    ("tensortranspose", "transpose", [p("dim0 / dim1", "int", "The two axes to swap; negative axes are normalized.", "交换的两个轴；负轴编号会被规范化。")], "transpose(0, 1)"),
    ("tensorsqueeze", "squeeze", [p("dim", "int | None", "Remove singleton axis dim, or all singleton axes when None.", "删除指定的长度 1 轴；None 删除全部长度 1 轴。")], "squeeze()"),
    ("tensorunsqueeze", "unsqueeze", [p("dim", "int", "Insertion position of a new size-one axis.", "插入长度 1 的新轴的位置。")], "unsqueeze(0)"),
    ("tensorbroadcast_to", "broadcast_to", [p("shape", "Sequence[int]", "Broadcast-compatible target extents, aligned from the right.", "按右对齐规则可广播的目标形状。")], "broadcast_to((3, 2, 2))"),
    ("tensormatmul", "matmul", [p("other", NATIVE, "Rank-2 right operand with compatible contraction dimension.", "收缩维匹配的 rank-2 右操作数。")], "matmul(x)"),
    ("tensorexp", "exp", [], "exp()"), ("tensorsqrt", "sqrt", [], "sqrt()"), ("tensorconj", "conj", [], "conj()"),
    ("tensorcumsum", "cumsum", [p("dim", "int", "Axis for the inclusive scan.", "包含式扫描的轴。"), p("reverse", "bool", "Scan from the last element when True; default False.", "True 从末端开始扫描；默认 False。")], "cumsum(1, reverse=True)"),
    ("tensorsum", "sum", [p("axis", "int | Sequence[int] | None", "Reduction axes; None reduces all axes.", "归约轴；None 归约全部轴。"), p("keepdims", "bool", "Keep reduced axes with length one; default False.", "保留长度 1 的归约轴；默认 False。")], "sum(axis=1, keepdims=True)"),
    ("tensorcheckpoint", "checkpoint", [], "checkpoint()"), ("tensorrealize", "realize", [], "realize()"),
):
    add(anchor, params, NATIVE, NATIVE_SETUP + "result = x." + call, runnable=True)
add("tensor-operators", [p("other", "tg.Tensor | bool | int | float | complex", "Compatible scalar or Tensor operand; @ requires matrix operands.", "兼容的标量或 Tensor 操作数；@ 要求矩阵操作数。")], NATIVE, NATIVE_SETUP + "result = (x + 1) * x\nmask = x > 2", runnable=True)
add("tensorgather", [p("index", NATIVE, "1-D integer row IDs on the same device; repeats are allowed.", "同设备一维整数行号；允许重复。")], NATIVE, NATIVE_SETUP + "result = x.gather(tg.tensor([1, 0, 1], dtype=tg.int64))", runnable=True)
add("tensorsegment_sum", [p("index", NATIVE, "One integer segment ID per input row, same device.", "每个输入行对应一个整数段号，与输入同设备。"), p("num_segments", "int", "Non-negative output row count; indices must be in range.", "非负输出行数；所有段号须在范围内。")], NATIVE, NATIVE_SETUP + "result = x.segment_sum(tg.tensor([0, 0], dtype=tg.int64), 2)", runnable=True)
add("tensorprepare", [], "Callable[[], tg.Tensor]", "# Requires a CUDA provider supporting prepared launches.\nimport tiga as tg\nx = tg.tensor([1., 2.], device='cuda')\nprepared = (x * 2).prepare()\nresult = prepared()")
add("tensorexecution", [], "dict[str, object] | None", NATIVE_SETUP + "result = (x * 2).realize()\nprint(result.execution)", runnable=True)
add("tensorgenerated_code", [p("kind", "str | None", "Artifact name such as llvm or ptx; None selects the default. Availability depends on the executed path.", "产物名称，如 llvm 或 ptx；None 选择默认产物，是否存在取决于实际执行路径。")], "str | bytes | None", NATIVE_SETUP + "result = (x * 2).realize()\nprint(result.generated_code())")
add("tensormlir", [p("verify", "bool", "Run native IR verification when True; default False.", "True 调用原生 IR verifier；默认 False。")], "str", NATIVE_SETUP + "print((x * 2).mlir(verify=True))", runnable=True)
add("tensor-host-interop", [p("copy", "bool", "to_torch only: False shares Torch-owned storage; True permits a copy from native contiguous storage.", "仅用于 to_torch：False 共享 Torch-owned 存储；True 允许拷贝原生连续存储。")], "Python scalar/list / numpy.ndarray / torch.Tensor", NATIVE_SETUP + "values = x.tolist()\narray = x.to_numpy()\ntorch_copy = x.to_torch(copy=True)", runnable=True)
add("gfautogradgrad", [OUTPUT, INPUTS, COTANGENT, p("allow_unused", "bool", "Return None for disconnected inputs when True; otherwise raise.", "True 对未连接输入返回 None；否则报错。"), CHECKPOINT], "tg.Tensor | None | tuple[tg.Tensor | None, ...]", NATIVE_SETUP + "dx = tg.autograd.grad((x * x).sum(), x)\nassert dx.tolist() == [[2., 4.], [6., 8.]]", runnable=True)
add("gfautogradvalue_and_grad", [p("function", "Callable", "Native Tensor function whose selected arguments are differentiable.", "指定参数可微的原生 Tensor 函数。"), p("argnums", "int | Sequence[int]", "Positional argument indices to differentiate; default 0.", "求导的位置参数编号；默认 0。")], "Callable", NATIVE_SETUP + "f = tg.autograd.value_and_grad(lambda a: (a * a).sum())\nvalue, dx = f(x)", runnable=True)
add("gfautogradjoint_plan", [OUTPUT, INPUTS, CHECKPOINT], "JointAutogradPlan", NATIVE_SETUP + "plan = tg.autograd.joint_plan((x * x).sum(), x)\nvalue, dx = plan.run()", runnable=True)
add("gfautogradgrad_mlir", [OUTPUT, p("input", NATIVE, "Single requires_grad input to differentiate.", "单个 requires_grad 求导输入。"), COTANGENT, p("lower", "bool", "Apply the VJP lowering pass; False retains the explicit request.", "是否执行 VJP lowering；False 保留显式请求。")], "str", NATIVE_SETUP + "print(tg.autograd.grad_mlir((x * x).sum(), x))", runnable=True)

add("graphfrom_csr", [ROW, COL, NUM_SRC, p("sorted_by_dst", "bool", "Destination grouping declaration, default True; CSR rows already group destinations.", "目标分组声明，默认 True；CSR 行已经按目标分组。"), p("validate", "Literal['basic', 'full']", "Basic structural checks or full index/boundary validation.", "基础结构检查，或完整索引/边界验证。")], "tg.Graph", GRAPH_SETUP + "assert graph.schema.num_dst == 3", runnable=True)
add("graphfrom_coo", [p("src / dst", INDEX, "Same-length 1-D source/destination integer IDs, same device and dtype.", "等长一维整数源/目标编号，同设备、同 dtype。"), NUM_SRC, NUM_DST], "tg.Graph", "import torch\nimport tiga as tg\ngraph = tg.Graph.from_coo(torch.tensor([0, 1]), torch.tensor([1, 0]), num_src=3, num_dst=3)", runnable=True)
add("graphregular", [p("num_nodes", "int", "Node count.", "节点数。"), p("degree", "int", "Fixed incoming-neighbor count per node.", "每个节点固定的入邻居数。"), DEVICE], "tg.Graph", "import tiga as tg\ngraph = tg.Graph.regular(4, 2)", runnable=True)
add("graphdense", [p("num_src", "int", "Non-negative source count.", "非负源节点数。"), p("num_dst", "int | None", "Destination count; None uses num_src.", "目标节点数；None 使用 num_src。"), DEVICE, INDEX_DTYPE], "tg.Graph", "import tiga as tg\ngraph = tg.Graph.dense(3, 2)\nassert graph.num_edges == 6", runnable=True)
add("graphtriangular", [p("num_entities", "int", "Non-negative node count; destination i receives from 0 through i.", "非负节点数；目标 i 从 0 到 i 接收。"), DEVICE, INDEX_DTYPE], "tg.Graph", "import tiga as tg\ngraph = tg.Graph.triangular(3)\nassert graph.num_edges == 6", runnable=True)
add("graphcu_seqlens", [p("cu_seqlens", INDEX, "1-D cumulative sequence boundaries, starting at zero and nondecreasing.", "从零开始、单调不减的一维序列累计边界。"), p("causal", "bool", "True uses triangular blocks; False uses dense blocks.", "True 使用三角块；False 使用稠密块。")], "tg.Graph", "import torch\nimport tiga as tg\ngraph = tg.Graph.cu_seqlens(torch.tensor([0, 2, 5]), causal=True)", runnable=True)
add("graphcat", [p("graphs", "list[tg.Graph] | tuple[tg.Graph, ...]", "Non-empty same-device blocks; offsets source and destination IDs independently, adding no cross-block edges.", "非空、同设备子图；分别平移源/目标编号，不添加跨块边。")], "tg.Graph", "import tiga as tg\ngraph = tg.Graph.cat([tg.Graph.triangular(2), tg.Graph.triangular(3)])\nassert graph.num_edges == 9", runnable=True)
add("graphstencil", [p("dims", "tuple[int, ...]", "Positive grid extents in row-major order.", "按行主序编号的正整数网格尺寸。"), p("offsets", "tuple[tuple[int, ...], ...] | tg.stencil.Neighborhood | None", "Integer source offsets or a neighborhood macro; None defaults to von_neumann(radius=1, include_center=True).", "整数源偏移或邻域宏；None 默认 von_neumann(radius=1, include_center=True)。"), p("periodic", "bool", "Wrap out-of-grid coordinates when True; otherwise omit them (not zero padding).", "True 对越界坐标取模回绕；否则省略，不是零填充。"), DEVICE], "tg.Graph", "import tiga as tg\ngraph = tg.Graph.stencil((3, 3), tg.stencil.von_neumann())\nassert graph.num_edges == 33", runnable=True)
for name in ("von_neumann", "moore"):
    add("stencil" + name, [p("radius", "int", "Positive inclusive integer radius; default 1, bool rejected.", "正整数闭区间半径，默认 1，不接受 bool。"), p("include_center", "bool", "Include the zero offset; default True.", "是否包含零偏移中心点，默认 True。")], "tg.stencil.Neighborhood", f"import tiga as tg\nmacro = tg.stencil.{name}(include_center=False)\ngraph = tg.Graph.stencil((4, 4), macro)", runnable=True)
add("stenciloffsets", [p("ndim", "int", "Positive number of grid axes; bool rejected.", "正整数网格维数，不接受 bool。")], "tuple[tuple[int, ...], ...]", "import tiga as tg\noffsets = tg.stencil.von_neumann(include_center=False).offsets(2)\nassert offsets == ((-1, 0), (0, -1), (0, 1), (1, 0))", runnable=True)
POS = p("positions", "torch.Tensor | tg.Tensor", "Floating coordinates with shape (N, D), on the target graph device.", "形状 (N, D) 的浮点坐标，位于图所在设备。")
EXCLUDE = p("exclude_self", "bool | None", "Exclude self edges; default depends on this constructor.", "是否排除自环；默认值取决于本构造器。")
add("graphradius", [POS, p("cutoff", "float", "Positive distance cutoff.", "正的距离阈值。"), EXCLUDE, p("fields", "Mapping[str, Tensor] | None", "Builder fields, distinct from MessagePassing bindings.", "关系 builder 字段，与 MessagePassing 绑定不同。"), p("metric", "Callable | None", "Optional distance UDF; None uses Euclidean distance.", "可选距离 UDF；None 为欧氏距离。"), p("select", "Callable | None", "Optional edge-selection UDF; unsupported compilation combinations fail closed.", "可选选边 UDF；不支持的编译组合会明确失败。"), p("periodic", "Tensor | Sequence | None", "Box lengths (D,) or lattice (D,D); None disables periodic boundaries.", "盒长 (D,) 或晶格 (D,D)；None 不使用周期边界。")], "tg.Graph", "import torch\nimport tiga as tg\ngraph = tg.Graph.radius(torch.tensor([[0., 0.], [0.5, 0.]]), cutoff=1.)", runnable=True)
add("graphknn", [POS, p("k", "int", "Number of nearest candidates per destination, subject to count/backend limits.", "每个目标的最近候选数，受实体数量与 backend 限制。"), p("candidates", "torch.Tensor | tg.Tensor | None", "Source coordinates (M,D); None reuses positions for self-kNN.", "源坐标 (M,D)；None 为 positions 自 kNN。"), EXCLUDE], "tg.Graph", "import torch\nimport tiga as tg\ngraph = tg.Graph.knn(torch.tensor([[0., 0.], [1., 0.], [3., 0.]]), k=1)", runnable=True)
add("graphopen", [PATH, DEVICE], "tg.Graph", "import tiga as tg\ngraph = tg.Graph.open('saved-graph.gfg')")
add("gfsave", [GRAPH, PATH, p("fields", "Mapping[str, Mapping[str, tg.Tensor]] | None", "Optional src/dst/edge native field payloads for a graph snapshot.", "图快照中可选的 src/dst/edge 原生字段 payload。")], "None", GRAPH_SETUP + "tg.save(graph, 'new-graph.gfg')")
add("graphfields", [p("role", "Literal['src', 'dst', 'edge']", "Role whose saved fields should be attached; paged graphs only.", "读取对应角色保存的字段，仅支持分页图。"), p("requires_grad", "bool | None", "Native leaf differentiation flag; None (default) is treated as False.", "原生叶子的求导标志；默认 None 按 False 处理。")], "dict[str, tg.Tensor]", "import tiga as tg\ngraph = tg.load('saved-graph.gfg')\nsource_fields = graph.fields('src', requires_grad=True)")
add("graphhalo", [p("mesh", "tg.DeviceMesh", "Logical device mesh; does not start worker processes.", "逻辑设备网格；不会启动 worker 进程。"), p("partition", "tg.ByDestination | None", "Destination partition policy; None uses the default policy.", "目标分区策略；None 使用默认策略。"), p("depth", "int | Literal['auto']", "Non-negative halo depth or compiler inference.", "非负 halo 深度或交由编译器推断。")], "tg.Graph", GRAPH_SETUP + "placed = graph.halo(tg.DeviceMesh('cpu', 2))", runnable=True)
add("graphpaged_rows", [p("begin / end", "int", "Half-open destination-row range within the paged graph.", "分页图内左闭右开的目标行范围。")], "tuple[tuple[int, ...], tuple[int, ...]]", "import tiga as tg\ngraph = tg.load('saved-graph.gfg')\nrow_ptr, col_idx = graph.paged_rows(0, 2)")
for anchor, call, result in (("graphresolve_csr", "resolve_csr()", "tuple[torch.Tensor | tg.Tensor, torch.Tensor | tg.Tensor]"), ("graphtranspose", "transpose()", "tg.Graph"), ("graphexplain", "explain()", "str")):
    add(anchor, [], result, GRAPH_SETUP + f"result = graph.{call}", runnable=True)
add("graph-properties", [], "GraphSchema / tg.Device / int | None / GraphPlacement | None / bool", GRAPH_SETUP + "print(graph.schema, graph.device, graph.num_edges)\nprint(graph.placement, graph.is_distributed)", runnable=True)
add("graph-planner-statistics", [p("samples", "int", "source_index_span_ratio only: sampled row count, default 4096.", "仅用于 source_index_span_ratio：采样行数，默认 4096。")], "Planner statistics (method-specific)", GRAPH_SETUP + "print(graph.degree_bounds())\nprint(graph.degree_statistics())\nprint(graph.fixed_degree())", runnable=True)

add("messagepassing-call", [GRAPH, SRC, DST, EDGE, NDATA, PARAMS], "torch.Tensor | tg.Tensor", MP_SETUP + "out = program(graph=graph, src={'x': x}, dst={})\nout.sum().backward()\nassert x.grad.tolist() == [1., 2.]", runnable=True)
add("messagepassingreference", [p("**kwargs", "Mapping[str, object]", "Same graph, field bindings and UDF arguments as a regular call; uses Torch.", "与普通调用相同的 graph、字段与 UDF 参数；使用 Torch。")], "torch.Tensor", MP_SETUP + "out = program.reference(graph=graph, src={'x': x}, dst={})", runnable=True)
add("messagepassingprepare", [GRAPH, SRC, DST, EDGE, NDATA, PARAMS], "Callable[[], Tensor]", "# CUDA scalar CSR only; graph and native fields must already be validated.\nprepared = program.prepare(graph=graph, src={'x': x}, dst={})\nout = prepared()")
for anchor, expression, result in (
    ("messagepassingexplain", "program.explain()", "str"),
    ("messagepassingdiagnostics", "program.diagnostics", "tuple[AnalysisFinding, ...]"),
    ("messagepassingschedules", "program.schedules", "tuple[MachineSchedule, ...]"),
    ("messagepassingcache_info", "program.cache_info", "dict[str, int]"),
    ("messagepassinglast_variant", "program.last_variant", "CompiledVariant (variants: tuple[CompiledVariant, ...])"),
):
    add(anchor, [], result, MP_SETUP + "out = program(graph=graph, src={'x': x}, dst={})\nprint(" + expression + ")", runnable=True)
add("messagepassingir", [p("stage", "str", "IR artifact stage; default domain. Missing artifacts raise KeyError.", "IR 产物阶段；默认 domain。缺失产物抛出 KeyError。")], "str", MP_SETUP + "out = program(graph=graph, src={'x': x}, dst={})\nprint(program.ir('domain'))", runnable=True)
add("messagepassingcode", [p("kind", "str", "Generated artifact kind, default ptx; requires a matching compiled execution.", "生成代码类型，默认 ptx；要求已执行对应编译路径。")], "str", "# After a CUDA-compiled program call:\nprint(program.code('ptx'))")
add("gfsum", [p("identity", "int | float", "Additive identity; default 0.", "加法幺元；默认 0。"), DETERMINISTIC], "SumReducer", "import tiga as tg\nreducer = tg.sum()", runnable=True)
for anchor, name, result in (("gfmean", "mean", "MeanReducer"), ("gfprod", "prod", "ProductReducer")):
    add(anchor, [DETERMINISTIC], result, f"import tiga as tg\nreducer = tg.{name}()", runnable=True)
add("gfonline_softmax", [p("accumulation_dtype", "tg.DType | torch.dtype | None", "Accumulation precision hint; None uses the provider default.", "累加精度提示；None 使用 provider 默认值。"), DETERMINISTIC, p("block_prune_threshold", "float | None", "None is exact; (0,1] explicitly enables approximate tile pruning on supported CUDA forward paths.", "None 为精确计算；(0,1] 显式启用受支持 CUDA 前向的近似 tile 剪枝。")], "OnlineSoftmaxReducer", "import tiga as tg\nreducer = tg.online_softmax()", runnable=True)
add("gfreducer", [DETERMINISTIC], "tg.Reducer subclass instance", "import tiga as tg\nclass Add(tg.Reducer):\n    associative = True\n    commutative = True\n    def identity(self): return 0.\n    def combine(self, left, right): return left + right\nreducer = Add()", runnable=True)
add("reducer-call", [p("*messages", "Tensor expressions", "One or more edge-local inputs for lift; this packages messages, not an immediate reduction.", "交给 lift 的一个或多个边局部输入；这里只打包消息，不立即归约。")], "ReducerCall | OnlineSoftmaxItem", "import tiga as tg\nclass Attention(tg.MessagePassing):\n    reducer = tg.online_softmax()\n    def edge(self, src, dst, edge):\n        return self.reducer(edge.score, src.value)", runnable=True)
add("reducermlir", [p("message_dtypes", "Sequence[tg.DType] | None", "One native dtype per message; None uses one float32 message.", "每个消息一个原生 dtype；None 表示单个 float32 消息。"), p("symbol", "str | None", "IR symbol override; None uses the reducer name.", "IR 符号覆盖值；None 使用 reducer 名称。")], "str", "import tiga as tg\nprint(tg.sum().mlir(message_dtypes=(tg.float32,)))", runnable=True)
add("gfnntrace", [p("module", "torch.nn.Module", "Supported traceable module; original parameters retain Torch autograd connections.", "可 trace 的受支持模块；原始参数保留 Torch autograd 连接。"), p("block_e", "int | None", "Optional tile size, power of two in [16,1024].", "可选 tile 大小，[16,1024] 内的 2 的幂。"), p("num_warps", "int | None", "Optional provider launch warp count; None uses lowering defaults.", "可选 provider 启动 warp 数；None 使用 lowering 默认值。")], "TracedModule", "import torch\nimport tiga as tg\nmodule = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh())\ntraced = tg.nn.trace(module)\ny = traced(torch.ones(2, 3))\ny.sum().backward()", runnable=True)


STATE = p("initial", "tg.Tensor | Sequence[tg.Tensor]", "Loop-carried native values; their shapes, dtypes and devices must remain unchanged.", "循环携带的原生值；shape、dtype 和设备须保持不变。")
BODY = p("body", "Callable[..., tg.Tensor | Sequence[tg.Tensor]]", "Captured once; returns one value per carried state.", "捕获一次；每个携带状态返回一个值。")
ITERATIONS = p("iterations", "int", "Non-negative fixed iteration count; bool is rejected.", "非负固定迭代次数；不接受 bool。")
BOUND = p("max_iterations", "int", "Non-negative finite loop bound, including zero.", "非负有限循环上界，允许零。")
CONDITION = p("condition", "Callable[..., tg.Tensor]", "Captured condition returning a scalar boolean Tensor.", "捕获的条件函数，返回标量布尔 Tensor。")
add("gfjit", [p("function", "Callable | None", "Decorated function; source must be available for AST capture.", "被装饰函数；AST 捕获要求能读取源码。"), p("max_iterations", "int | None", "Finite bound for while loops; None is allowed when no while loop is captured.", "while 的有限上界；无 while 时允许 None。")], "Callable", "import tiga as tg\n@tg.jit\ndef decay(x):\n    for i in range(3):\n        x = x * 0.5\n    return x\nprint(decay(tg.tensor([8.])).tolist())")
add("gfrepeat", [STATE, BODY, ITERATIONS], "tg.Tensor | tuple[tg.Tensor, ...]", "import tiga as tg\ny = tg.repeat(tg.tensor([8.]), lambda x: x * 0.5, iterations=3)\nassert y.tolist() == [1.]", runnable=True)
add("gfwhile_loop", [STATE, CONDITION, BODY, BOUND], "tg.Tensor | tuple[tg.Tensor, ...]", "import tiga as tg\ny = tg.while_loop(tg.tensor(1.), lambda x: x < 4., lambda x: x + 1., max_iterations=8)\nassert y.tolist() == 4.", runnable=True)
add("gfcontrolrepeat", [STATE, ITERATIONS], "tg.Tensor | tuple[tg.Tensor, ...]", "import tiga as tg\nclass Decay(tg.control.Repeat):\n    def body(self, x): return x * 0.5\nassert Decay()(tg.tensor([8.]), iterations=3).tolist() == [1.]", runnable=True)
add("gfcontrolwhile", [STATE, BOUND], "tg.Tensor | tuple[tg.Tensor, ...]", "import tiga as tg\nclass Grow(tg.control.While):\n    def condition(self, x): return x < 4.\n    def body(self, x): return x + 1.\nassert Grow()(tg.tensor(1.), max_iterations=8).tolist() == 4.", runnable=True)

MESH = p("mesh", "tg.DeviceMesh", "Logical device topology, not a running communicator.", "逻辑设备拓扑，不是运行中的 communicator。")
PARTITION = p("partition", "tg.ByDestination", "Destination ownership policy.", "目标实体的所有权策略。")
DEPTH = p("halo_depth", "int | Literal['auto']", "Non-negative depth or compiler inference.", "非负深度或由编译器推断。")
WORLD = p("world_size", "int", "Positive rank count.", "正的 rank 数。")
RANK = p("rank", "int", "Local rank in [0, world_size).", "本地 rank，范围 [0, world_size)。")
ENTITIES = p("num_entities", "int", "Global entity count for the homogeneous CSR relation.", "同构 CSR 关系的全局实体数。")
HALO_ROWS = p("row_ptr / col_idx", "Sequence[int]", "Global CSR row boundaries and source IDs as host integer sequences.", "全局 CSR 行边界与源编号，使用 host 整数序列。")
HALO_SETUP = "import tiga as tg\nhalos = tg.collective_halo_maps([0, 1, 2], [1, 0], num_entities=2, world_size=2)\n"
TRANSPORT = p("transport", "NeighborTransport", "Bound transport implementing rank, world_size and exchange; must match the halo topology.", "实现 rank、world_size 与 exchange 的 transport；须与 halo 拓扑匹配。")
PROGRESS = p("progress_threads", "int", "Positive number of background progress workers; default 1.", "后台 progress worker 正整数数量；默认 1。")
add("gfdevicemesh", [p("device_type", "str", "Logical device family such as cpu or cuda; does not initialize devices.", "逻辑设备族，如 cpu 或 cuda；不会初始化设备。"), p("shape", "int | tuple[int, ...]", "Positive mesh extents; product is rank count.", "正的网格尺寸；乘积为 rank 数。"), p("names", "tuple[str, ...]", "Optional axis names, one per mesh dimension.", "可选网格轴名，每个维度一个。")], "tg.DeviceMesh", "import tiga as tg\nmesh = tg.DeviceMesh('cpu', (2,), names=('workers',))", runnable=True)
add("gfbydestination", [p("mesh_axis", "int | str", "Mesh axis index or name.", "网格轴编号或名称。"), p("balance", "Literal['auto', 'edges', 'entities']", "Partition balancing objective.", "分区负载平衡目标。")], "tg.ByDestination", "import tiga as tg\npartition = tg.ByDestination(balance='edges')", runnable=True)
add("gfgraphplacement", [MESH, PARTITION, DEPTH], "tg.GraphPlacement", "import tiga as tg\nplacement = tg.GraphPlacement(tg.DeviceMesh('cpu', 2), tg.ByDestination(), 'auto')", runnable=True)
add("gfhalomap", [RANK, WORLD, p("owned_begin / owned_end", "int", "Half-open interval of globally numbered owned entities.", "拥有实体的全局编号左闭右开区间。"), p("ghost_ids", "tuple[int, ...]", "Sorted external source IDs required by owned rows.", "拥有行需要的外部源编号，按序排列。"), p("receive_from / send_to", "tuple[tuple[int, tuple[int, ...]], ...]", "Peer rank and ordered global entity IDs.", "对端 rank 与有序全局实体编号。"), p("bytes_for: itemsize / trailing_elements", "int", "Positive bytes per scalar and scalars per entity (default 1).", "每标量字节数、每实体标量数（默认 1），均为正数。")], "tg.HaloMap; bytes_for → int", HALO_SETUP + "print(halos[0].ghost_ids, halos[0].bytes_for(4))", runnable=True)
add("gfderive_halo_map", [HALO_ROWS, ENTITIES, WORLD, RANK, p("peer_requests", "Sequence[Sequence[int]] | None", "IDs requested by each rank; None leaves send_to empty.", "每个 rank 请求的编号；None 使 send_to 为空。")], "tg.HaloMap", "import tiga as tg\nhalo = tg.derive_halo_map([0, 1, 2], [1, 0], num_entities=2, world_size=2, rank=0)", runnable=True)
add("gfcollective_halo_maps", [HALO_ROWS, ENTITIES, WORLD], "tuple[tg.HaloMap, ...]", HALO_SETUP, runnable=True)
add("gfexchange_halo", [p("halo", "tg.HaloMap", "Per-rank ownership and communication map.", "本 rank 的所有权与通信映射。"), p("owned_data", "bytes | bytearray | memoryview", "Packed owned entities in global-ID order, starting at owned_begin.", "从 owned_begin 开始、按全局编号排列的拥有实体字节。"), p("element_bytes", "int", "Positive total bytes per entity, including feature dimensions.", "每实体总字节数，包含特征维，必须为正。"), TRANSPORT], "HaloBuffer", "# Inside an initialized rank; peers must execute the matching exchange.\nreceived = tg.exchange_halo(halo, owned_data, element_bytes=4, transport=transport)")
add("distributedruntime", [TRANSPORT, PROGRESS], "tg.DistributedRuntime", "# transport is a deployment-bound NeighborTransport.\nimport tiga as tg\nwith tg.DistributedRuntime(transport) as runtime:\n    output = program(graph=placed_graph, src={'x': local_x}, dst={})")
add("distributedruntimefrom_provider", [p("name", "str", "Registered transport: tcp, mpi, nccl, or an installed plugin.", "已注册的 tcp、mpi、nccl 或已安装插件。"), PROGRESS, p("**options", "provider-specific keyword arguments", "TCP: rank, world_size, host, port, bind, timeout. MPI: communicator, tag. NCCL requires a communicator/deployment setup; see the distributed guide.", "TCP：rank、world_size、host、port、bind、timeout。MPI：communicator、tag。NCCL 需要 communicator/部署配置，见分布式指南。")], "tg.DistributedRuntime", "# Run collectively with an MPI launcher and the mpi extra installed.\nimport tiga as tg\nwith tg.DistributedRuntime.from_provider('mpi') as runtime:\n    print(runtime.transport.rank)")

CONTRACTS["distributedruntimefrom_provider"]["params"] += [
    p("tcp/nccl: rank / world_size", "int", "Required rank and positive group size; 0 <= rank < world_size.", "必填 rank 与正的组规模；0 <= rank < world_size。"),
    p("tcp: port", "int", "Required rendezvous port; peers must agree and peer listener ports must be reachable.", "必填 rendezvous 端口；所有对端须一致且 peer 监听端口可达。"),
    p("tcp: host / bind", "str | None / str", "host selects a rendezvous peer; None hosts it. bind defaults to the empty listening address.", "host 指定 rendezvous 对端；None 在本地监听。bind 默认为空监听地址。"),
    p("tcp: timeout", "float", "Finite positive seconds for connection and data waits; default 30.", "连接与数据等待的有限正秒数；默认 30。"),
    p("mpi: communicator", "mpi4py.MPI.Comm | None", "None uses MPI.COMM_WORLD; mpi4py and an MPI runtime are required.", "None 使用 MPI.COMM_WORLD；需要 mpi4py 与 MPI runtime。"),
    p("mpi: tag", "int", "Message tag on the supplied communicator; default 0.", "指定 communicator 上的消息 tag；默认 0。"),
    p("nccl: communicator_id", "bytes | None", "Shared 128-byte NCCL unique ID, distributed by the launcher; None is only valid for a one-rank group.", "launcher 分发的共享 128 字节 NCCL unique ID；None 仅适用于单 rank。"),
    p("nccl: device", "str | tg.Device", "Local CUDA device, default cuda:0; not a global rank identifier.", "本地 CUDA 设备，默认 cuda:0；不是全局 rank 编号。"),
    p("nccl: library", "str | os.PathLike | None", "Explicit libnccl path or automatic discovery.", "显式 libnccl 路径，或自动查找。"),
]
add("gfruntimeautoffload", [p("ram", "int", "Non-negative byte threshold for graph CSR offload; use tg.execution for IEC strings.", "图 CSR 自动卸载的非负字节阈值；IEC 字符串使用 tg.execution。")], "context manager", "import tiga as tg\nwith tg.runtime.auto_offload(1024 ** 3):\n    graph = tg.Graph.regular(4, 2)", runnable=True)
add("tgexecution", [DEVICE, p("memory", "Mapping[str, int | str] | None", "Tier budgets in bytes or IEC sizes; omitted tiers are unbounded, not reserved.", "各层字节数或 IEC 大小预算；省略表示不设限，不预留物理内存。"), p("spill_dir", "str | os.PathLike | None", "Temporary spill directory; required for explicit LRU policy.", "临时 spill 目录；显式 LRU 策略需要提供。"), p("page_rows", "int", "Positive destination rows per CPU CSR page; default 100000.", "CPU CSR 每页目标行数，正整数；默认 100000。"), p("prefetch_depth", "int", "Positive in-flight page count; default 2.", "预取页数，正整数；默认 2。"), p("eviction", "Literal['error', 'lru']", "Reject excess allocation or evict idle whole native Tensors.", "超预算时报错，或淘汰空闲的完整原生 Tensor。")], "ExecutionScope (context manager)", "import tempfile\nimport tiga as tg\nwith tempfile.TemporaryDirectory() as directory:\n    with tg.execution(memory={'ram': '1 MiB'}, spill_dir=directory) as run:\n        x = tg.tensor([1., 2.])\n        print(run.memory_report())", runnable=True)
add("tensorspill", [], NATIVE, NATIVE_SETUP + "x.spill()\nassert not x.residency['resident']\nassert x.tolist() == [[1., 2.], [3., 4.]]", runnable=True)
add("tensorto", [p("device", "str | tg.Device", "Destination device; same-device calls realize and return self.", "目标设备；同设备调用会 realize 并返回自身。")], NATIVE, NATIVE_SETUP + "host = x.to('cpu')", runnable=True)
add("tensorcpu", [], NATIVE, NATIVE_SETUP + "host = x.cpu()", runnable=True)
add("tensorresidency", [], "dict[str, object]", NATIVE_SETUP + "print(x.residency)", runnable=True)
STORAGE_EXAMPLE = NATIVE_SETUP + "import tempfile\nfrom pathlib import Path\nwith tempfile.TemporaryDirectory() as directory:\n    path = Path(directory) / 'value.tga'\n    x.save(path)\n    restored = tg.load(path)\n    assert restored.tolist() == x.tolist()"
add("tensorsave", [PATH, OVERWRITE], "pathlib.Path (method); None (tg.save)", STORAGE_EXAMPLE, runnable=True)
add("tgload", [PATH, DEVICE], "tg.Tensor | tg.Graph", STORAGE_EXAMPLE, runnable=True)
add("tensordisk", [p("name", "str | None", "None spills temporarily; a safe basename persists to the legacy named store.", "None 临时 spill；合法 basename 保存至旧式命名存储。")], NATIVE, NATIVE_SETUP + "x.disk()\nprint(x.tolist())", runnable=True)
add("gffrom_disk", [p("name", "str", "Existing snapshot basename; no dots at the start or path separators.", "已有快照 basename；不能以点开头或含路径分隔符。")], NATIVE, "import tiga as tg\n# Requires an existing named snapshot; prefer explicit tg.save/tg.load paths.\nx = tg.from_disk('existing-snapshot')")

PROGRAM_SETUP = MP_SETUP + "composition = tg.GraphProgram()\nvalue = composition.apply(program, graph=graph, src={'x': x}, dst={})\n"
add("gfprogram", [p("function", "Callable", "Straight-line Python function containing MessagePassing calls.", "包含 MessagePassing 调用的直线 Python 函数。")], "Callable", MP_SETUP + "@tg.program\ndef composed(x):\n    return program(graph=graph, src={'x': x}, dst={})\nvalue = composed(x)\nprint(value.materialize())")
add("graphprogramapply", [p("kernel", "tg.MessagePassing", "Leaf program to capture.", "待捕获的叶程序。"), GRAPH, SRC, DST, EDGE, PARAMS], "ProgramValue", PROGRAM_SETUP)
add("graphprogramoutputs", [p("*values", "ProgramValue | tg.Tensor", "Outputs belonging to this composition; foreign program values are rejected.", "属于此组合的输出；不接受其他程序的值。")], "GraphProgram (self)", PROGRAM_SETUP + "composition.outputs(value)")
for anchor, call, result in (("graphprogramrun", "composition.run()", "Tensor | tuple[Tensor, ...]"), ("graphprogramexplain", "composition.explain()", "str (semantic_hash: str)"), ("programvaluematerialize", "value.materialize()", "torch.Tensor | tg.Tensor")):
    add(anchor, [], result, PROGRAM_SETUP + f"result = {call}")
add("graphprogramir", [p("stage", "str", "domain, fused, iteration/iter, or kernel.", "domain、fused、iteration/iter 或 kernel。")], "str", PROGRAM_SETUP + "print(composition.ir('domain'))")
add("graphprogramcode", [p("kind", "str", "Provider artifact such as ptx or ttir; unavailable artifacts raise KeyError.", "如 ptx 或 ttir 的 provider 产物；不存在时抛出 KeyError。")], "str | bytes | dict", PROGRAM_SETUP + "# Requires a matching provider; CPU paths do not produce PTX.\nprint(composition.code('ptx'))")

COLOR = "tuple[float, float, float]"
CMAP = p("cmap", "str | Sequence[RGB] | Sequence[tuple[float, RGB]] | None", "Named map or color stops in [0,1]; explicit cmap overrides low/high.", "内置色图名或 [0,1] 内的颜色节点；显式 cmap 覆盖 low/high。")
LIMITS = [p("vmin / vmax", "float | None", "Scalar color range; None derives bounds for geometry renderers. heatmap defaults to 0 and 1.", "标量颜色范围；几何渲染器的 None 从数据推断。heatmap 默认 0 和 1。"), p("low / high", COLOR + " | None", "Optional RGB endpoints in [0,1] for a two-color ramp.", "可选的双色渐变 RGB 端点，各通道在 [0,1] 内。"), CMAP]
CAMERA = p("camera", "Camera | Literal['auto'] | tuple[float, float]", "Explicit camera, automatic framing, or elevation/azimuth in degrees.", "显式相机、自动取景或角度制 elevation/azimuth。")
SIZE = p("width / height", "int", "Positive image dimensions in pixels; default 512 each.", "正的图像像素宽高，默认各 512。")
XYZ = p("positions", "torch.Tensor | tg.Tensor | numpy.ndarray | Sequence", "Coordinates (N,2) or (N,3), copied to host for geometry rendering/export.", "(N,2) 或 (N,3) 坐标，几何渲染/导出会复制到 host。")
VALUES = p("values", "torch.Tensor | tg.Tensor | numpy.ndarray | Sequence | None", "One scalar per vertex; None uses the renderer's uniform/density default.", "每顶点一个标量；None 使用渲染器的均匀值/密度默认值。")
FACES = p("faces", "numpy.ndarray | Sequence[Sequence[int]]", "Triangle vertex IDs (M,3), all in range.", "(M,3) 三角形顶点编号，均须在有效范围内。")
WIRE = p("wireframe", "bool", "Draw edges instead of filled triangles; default False.", "是否仅绘制边框；默认 False。")
BACKGROUND = p("background", COLOR, "Background RGB in [0,1]; default black.", "[0,1] 内的背景 RGB；默认黑色。")
VIZ_SETUP = "import tiga as tg\npositions = [[0., 0.], [1., 0.], [0., 1.]]\n"
add("gfvisualizeheatmap", [p("values", "tg.Tensor", "Contiguous rank-two native floating Tensor. Torch input currently requires tg.from_torch; that conversion does not bridge autograd.", "连续 rank-2 原生浮点 Tensor。Torch 输入当前须经 tg.from_torch；此转换不连接 autograd。"), *LIMITS], "tg.visualize.Raster", "import tiga as tg\nraster = tg.visualize.heatmap(tg.tensor([[0., 0.5], [0.75, 1.]]))\nassert raster.to_numpy().shape == (2, 2, 3)", runnable=True)
add("gfvisualizecolormaps", [], "tuple[str, ...]", "import tiga as tg\nprint(tg.visualize.colormaps())", runnable=True)
add("raster", [p("pixels", "tg.Tensor", "Lazy RGB pixel values; prefer a rendering factory over manual construction.", "惰性 RGB 像素值；优先使用渲染工厂而非手动构造。"), SIZE, p("to_numpy: clip", "bool", "Clamp exported RGB to [0,1]; default True.", "是否将导出 RGB 裁剪至 [0,1]；默认 True。"), p("save: path", "str | os.PathLike", "Image destination; Pillow encodes and existing files are replaced.", "图像目标；Pillow 编码，已有文件会被替换。"), p("mlir: verify", "bool", "Run the IR verifier; default False.", "是否执行 IR verifier；默认 False。"), p("generated_code: kind", "str | None", "Artifact name; inherits Tensor.generated_code behavior.", "产物名称；与 Tensor.generated_code 一致。"), p("show: **imshow_options", "object", "Keyword arguments forwarded to Matplotlib imshow.", "转交 Matplotlib imshow 的关键字参数。")], "Raster; to_numpy → numpy.ndarray; save → pathlib.Path; show → None", "import tiga as tg\nraster = tg.visualize.heatmap(tg.tensor([[0., 1.]]))\nprint(raster.realize().execution)\nprint(raster.to_numpy())", runnable=True)
add("gfvisualizecamera", [p("position / target / up", COLOR, "World-space camera position, look-at target and up vector; up defaults to (0,0,1).", "世界坐标下相机位置、观察目标与上方向；up 默认 (0,0,1)。"), p("fov", "float", "Vertical field of view in degrees; default 45.", "角度制垂直视场角；默认 45。"), p("auto / from_angles: positions", "array-like | None", "Cloud used for framing; auto requires positions.", "用于取景的点云；auto 必填。"), p("auto: margin", "float", "Bounding-sphere distance multiplier; default 1.2.", "包围球距离倍率；默认 1.2。"), p("from_angles: elevation / azimuth", "float", "Camera angles in degrees; defaults 30 and -60.", "角度制相机角；默认 30 和 -60。"), p("from_angles: distance", "float | None", "Camera-to-target distance; None fits positions or uses the default.", "相机到目标的距离；None 按 positions 或默认值计算。"), p("world_to_ndc: points_xyz", "numpy.ndarray", "World-space coordinates (N,3).", "世界坐标 (N,3)。")], "Camera; world_to_ndc → tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]", VIZ_SETUP + "camera = tg.visualize.Camera.auto(positions)", runnable=True)
add("gfvisualizeparticles", [XYZ, VALUES, CAMERA, SIZE, p("point_radius", "float", "Disk radius in pixels; default 2.", "圆盘像素半径；默认 2。"), *LIMITS], "Raster", VIZ_SETUP + "raster = tg.visualize.particles(positions, width=32, height=32)", runnable=True)
add("gfvisualizedelaunay", [XYZ, VALUES, CAMERA, SIZE, *LIMITS, WIRE], "Raster", VIZ_SETUP + "# Requires Matplotlib for triangulation.\nraster = tg.visualize.delaunay(positions, width=32, height=32)")
add("gfvisualizemesh", [XYZ, FACES, VALUES, CAMERA, SIZE, WIRE, *LIMITS], "Raster", VIZ_SETUP + "raster = tg.visualize.mesh(positions, [[0, 1, 2]], width=32, height=32)", runnable=True)
add("tgvisualizegaussians", [p("positions", "array-like", "3-D means (N,3); geometry is rendered on CPU, not differentiable.", "三维均值 (N,3)；几何部分在 CPU 渲染，不可微。"), p("colors", "numpy.ndarray | Sequence", "RGB (N,3), values in [0,1].", "RGB (N,3)，值在 [0,1] 内。"), p("scales", "numpy.ndarray | Sequence", "Positive standard deviations (N,) or (N,3).", "正的标准差 (N,) 或 (N,3)。"), p("rotations", "numpy.ndarray | Sequence | None", "Unit quaternions (N,4), ordered w,x,y,z; None is identity.", "单位四元数 (N,4)，顺序 w,x,y,z；None 为单位旋转。"), p("opacities", "numpy.ndarray | Sequence | None", "Opacity (N,) in [0,1]; None means opaque.", "[0,1] 内的不透明度 (N,)；None 为完全不透明。"), CAMERA, SIZE, p("cutoff", "float", "Ellipse extent in standard deviations; default 3.", "以标准差计的椭圆截断半径；默认 3。"), BACKGROUND], "Raster", "import tiga as tg\nraster = tg.visualize.gaussians([[0., 0., 0.]], [[1., 0., 0.]], [0.1], width=32, height=32)", runnable=True)
add("gfvisualizevolume", [p("density", "torch.Tensor | tg.Tensor | numpy.ndarray", "3-D density grid; host ray marching is not differentiable.", "三维密度网格；host ray marching 不可微。"), CAMERA, SIZE, p("steps", "int", "Positive sample count per ray; default 128.", "每条射线采样次数，正整数；默认 128。"), CMAP, LIMITS[0], p("scale", "float", "Absorption multiplier; default 8.", "吸收倍率；默认 8。"), BACKGROUND], "Raster", "import numpy as np\nimport tiga as tg\nraster = tg.visualize.volume(np.ones((3, 3, 3)), width=8, height=8, steps=8)", runnable=True)
PLY_EXAMPLE = VIZ_SETUP + "import tempfile\nfrom pathlib import Path\nwith tempfile.TemporaryDirectory() as directory:\n    path = Path(directory) / 'triangle.ply'\n    tg.visualize.export_ply(path, positions, faces=[[0, 1, 2]])\n    points, faces, extras = tg.visualize.load_ply(path)"
OBJ_EXAMPLE = PLY_EXAMPLE.replace(".ply'", ".obj'").replace("export_ply", "export_obj").replace("load_ply", "load_obj").replace("points, faces, extras", "points, faces")
add("gfvisualizeloadply", [PATH], "tuple[numpy.ndarray, numpy.ndarray | None, dict[str, numpy.ndarray]]", PLY_EXAMPLE, runnable=True)
add("gfvisualizeloadobj", [PATH], "tuple[numpy.ndarray, numpy.ndarray]", OBJ_EXAMPLE, runnable=True)
add("gfvisualizeexportply", [PATH, XYZ, p("faces", "array-like | None", "Optional triangle vertex indices (M,3).", "可选三角形顶点编号 (M,3)。"), VALUES, LIMITS[1], CMAP], "pathlib.Path", PLY_EXAMPLE, runnable=True)
add("gfvisualizeexportobj", [PATH, XYZ, FACES], "pathlib.Path", OBJ_EXAMPLE, runnable=True)
add("gfvisualizeexportvdb", [PATH, XYZ, VALUES, p("voxel_size", "float", "Positive world-space voxel width; default 0.05.", "世界坐标体素边长，正数；默认 0.05。")], "pathlib.Path", VIZ_SETUP + "# Requires pyopenvdb installed separately.\ntg.visualize.export_vdb('cloud.vdb', positions, [1., 2., 3.])")
add("gfvisualizesavevideo", [p("frames", "Iterable[Raster | numpy.ndarray]", "Same-size RGB frames, arrays shaped (H,W,3) with values in [0,1].", "尺寸一致的 RGB 帧；数组为 (H,W,3)，值在 [0,1] 内。"), PATH, p("fps", "float", "Finite positive playback frame rate; bool and non-finite values are rejected. Default 30.", "有限的正播放帧率；拒绝 bool 与非有限值，默认 30。")], "pathlib.Path", "import tempfile\nfrom pathlib import Path\nimport tiga as tg\nframe = tg.visualize.heatmap(tg.tensor([[0., 1.]]))\nwith tempfile.TemporaryDirectory() as directory:\n    tg.visualize.save_video([frame, frame], Path(directory) / 'demo.gif', fps=10)", runnable=True)
add("gfcompiler-symbolic-shapes", [p("Dim: name", "str", "Symbol identifier shared across argument shapes.", "跨参数 shape 共享的符号标识符。"), p("Dim: minimum / maximum / multiple_of", "int / int | None / int", "Inclusive bounds and positive divisibility constraint; defaults 1, None, 1.", "含端点上下界与正的整除约束；默认 1、None、1。"), p("TensorSpec: shape", "Sequence[int | Dim]", "Concrete non-negative extents or constrained dimensions.", "非负具体尺寸或受约束符号维。"), DTYPE, DEVICE, p("ShapeSpecializer: specs", "Sequence[TensorSpec]", "One specification per positional input.", "每个位置输入一个规格。"), p("ShapeSpecializer: compiler", "Callable[[ShapeBinding], Callable]", "Build a callable for a validated concrete shape binding.", "为验证通过的具体 shape 绑定构建可调用对象。"), p("ShapeSpecializer.__call__: *values", "tg.Tensor", "Inputs checked before specialization lookup/compilation.", "在查找/编译特化前验证的输入。")], "Dim / TensorSpec / ShapeSpecializer; calling specializer → compiler-defined result", "import tiga as tg\nn = tg.compiler.Dim('N', maximum=16)\nspec = tg.compiler.TensorSpec((n,))\nrun = tg.compiler.ShapeSpecializer([spec], lambda binding: lambda x: x * 2)\nassert run(tg.tensor([1., 2.])).tolist() == [2., 4.]", runnable=True)


def render(contract, lang):
    zh = lang == "zh"
    lines = [START, "", "| 参数 | 类型 | 含义 |" if zh else "| Parameter | Type | Meaning |", "|---|---|---|"]
    for name, kind, en, cn in contract["params"]:
        escaped_kind = html.escape(kind).replace('|', '&#124;')
        lines.append(f"| `{name}` | <code>{escaped_kind}</code> | {cn if zh else en} |")
    if not contract["params"]:
        lines = [START, "", "无显式参数（实例方法的 `self` 省略）。" if zh else "No explicit parameters (`self` omitted for instance methods)."]
    lines += ["", ("返回类型：" if zh else "Return type: ") + f"`{contract['result']}`" + ("。" if zh else ".")]
    if not contract["runnable"]:
        lines += ["", "运行示例前，先按下文配置所需设备、文件或分布式环境。" if zh else "Before running this example, configure the required device, files or distributed environment as described below."]
    lines += ["", '??? example "' + ("最小用法" if zh else "Minimal usage") + '"', "", "    ```python"]
    lines += ["    " + line for line in contract["example"].splitlines()]
    lines += ["    ```", "", END, ""]
    return "\n".join(lines)


def update(source, lang):
    source = re.sub(re.escape(START) + r".*?" + re.escape(END) + r"\n*", "", source, flags=re.S)
    pattern = re.compile(r"^(### .*\{ #([^ }]+) \}[^\n]*\n)\n*", re.M)
    return pattern.sub(lambda m: m[1] + "\n" + (render(CONTRACTS[m[2]], lang) + "\n" if m[2] in CONTRACTS else ""), source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for lang, suffix in (("en", ""), ("zh", ".zh")):
        path = ROOT / f"docs/api{suffix}.md"
        source = path.read_text()
        result = update(source, lang)
        if args.check:
            if result != source:
                raise SystemExit(f"stale API contracts: {path}")
        else:
            path.write_text(result)
    print(f"{len(CONTRACTS)} typed API contracts checked" if args.check else f"rendered {len(CONTRACTS)} typed API contracts")


if __name__ == "__main__":
    main()
