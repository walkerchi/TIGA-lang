# 排错与问题反馈

Tiga 当前为 alpha。存在某个构造器，不代表所有 dtype、layout、梯度、设备和
provider 组合均受支持。参见[支持矩阵](roadmap.md)、[执行诊断](execution.md)
与[带类型的 API 参考](api.md)。

## 安装 { #installation }

官方发布目标：[GitHub](https://github.com/walkerchi/TIGA-lang) 代码与 issue、
PyPI 的 `tiga-lang` 包，以及后续 arXiv tech report。首次 PyPI 发布与 arXiv
编号尚未完成。当前文档域名为登录保护的预览站点，不是公开文档服务。

匹配的 wheel 无需另装 LLVM SDK。本地验证覆盖 Linux x86-64、CPython 3.11/3.12、
glibc 2.38 及以上；其他 Python 版本和平台不在首发范围内。
可选 adapter 面向 Torch 2.11.x 与 Triton 3.6.x，基础 wheel 仍不依赖 Torch。不要改 wheel
文件名以绕过 pip 的兼容性检查。

| 现象 | 处理方式 |
|---|---|
| `No matching distribution found` | 检查发布状态、Python/OS/架构与 wheel 标签；使用匹配的本地 wheel 或源码构建，不要安装无关的 `tiga` 包。 |
| pip 开始 CMake 构建 | 未选中兼容 wheel。用 `--only-binary=:all:` 强制 wheel，或明确准备固定版本工具链。 |
| 缺少 Torch | Torch 示例另装 Torch；无 Torch 可用执行指南中的原生 CPU 示例。 |
| 找不到原生工具 | 运行 `python -m tiga`，检查解释器及过时的 `TIGA_OPT` / `TIGA_TRANSLATE` 覆盖值；wheel 自带这些工具。 |
| CUDA 失败 | 检查驱动/设备和 `cuda` extra；安装 Tiga 不会安装驱动或选择匹配的 Torch。 |
| 缺少可视化依赖 | 安装 `tiga-lang[visualization]`；MP4 另需外部 `ffmpeg`，OpenVDB 导出另需 `pyopenvdb`。 |

## 输入与执行中的常见误用 { #input-contracts }

| 情形 | 契约与检查方向 |
|---|---|
| CSR 边方向反了 | 第 i 行为目标 i，`col_idx` 保存源编号，参见[连接关系示意图](programming-model.md)。 |
| 空行/孤立节点 | 重复行边界表示空目标行。传 `num_src` 保留孤立源节点；检查 reducer 的空行语义。 |
| 重复 COO/CSR 边 | 每条边是独立消息，不隐式去重。COO 导入按目标排序，边字段须匹配转换后的顺序。 |
| 外部或手工 CSR | 用 `validate="full"` 验证行边界与索引范围，`basic` 不是完整验证。 |
| 二部图 | 分别绑定 `src`、`dst`，首维为 `num_src`、`num_dst`；不要用同构图 `ndata` 简写。 |
| 设备或 layout 不一致 | 字段、索引与图同设备；遵守每个算子的 dtype/layout 约束。存储 dtype 支持不等于 kernel 支持。 |
| 改变拓扑 | 重新构图，不原地修改已捕获的索引 buffer。transpose/cat 改变边顺序，旧边字段需重排。 |
| 缺少梯度 | 调用前设置 `requires_grad`，使用对应 autograd。`tg.from_torch` 共享存储，但不连接 Torch 与原生求导历史。 |
| 动态邻居/剪枝 | 梯度针对已选定的固定拓扑，离散选邻居不可微。近似 tile 剪枝注意力仅支持前向。 |
| Prepared launch | `prepare()` 受执行路径限制并绑定固定存储，并非所有 backend 都支持；不能与自动完整 Tensor LRU 淘汰混用。 |
| RAM 空闲却 `MemoryError` | 可能超过作用域分配预算；报告不统计 Torch 分配或整个进程 RSS。 |
| 多 rank 卡住/超时 | 启动全部 rank 且 collective 顺序一致；检查地址与有限 timeout，包含冷 JIT 耗时。DeviceMesh 不会启动 worker。 |
| GPU/线程初始化后启动多进程 | 像分布式示例一样使用 `spawn` context 与 `if __name__ == "__main__":` 入口保护，不要 fork 已初始化的 CUDA runtime；先启动全部 rank 再等待结果。 |

在小输入上用 `program.reference(...)` 与明确阈值的 `torch.testing.assert_close`
对比。前向一致不代表梯度正确、已经编译或按位确定性。高阶梯度和并发修改
不是普遍支持的契约。

## 反馈 bug { #bug-report }

普通 bug 与文档建议提交至官方仓库公开后的
[GitHub Issues](https://github.com/walkerchi/TIGA-lang/issues/new/choose)。
唯一维护者：**walkerchi**，Independent Developer。支持邮箱：
[walker.chi.000@gmail.com](mailto:walker.chi.000@gmail.com)，邮件主题前缀
`[Tiga bug]`。issue 尚不可用时可通过邮件发送报告，内容包括：

1. 期望行为、实际行为与完整异常。
2. 独立、固定随机种子、合成数据、不依赖私有资源的最小复现。
3. 环境、安装命令、源码 commit 或 wheel 文件名；相关时附 GPU/驱动。
4. 图类型、尺寸、dtype/device、前向/反向与 `program.explain()`；可用 IR 须移除私有路径/数据。
5. 梯度问题附 oracle、loss/cotangent、阈值与误差；性能问题按[测量协议](performance.md)提供匹配 baseline 和原始耗时，分开构图、编译、传输、热执行。

```bash
python -m tiga
python -m pip show tiga-lang torch triton
python -m pip check
```

`pip show` 提示缺少可选包不等于安装失败。修复应添加回归测试，同步中英文文档。

## 安全问题 { #security }

不要在公开 issue 上传漏洞或私有 crash dump。遵循
[SECURITY.md](https://github.com/walkerchi/TIGA-lang/blob/main/SECURITY.md)。
私密报告发送至 [walker.chi.000@gmail.com](mailto:walker.chi.000@gmail.com)，
邮件主题前缀 `[Tiga security]`，不承诺固定响应时间。
Tiga 不是运行不可信代码或文件的安全沙箱。

## 发布与论文 { #publication }

本地 `tiga-lang-paper` 项目保存 tech report 草稿，尚无 arXiv ID 或论文 DOI。
正式论文记录产生前，使用[软件引用元数据](https://github.com/walkerchi/TIGA-lang/blob/main/CITATION.cff)。

公开发布需要可访问的公开文档、有效的支持/安全渠道、版本一致的源码和 wheel、
干净环境测试、性能证据对应的源码版本以及带 tag 的产物。本地测试和历史图表不能
代替这些发布检查。
