# Tiga 开源发布审查交接

## 目标与边界

项目计划以 `tiga-lang` 开源发布，定位建议：

> Tiga is a differentiable JIT compiler for graph message-passing programs.

品牌使用 Tiga，仓库与 Python distribution 使用 `tiga-lang`，Python import 使用 `tiga`。本文件整理已有审查结论，供 Kimi Code 接续处理；文件创建不代表消息已发送至 Kimi Code，也不代表各项修改已完成。

审查时工作区已有大量未提交修改，包括 `python/graphforge` 向 `python/tiga` 的迁移。现有修改属于用户，应保留；禁止为清理工作区而 reset、checkout 或覆盖无关文件。实施前重新核对当前文件，避免依据过期行号修改。

遵循根目录 [AGENTS.md](AGENTS.md)，尤其是文档术语、语言和图示规则。优先完成本地可验证的修复；发布包、推送、修改远程仓库或部署站点需要依据后续明确授权执行。本文件不授权这些外部操作。

## 已确认问题及建议顺序

| 优先级 | 问题与证据 | 修改方向与验收标准 |
|---|---|---|
| P0 | `.github/workflows/compiler-ci.yml` 的文档构建环境只安装 `mkdocs-material`，但 `mkdocs.yml` 使用 `i18n` 插件。`pyproject.toml` 的 docs extra 已列出 `mkdocs-static-i18n`。 | 对齐文档依赖来源；在干净环境安装声明依赖后执行 `mkdocs build --strict`，避免依赖本机预装插件。 |
| P0 | compiler CI 仍传入 `GRAPHFORGE_LLVM_TEST_TOOLS_DIR`、`GRAPHFORGE_STRICT_LLVM_VERSION`、`GRAPHFORGE_BUNDLE_LLVM_RUNTIME`；CMake 已改读 `TIGA_*`。release workflow 也保留旧 strict 参数。 | 核对并修改 workflow/CMake 配置映射；确认测试工具目录实际生效，执行干净构建与 `check-tiga`。旧 strict/bundle 开关当前可能因默认 ON 而未暴露故障，不能据此认为旧参数有效。 |
| P0 | `docs/getting-started.md` clone `tiga-lang.git` 后使用 `cd tiga`。 | 修正目录，核对中文版本与 README；从空目录验证教程命令。 |
| P1 | `uv.lock` 根项目仍为 `graphforge-compiler`、版本 `0.1.0a1`，当前 metadata 已为 `tiga-lang`、`0.1.0`，且 extras 有变化。 | 按当前 metadata 更新锁文件，执行锁文件一致性检查。 |
| P1 | README 标注 Alpha，`pyproject.toml` classifier 标注 Beta。 | 统一成熟度、版本号及版本说明；不要仅为外观一致而宣称未验证的成熟度。 |
| P1 | getting-started 推荐 `unittest discover`，但 `tests/python/test_examples.py` 等使用 pytest 函数式测试。 | 统一公开测试命令为 pytest，确保新增测试被收集；核对开发文档和 CI。 |
| P1 | 原生 CPU MessagePassing 重复调用时，公开 cache misses 每次增长；`python/tiga/message_passing/core.py` 中直接执行 `self._cache_misses += 1`。 | 对齐公开统计和实际缓存层级；增加有意义的回归测试，覆盖相同输入、结构相同的新输入、需要重新编译的变化。详情见下一节。 |
| P1 | `docs/comparison.md` 将 PyG 概括为不做 fusion；`benchmarks/sparse_compute/weighted_aggregation.py` 的 `optional_pyg` 实际使用自定义 gather-scatter 路径。 | 将比较结论限定到实测实现、版本和 workload；同步中文页面。若要宣称优于其他 PyG 路径，应另行测量。 |

## JIT 缓存与执行语义：已复现的具体问题

使用 `examples/message_passing_autograd.py` 中的原生 CPU kernel，对相同 graph 和字段重复调用并通过 `tolist()` 取得结果：

```text
首次调用：hits=0, misses=1, variants=1
重复调用：hits=0, misses=2, variants=1
再次调用：hits=0, misses=3, variants=1
```

输出约为 `[11.1, 8.2, 17.3]`，temperature 梯度为 `[7.0, 10.0, 3.0]`，计算结果正确。

`kernel.explain()` 将上述统计展示为 executable cache。底层 `python/tiga/compiler/cpu_tensor.py` 存在独立的 identity/structure executable cache，因此本次复现**不能证明每次都重新编译**。修复应解决统计含义与实际执行不一致，不能仅把计数改为看似正常。

同时核对公开契约：原生 Tensor 使用 lazy execution，而 README 把 MessagePassing 调用描述为 JIT 边界。需要说明 capture、compile、materialize、launch 的触发时机，并让 diagnostics 和 timing 与实际事件一致。

## 重命名收尾

- 检查 README、examples、文档中的 `import tiga as gf` 是否统一为更符合品牌的写法；这是公共示例风格选择，不要求同步改内部 IR。
- 根 metadata、MkDocs 配置仍使用旧品牌文档地址；先确定公开地址，再处理链接和重定向。
- 审查时 git remote 为 `git@github.com:walkerchi/graphforge.git`。该信息仅用于迁移清单，本次未修改 remote，也未验证远程仓库重命名状态。
- `docs/development.md` 写出 `include/tiga/`，当前目录实际仍为 `include/graphforge/`；对齐真实布局或迁移方案。
- 内部 `gf.*` IR、工具名、C ABI、扩展模块名、持久化格式应单独评估兼容性，不进行无差别全仓替换。

## 发布完整性：需要补足的工程工作

1. 明确首版支持范围。支持矩阵应区分 relation、backend、dtype、reducer、autograd 和原生 Tensor/Torch adapter；已有 provider 接口不等于已支持该硬件。当前原生 differentiable MessagePassing 显式限制部分 relation 类型，不能从 adapter 能力推断所有入口支持相同组合。
2. 发布前增加 GPU 验证。当前主 CI 配置没有 GPU job；对核心 forward/backward、缓存失效、编译诊断建立定期或发布前 gate。硬件/runner 不可用时记录未验证项，不能用 skip 代替通过。
3. 加强安装后 smoke。`tests/smoke/wheel_without_torch.py` 已验证基础 Tensor 运算、metadata 和编译工具版本；补充代表性的 MessagePassing forward/backward，并覆盖承诺支持的平台。
4. 明确安装兼容范围。当前 Linux wheel 目标为 `manylinux_2_38_x86_64`，要求 glibc 2.38 或更新；安装页应标明限制。是否降低构建基线需结合 LLVM SDK 与目标用户评估。
5. 运行实际 release matrix。workflow 文件存在不代表各平台发布链路已验证。区分本地 wheel、干净环境安装、sdist 重建和公开发布状态。
6. 首版发布工程应前移。`docs/roadmap.md` 当前把发布排在扩展 kNN、稀疏覆盖和多设备验证之后；建议先完成有限但可靠的公开 alpha。

## 文档、性能与维护建议

- 精简 README 首屏顺序：一句定位、最小程序、安装、支持范围、代表性性能证据和详细链接。首次示例应有明确的执行结果和观察方式。
- 提炼语言语义说明：支持的 Python 语法/副作用、specialization 规则、空邻居/重复边/边顺序、reducer 顺序、dtype/累加精度、确定性、动态图梯度边界，以及稳定/experimental API。
- 保留现有 benchmark 协议优势：固定 baseline、置信区间、cold/warm、已知失败边界。补真实图应用的 forward/backward、转换成本、峰值内存、普通调用与 prepared 调用、编译成本摊平次数。每次发布绑定原始结果、代码和依赖版本、复现命令。
- 对比 PyG 时明确实现范围。PyG 官方提供通过 `message_and_aggregate()` 融合计算的路径，现有 gather-scatter baseline 的结果不能泛化为整个框架的能力。
- 补充 `CONTRIBUTING.md`、`CHANGELOG.md` 和 bug/PR 模板；根据项目需要增加 `SECURITY.md`、`CITATION.cff`。编译问题模板应包含环境、最小程序和 diagnostics。
- `lib/Target/Triton/Translate.cpp` 审查时约 8,600 行；按 lowering 家族和公共 emitter 逐步拆分，配套现有回归测试。文件长度本身不是缺陷，不为满足行数指标做大规模重构。

## 已完成验证与限制

- 工程边界测试：10 个通过。
- 核心示例结果测试：6 个通过，覆盖 MessagePassing autograd、GCN、diffusion、自定义 reducer、radius autograd、tensor matmul。
- `python -m tiga` 在 `PYTHONPATH=python` 下正常运行，能找到本地编译工具。
- 文档严格构建通过，使用已有完整依赖的本机环境，输出到临时目录；不能据此证明 CI 依赖完整。
- `uv lock --check --offline` 未通过，因为新 i18n 依赖不在本地缓存。这个结果不能证明线上依赖不可解析；锁文件内容与 metadata 不一致另有直接文件证据。
- 未重新运行完整 Python/MLIR/GPU suite、完整 benchmark、干净的跨平台 release matrix。
- 前一轮审查未修改源码；本次只新增此交接文件。

## 外部依据

- [PyG Memory-Efficient Aggregations](https://pytorch-geometric.readthedocs.io/en/latest/notes/sparse_tensor.html)：说明 `message_and_aggregate()` 与 sparse 路径。
- [PyPA Platform compatibility tags](https://packaging.python.org/en/latest/specifications/platform-compatibility-tags/)：说明 manylinux 标签与 glibc 基线。

## 建议交付方式

先完成 P0 修复和必要的重命名一致性修改，再处理缓存诊断与对比文案。每批交付说明具体修改、验证结果和剩余限制。公共 API 变化、大范围重构与外部发布单独列出，避免混入机械迁移。
