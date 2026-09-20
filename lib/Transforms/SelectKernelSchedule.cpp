#include "mlir/IR/BuiltinOps.h"
#include "graphforge/Transforms/Passes.h"

#include "graphforge/Dialect/Kernel/KernelDialect.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinTypes.h"

#include <algorithm>
#include <optional>

namespace mlir::graphforge {

#define GEN_PASS_DEF_GFSELECTKERNELSCHEDULE
#include "graphforge/Transforms/Passes.h.inc"

namespace {

static int64_t nextPowerOfTwo(int64_t value) {
  int64_t result = 1;
  while (result < value)
    result *= 2;
  return result;
}

static bool hasScalarFields(ValueRange inputs, ValueRange outputs) {
  return llvm::all_of(llvm::concat<Value>(inputs, outputs), [](Value value) {
    // Runtime scalar parameters participate in edge/node algebra but are not
    // entity feature lanes. They must not deoptimize an otherwise scalar
    // relation kernel to provider-deferred.
    if (isa<FloatType, IntegerType>(value.getType()))
      return true;
    auto tensor = dyn_cast<RankedTensorType>(value.getType());
    return tensor && tensor.getRank() == 1;
  });
}

static std::optional<int64_t> fixedVectorWidth(kernel::LaunchOp launch) {
  if (launch.getInputs().size() != 2 || launch.getOutputs().size() != 1)
    return std::nullopt;
  auto source = dyn_cast<RankedTensorType>(launch.getInputs()[0].getType());
  auto weight = dyn_cast<RankedTensorType>(launch.getInputs()[1].getType());
  auto output = dyn_cast<RankedTensorType>(launch.getOutputs()[0].getType());
  bool scalarWeight = weight &&
      (weight.getRank() == 1 ||
       (weight.getRank() == 2 && weight.getDimSize(1) == 1));
  if (!source || !weight || !output || source.getRank() != 2 ||
      !scalarWeight || output.getRank() != 2 ||
      source.getElementType() != Float32Type::get(launch.getContext()) ||
      weight.getElementType() != source.getElementType() ||
      output.getElementType() != source.getElementType() ||
      source.getDimSize(1) <= 1 ||
      output.getDimSize(1) != source.getDimSize(1))
    return std::nullopt;
  return source.getDimSize(1);
}

static void setSchedule(Operation *operation, StringRef kind,
                        int64_t blockRows, int64_t blockNeighbors,
                        int64_t numWarps, Builder &builder) {
  auto strings = [&](ArrayRef<StringRef> values) {
    SmallVector<Attribute> attributes;
    llvm::transform(values, std::back_inserter(attributes),
                    [&](StringRef value) { return builder.getStringAttr(value); });
    return builder.getArrayAttr(attributes);
  };
  SmallVector<StringRef> resources = {
      "input.global.read", "state.register.private", "output.global.write"};
  SmallVector<StringRef> roles;
  SmallVector<StringRef> handoffs = {
      "load.global-to-register", "store.register-to-global"};
  SmallVector<StringRef> instructions;
  BoolAttr deterministicAttr =
      operation->getAttrOfType<BoolAttr>("deterministic");
  bool deterministic = deterministicAttr && deterministicAttr.getValue();
  if (kind == "generated-hash-query-lanes") {
    resources.insert(resources.begin(), "directory.global.read");
    roles = {"workgroup.destination-queries", "subgroup.query-traversal", "lane.query"};
    handoffs.insert(handoffs.begin() + 1, "loop.active-any");
    instructions = {"masked-memory", "distance-filter", "ordered-reduce"};
  } else if (kind == "generated-cell-neighbor") {
    resources.insert(resources.begin(), "directory.global.read");
    roles = {"workgroup.destination-rows", "subgroup.neighbor-reduction",
             "lane.scalar"};
    handoffs.insert(handoffs.begin() + 1, "reduce.subgroup");
    instructions = {"masked-memory", "distance-filter",
                    deterministic ? "ordered-reduce" : "associative-reduce"};
  } else if (kind == "dense-query-key-tile") {
    roles = {"workgroup.destination-queries", "subgroup.key-reduction",
             "lane.feature"};
    handoffs.insert(handoffs.begin() + 1, "reduce.subgroup");
    instructions = {"masked-memory", "vector-contraction",
                    deterministic ? "ordered-reduce" : "associative-reduce"};
  } else if (kind == "ranked-candidate-select") {
    resources.insert(resources.begin(), "selection.register.private");
    roles = {"workgroup.destination-queries", "subgroup.ranked-selection",
             "lane.feature"};
    handoffs.insert(handoffs.begin() + 1, "reduce.subgroup");
    instructions = {"masked-memory", "hierarchical-topk",
                    "stable-index-tie-break"};
  } else if (kind == "provider-deferred") {
    roles = {"workgroup.destination-rows", "lane.scalar"};
    instructions = {"provider-selection-required"};
  } else if (kind == "scalar-row-loop") {
    resources.insert(resources.begin(), "relation.global.read");
    roles = {"workgroup.destination-rows", "lane.scalar"};
    instructions = {"scalar-control-flow",
                    deterministic ? "ordered-reduce" : "associative-reduce"};
  } else {
    resources.insert(resources.begin(), "relation.global.read");
    roles = {"workgroup.destination-rows", "subgroup.neighbor-reduction",
             (kind == "fixed-row-neighbor-feature" ||
              kind == "bounded-ragged-row-neighbor-feature")
                 ? "lane.feature"
                 : "lane.scalar"};
    handoffs.insert(handoffs.begin() + 1, "reduce.subgroup");
    instructions = {"masked-memory",
                    deterministic ? "ordered-reduce" : "associative-reduce"};
  }
  operation->setAttr("schedule_kind", builder.getStringAttr(kind));
  operation->setAttr("block_rows", builder.getI64IntegerAttr(blockRows));
  operation->setAttr("block_neighbors",
                     builder.getI64IntegerAttr(blockNeighbors));
  operation->setAttr("num_warps", builder.getI64IntegerAttr(numWarps));
  // One means that GraphForge has not admitted a compiler-controlled async
  // producer/consumer pipeline. A provider may still perform instruction
  // scheduling, but it cannot be reported as a semantic multi-stage pipeline.
  operation->setAttr("pipeline_stages", builder.getI64IntegerAttr(1));
  operation->setAttr("target_contract",
                     builder.getStringAttr("provider-neutral-v1"));
  operation->setAttr("schedule_resources", strings(resources));
  operation->setAttr("execution_roles", strings(roles));
  operation->setAttr("schedule_handoffs", strings(handoffs));
  operation->setAttr("instruction_contracts", strings(instructions));
}

static void annotateLoadBalance(kernel::LaunchOp launch, Builder &builder) {
  IntegerAttr maximum = launch.getDegreeMaxAttr();
  IntegerAttr sum = launch.getDegreeSumAttr();
  int64_t rows = launch.getNumRowsAttr().getInt();
  if (!maximum || !sum || rows <= 0 || maximum.getInt() <= 0)
    return;
  if (maximum.getInt() > 64) {
    // Oversized rows still need a compact row-id materialization: the common
    // rows remain row-tiled while only the high-degree tail is split.  The
    // split-row pass consumes the tail bucket after degree planning.
    launch->setAttr("load_balance_plan",
                    builder.getStringAttr("degree-bucketing-required"));
    launch->setAttr("split_row_threshold",
                    builder.getI64IntegerAttr(64));
    return;
  }
  int64_t paddedDegree = nextPowerOfTwo(maximum.getInt());
  long double padded = static_cast<long double>(rows) * paddedDegree;
  int64_t utilizationMilli = static_cast<int64_t>(
      1000.0L * static_cast<long double>(sum.getInt()) / padded);
  launch->setAttr("padding_utilization_milli",
                  builder.getI64IntegerAttr(utilizationMilli));
  launch->setAttr(
      "load_balance_plan",
      builder.getStringAttr(utilizationMilli < 600
                                ? "degree-bucketing-required"
                                : "row-neighbor-tile"));
}

class SelectKernelSchedulePass
    : public impl::GFSelectKernelScheduleBase<SelectKernelSchedulePass> {
public:
  using impl::GFSelectKernelScheduleBase<
      SelectKernelSchedulePass>::GFSelectKernelScheduleBase;

  void runOnOperation() final {
    Builder builder(&getContext());
    getOperation().walk([&](kernel::LaunchOp launch) {
      IntegerAttr minimum = launch.getDegreeMinAttr();
      IntegerAttr maximum = launch.getDegreeMaxAttr();
      if (std::optional<int64_t> width = fixedVectorWidth(launch);
          width && maximum && maximum.getInt() > 0 &&
          maximum.getInt() <= 64) {
        int64_t neighbors = nextPowerOfTwo(maximum.getInt());
        int64_t capacity = std::max<int64_t>(1, 16384 / (neighbors * *width));
        int64_t rows = 1;
        while (rows * 2 <= std::min<int64_t>(32, capacity)) rows *= 2;
        int64_t warps = std::clamp<int64_t>(rows * *width / 128, 1, 8);
        bool fixed = minimum && minimum.getInt() == maximum.getInt();
        setSchedule(
            launch,
            fixed ? "fixed-row-neighbor-feature"
                  : "bounded-ragged-row-neighbor-feature",
            rows, neighbors, warps, builder);
        launch->setAttr("block_features", builder.getI64IntegerAttr(*width));
        return;
      }
      if (!hasScalarFields(launch.getInputs(), launch.getOutputs())) {
        setSchedule(launch, "provider-deferred", 1, 1, 1, builder);
        return;
      }
      if (minimum && maximum && minimum.getInt() == maximum.getInt() &&
          minimum.getInt() > 0 && minimum.getInt() <= 64) {
        // A horizontally fused product carries more independent arithmetic
        // per relation tile.  Thirty-two rows amortize its wider ABI and two
        // result stores without increasing the neighbor reduction width.
        int64_t rows = launch.getNumResults() > 1 ? 32 : 16;
        setSchedule(launch, "fixed-row-neighbor", rows,
                    nextPowerOfTwo(minimum.getInt()), 1, builder);
        return;
      }
      if (maximum && maximum.getInt() > 0 && maximum.getInt() <= 64) {
        bool wide = maximum.getInt() > 32;
        setSchedule(launch, "bounded-ragged-row-neighbor", wide ? 32 : 16,
                    nextPowerOfTwo(maximum.getInt()), wide ? 1 : 4, builder);
        return;
      }
      setSchedule(launch, "scalar-row-loop", 1, 1, 4, builder);
    });
    getOperation().walk(
        [&](kernel::LaunchOp launch) { annotateLoadBalance(launch, builder); });
    getOperation().walk([&](kernel::GeneratedLaunchOp launch) {
      if (launch.getHashGrid())
        setSchedule(launch, "generated-hash-query-lanes", 32, 1, 1, builder);
      else
        setSchedule(launch, "generated-cell-neighbor", 1, 32, 1, builder);
    });
    getOperation().walk([&](kernel::DenseLaunchOp launch) {
      auto boundary = launch->getAttrOfType<StringAttr>("boundary");
      auto field = launch.getInputs().empty()
                       ? RankedTensorType()
                       : dyn_cast<RankedTensorType>(launch.getInputs()[0].getType());
      int64_t width = field && field.getRank() >= 3 ? field.getDimSize(2) : -1;
      // Wide payload state and triangular diagonal masking both favor a
      // 64-row query tile. Narrow full Cartesian reductions keep 128 rows.
      int64_t rows =
          (boundary && boundary.getValue() == "lower_inclusive") || width >= 64
              ? 64
              : 128;
      setSchedule(launch, "dense-query-key-tile", rows, 64, 4, builder);
    });
    getOperation().walk([&](kernel::RankedLaunchOp launch) {
      // These are provider-neutral capacities.  A provider may refine the
      // register layout, but must retain exact local selection and stable
      // source-index tie breaking before hierarchical merge.
      int64_t candidateTile = 256;
      int64_t k = launch.getKAttr().getInt();
      while (candidateTile < k) candidateTile *= 2;
      // One destination per program keeps the candidate tile and its sorting
      // network register-resident. Query packing is a provider refinement
      // once register pressure is modeled for a concrete target.
      int64_t rows = 1;
      setSchedule(launch, "ranked-candidate-select", rows, candidateTile, 4,
                  builder);
      launch->setAttr("candidate_tile",
                      builder.getI64IntegerAttr(candidateTile));
      launch->setAttr("merge_fan_in", builder.getI64IntegerAttr(2));
    });
  }
};

} // namespace
} // namespace mlir::graphforge
