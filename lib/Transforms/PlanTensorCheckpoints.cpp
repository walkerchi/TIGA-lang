#include "tiga/Transforms/Passes.h"

#include "tiga/Dialect/Tensor/TensorDialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>

namespace mlir::tiga {

#define GEN_PASS_DEF_GFPLANTENSORCHECKPOINTS
#include "tiga/Transforms/Passes.h.inc"

namespace {
namespace gft = mlir::tiga::tensor;

static FailureOr<int64_t> staticBytes(RankedTensorType type) {
  if (!type.hasStaticShape())
    return failure();
  Type element = type.getElementType();
  int64_t bits = 0;
  if (auto integer = dyn_cast<IntegerType>(element))
    bits = integer.getWidth();
  else if (auto floating = dyn_cast<FloatType>(element))
    bits = floating.getWidth();
  else if (auto complex = dyn_cast<ComplexType>(element)) {
    auto component = dyn_cast<FloatType>(complex.getElementType());
    if (!component)
      return failure();
    bits = 2 * component.getWidth();
  } else {
    return failure();
  }
  if (bits <= 0)
    return failure();
  return type.getNumElements() * ((bits + 7) / 8);
}

static int64_t saturatingAdd(int64_t lhs, int64_t rhs) {
  if (rhs > 0 && lhs > std::numeric_limits<int64_t>::max() - rhs)
    return std::numeric_limits<int64_t>::max();
  return lhs + rhs;
}

static int64_t saturatingMultiply(int64_t lhs, int64_t rhs) {
  if (lhs <= 0 || rhs <= 0)
    return 0;
  if (lhs > std::numeric_limits<int64_t>::max() / rhs)
    return std::numeric_limits<int64_t>::max();
  return lhs * rhs;
}

static int64_t resultElements(Operation *operation) {
  int64_t elements = 1;
  bool foundTensor = false;
  for (Value result : operation->getResults()) {
    auto type = dyn_cast<RankedTensorType>(result.getType());
    if (!type || !type.hasStaticShape())
      continue;
    foundTensor = true;
    elements = std::max<int64_t>(elements, type.getNumElements());
  }
  return foundTensor ? elements : 1;
}

/// A deterministic target-independent recomputation estimate.  It deliberately
/// remains in abstract scalar-work units: target latency belongs in the later
/// schedule cost model, while this pass only needs a stable profitability
/// ordering.  Irregular reads and reductions carry larger weights than
/// componentwise arithmetic.
static int64_t operationCost(Operation *operation) {
  if (isa<gft::InputOp>(operation))
    return 0;
  int64_t weight = 1;
  if (isa<gft::GatherOp>(operation))
    weight = 8;
  else if (isa<gft::SegmentSumOp, gft::CSRSegmentSumOp>(operation))
    weight = 6;
  else if (isa<gft::ReduceSumOp>(operation))
    weight = 3;
  else if (isa<gft::PermuteOp, gft::BroadcastOp, gft::ReshapeOp>(operation))
    weight = 1;
  return saturatingMultiply(resultElements(operation), weight);
}

static int64_t producerCost(Value value, DenseMap<Value, int64_t> &memo) {
  auto found = memo.find(value);
  if (found != memo.end())
    return found->second;
  Operation *producer = value.getDefiningOp();
  if (!producer || isa<gft::InputOp>(producer)) {
    memo[value] = 0;
    return 0;
  }
  int64_t cost = operationCost(producer);
  for (Value operand : producer->getOperands())
    cost = saturatingAdd(cost, producerCost(operand, memo));
  memo[value] = cost;
  return cost;
}

struct CandidateInfo {
  gft::CheckpointCandidateOp operation;
  int64_t ordinal = 0;
  int64_t bytes = 0;
  int64_t recomputeCost = 0;
  int64_t liveStart = 0;
  int64_t liveEnd = 0;
  // 0 = recompute, 1 = device checkpoint, 2 = host-pinned spill.
  int64_t decision = 0;
};

static bool fitsInterval(SmallVectorImpl<int64_t> &occupancy,
                         const CandidateInfo &candidate, int64_t budget) {
  if (budget < 0)
    return true;
  for (int64_t point = candidate.liveStart; point <= candidate.liveEnd; ++point)
    if (occupancy[point] > budget - candidate.bytes)
      return false;
  return true;
}

static void reserveInterval(SmallVectorImpl<int64_t> &occupancy,
                            const CandidateInfo &candidate) {
  for (int64_t point = candidate.liveStart; point <= candidate.liveEnd; ++point)
    occupancy[point] += candidate.bytes;
}

static int64_t peak(const SmallVectorImpl<int64_t> &occupancy) {
  return occupancy.empty()
             ? 0
             : *std::max_element(occupancy.begin(), occupancy.end());
}

class PlanTensorCheckpointsPass
    : public impl::GFPlanTensorCheckpointsBase<PlanTensorCheckpointsPass> {
public:
  using impl::GFPlanTensorCheckpointsBase<
      PlanTensorCheckpointsPass>::GFPlanTensorCheckpointsBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    SmallVector<Operation *> orderedOperations;
    module.walk([&](Operation *operation) {
      if (operation != module.getOperation())
        orderedOperations.push_back(operation);
    });
    DenseMap<Operation *, int64_t> positions;
    for (auto [position, operation] : llvm::enumerate(orderedOperations))
      positions[operation] = position;

    SmallVector<CandidateInfo> candidates;
    DenseMap<Value, int64_t> costMemo;
    module.walk([&](gft::CheckpointCandidateOp candidate) {
      CandidateInfo info;
      info.operation = candidate;
      info.ordinal = candidates.size();
      info.liveStart = positions.lookup(candidate.getOperation());
      info.liveEnd = info.liveStart;
      for (OpOperand &use : candidate.getResult().getUses()) {
        auto found = positions.find(use.getOwner());
        if (found != positions.end())
          info.liveEnd = std::max(info.liveEnd, found->second);
      }
      if (auto type = dyn_cast<RankedTensorType>(candidate.getResult().getType()))
        if (FailureOr<int64_t> bytes = staticBytes(type); succeeded(bytes))
          info.bytes = *bytes;
      info.recomputeCost = producerCost(candidate.getInput(), costMemo);
      candidates.push_back(info);
    });

    SmallVector<int64_t> priority(candidates.size());
    for (auto [ordinal, slot] : llvm::enumerate(priority))
      slot = ordinal;
    llvm::stable_sort(priority, [&](int64_t lhsIndex, int64_t rhsIndex) {
      const CandidateInfo &lhs = candidates[lhsIndex];
      const CandidateInfo &rhs = candidates[rhsIndex];
      // Compare benefit/byte without floating point instability.
      __int128 lhsScore = static_cast<__int128>(lhs.recomputeCost) * rhs.bytes;
      __int128 rhsScore = static_cast<__int128>(rhs.recomputeCost) * lhs.bytes;
      if (lhsScore != rhsScore)
        return lhsScore > rhsScore;
      if (lhs.recomputeCost != rhs.recomputeCost)
        return lhs.recomputeCost > rhs.recomputeCost;
      return lhs.ordinal < rhs.ordinal;
    });

    SmallVector<int64_t> deviceOccupancy(orderedOperations.size() + 1, 0);
    SmallVector<int64_t> spillOccupancy(orderedOperations.size() + 1, 0);
    for (int64_t index : priority) {
      CandidateInfo &candidate = candidates[index];
      if (candidate.bytes <= 0 || candidate.recomputeCost <= 0)
        continue;
      if (fitsInterval(deviceOccupancy, candidate, memoryBudgetBytes)) {
        candidate.decision = 1;
        reserveInterval(deviceOccupancy, candidate);
      } else if (spillBudgetBytes != 0 &&
                 fitsInterval(spillOccupancy, candidate, spillBudgetBytes)) {
        candidate.decision = 2;
        reserveInterval(spillOccupancy, candidate);
      }
    }

    SmallVector<int64_t> decisions(candidates.size(), 0);
    SmallVector<int64_t> costs(candidates.size(), 0);
    SmallVector<int64_t> intervals;
    SmallVector<Attribute> tiers;
    intervals.reserve(candidates.size() * 2);
    tiers.reserve(candidates.size());
    int64_t selectedDeviceBytes = 0;
    int64_t selectedSpillBytes = 0;
    OpBuilder builder(&getContext());
    for (CandidateInfo &candidate : candidates) {
      decisions[candidate.ordinal] = candidate.decision;
      costs[candidate.ordinal] = candidate.recomputeCost;
      intervals.push_back(candidate.liveStart);
      intervals.push_back(candidate.liveEnd);
      StringRef tier = candidate.decision == 1 ? "device"
                       : candidate.decision == 2 ? "host-pinned"
                                                 : "recompute";
      tiers.push_back(builder.getStringAttr(tier));
      if (candidate.decision != 0) {
        if (candidate.decision == 1)
          selectedDeviceBytes += candidate.bytes;
        else
          selectedSpillBytes += candidate.bytes;
        builder.setInsertionPoint(candidate.operation);
        auto checkpoint = builder.create<gft::CheckpointOp>(
            candidate.operation.getLoc(),
            candidate.operation.getResult().getType(),
            candidate.operation.getInput());
        checkpoint->setAttr("storage_tier", builder.getStringAttr(tier));
        checkpoint->setAttr("bytes", builder.getI64IntegerAttr(candidate.bytes));
        checkpoint->setAttr(
            "recompute_cost",
            builder.getI64IntegerAttr(candidate.recomputeCost));
        checkpoint->setAttr("live_start",
                            builder.getI64IntegerAttr(candidate.liveStart));
        checkpoint->setAttr("live_end",
                            builder.getI64IntegerAttr(candidate.liveEnd));
        candidate.operation.getResult().replaceAllUsesWith(
            checkpoint.getResult());
      } else {
        candidate.operation.getResult().replaceAllUsesWith(
            candidate.operation.getInput());
      }
      candidate.operation.erase();
    }

    Builder attributes(&getContext());
    module->setAttr("gf_tensor.checkpoint_decisions",
                    attributes.getDenseI64ArrayAttr(decisions));
    module->setAttr("gf_tensor.checkpoint_tiers",
                    attributes.getArrayAttr(tiers));
    module->setAttr("gf_tensor.checkpoint_recompute_costs",
                    attributes.getDenseI64ArrayAttr(costs));
    module->setAttr("gf_tensor.checkpoint_live_intervals",
                    attributes.getDenseI64ArrayAttr(intervals));
    module->setAttr("gf_tensor.checkpoint_candidate_count",
                    attributes.getI64IntegerAttr(candidates.size()));
    // Preserve the original attribute as the device-resident sum for API
    // compatibility; peak_live_bytes is the actual capacity requirement.
    module->setAttr("gf_tensor.checkpoint_saved_bytes",
                    attributes.getI64IntegerAttr(selectedDeviceBytes));
    module->setAttr("gf_tensor.checkpoint_spilled_bytes",
                    attributes.getI64IntegerAttr(selectedSpillBytes));
    module->setAttr("gf_tensor.checkpoint_peak_live_bytes",
                    attributes.getI64IntegerAttr(peak(deviceOccupancy)));
    module->setAttr("gf_tensor.checkpoint_peak_spill_bytes",
                    attributes.getI64IntegerAttr(peak(spillOccupancy)));
    module->setAttr("gf_tensor.checkpoint_memory_budget_bytes",
                    attributes.getI64IntegerAttr(memoryBudgetBytes));
    module->setAttr("gf_tensor.checkpoint_spill_budget_bytes",
                    attributes.getI64IntegerAttr(spillBudgetBytes));
  }
};

} // namespace
} // namespace mlir::tiga
