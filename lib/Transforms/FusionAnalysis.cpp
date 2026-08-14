#include "graphforge/Transforms/FusionAnalysis.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/BuiltinAttributes.h"

using namespace mlir;
using namespace mlir::graphforge;

llvm::StringRef FusionDecision::reason() const {
  switch (rejection) {
  case FusionRejection::None:
    return "legal";
  case FusionRejection::DifferentBlock:
    return "applies are in different blocks";
  case FusionRejection::DifferentRelation:
    return "relation SSA values differ";
  case FusionRejection::InterveningOperation:
    return "applies are not adjacent";
  case FusionRejection::DataDependency:
    return "consumer reads a producer result";
  case FusionRejection::SnapshotVersionMismatch:
    return "a shared input has different snapshot versions";
  case FusionRejection::UnknownOrMutatingEffect:
    return "an apply has a non-read or unknown effect";
  case FusionRejection::DeterminismMismatch:
    return "determinism policies differ";
  }
  llvm_unreachable("unhandled fusion rejection");
}

static bool hasReadOnlyEffects(ApplyOp op) {
  auto effects = op->getAttrOfType<ArrayAttr>("effects");
  return llvm::all_of(effects, [](Attribute attribute) {
    auto effect = dyn_cast<StringAttr>(attribute);
    return effect && effect.getValue() == "read";
  });
}

static bool sharedInputsHaveSameVersion(ApplyOp first, ApplyOp second) {
  ArrayRef<int64_t> firstVersions =
      first->getAttrOfType<DenseI64ArrayAttr>("snapshot_versions")
          .asArrayRef();
  ArrayRef<int64_t> secondVersions =
      second->getAttrOfType<DenseI64ArrayAttr>("snapshot_versions")
          .asArrayRef();
  for (auto [firstIndex, firstInput] : llvm::enumerate(first.getInputs()))
    for (auto [secondIndex, secondInput] : llvm::enumerate(second.getInputs()))
      if (firstInput == secondInput &&
          firstVersions[firstIndex] != secondVersions[secondIndex])
        return false;
  return true;
}

FusionDecision mlir::graphforge::analyzeHorizontalFusion(ApplyOp producer,
                                                          ApplyOp consumer) {
  if (producer->getBlock() != consumer->getBlock())
    return {FusionRejection::DifferentBlock};
  if (producer.getRelation() != consumer.getRelation())
    return {FusionRejection::DifferentRelation};
  if (producer->getNextNode() != consumer.getOperation())
    return {FusionRejection::InterveningOperation};
  for (Value input : consumer.getInputs())
    if (input.getDefiningOp() == producer.getOperation())
      return {FusionRejection::DataDependency};
  if (!sharedInputsHaveSameVersion(producer, consumer))
    return {FusionRejection::SnapshotVersionMismatch};
  if (!hasReadOnlyEffects(producer) || !hasReadOnlyEffects(consumer))
    return {FusionRejection::UnknownOrMutatingEffect};
  if (producer->getAttr("deterministic") !=
      consumer->getAttr("deterministic"))
    return {FusionRejection::DeterminismMismatch};
  return {};
}
