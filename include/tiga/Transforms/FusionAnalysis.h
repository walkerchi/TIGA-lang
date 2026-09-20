#ifndef TIGA_TRANSFORMS_FUSIONANALYSIS_H
#define TIGA_TRANSFORMS_FUSIONANALYSIS_H

#include "tiga/Dialect/Domain/DomainDialect.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::tiga {

enum class FusionRejection {
  None,
  DifferentBlock,
  DifferentRelation,
  InterveningOperation,
  DataDependency,
  SnapshotVersionMismatch,
  UnknownOrMutatingEffect,
  DeterminismMismatch,
};

struct FusionDecision {
  FusionRejection rejection = FusionRejection::None;

  bool isLegal() const { return rejection == FusionRejection::None; }
  llvm::StringRef reason() const;
};

FusionDecision analyzeHorizontalFusion(ApplyOp producer, ApplyOp consumer);

} // namespace mlir::tiga

#endif
