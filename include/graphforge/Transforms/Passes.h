#ifndef GRAPHFORGE_TRANSFORMS_PASSES_H
#define GRAPHFORGE_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir::graphforge {

#define GEN_PASS_DECL
#include "graphforge/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "graphforge/Transforms/Passes.h.inc"

} // namespace mlir::graphforge

#endif

