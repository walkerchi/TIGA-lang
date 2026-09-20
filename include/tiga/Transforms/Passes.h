#ifndef TIGA_TRANSFORMS_PASSES_H
#define TIGA_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir::tiga {

#define GEN_PASS_DECL
#include "tiga/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "tiga/Transforms/Passes.h.inc"

} // namespace mlir::tiga

#endif
