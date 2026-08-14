#include "graphforge/Dialect/Kernel/KernelDialect.h"

#include "graphforge/Dialect/Kernel/KernelOpsDialect.cpp.inc"

using namespace mlir;
using namespace mlir::graphforge::kernel;

void GraphForgeKernelDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "graphforge/Dialect/Kernel/KernelOps.cpp.inc"
      >();
}
