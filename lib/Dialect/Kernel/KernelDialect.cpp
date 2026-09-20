#include "tiga/Dialect/Kernel/KernelDialect.h"

#include "tiga/Dialect/Kernel/KernelOpsDialect.cpp.inc"

using namespace mlir;
using namespace mlir::tiga::kernel;

void TigaKernelDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tiga/Dialect/Kernel/KernelOps.cpp.inc"
      >();
}
