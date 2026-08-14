#include "graphforge/Dialect/Control/ControlDialect.h"

#include "graphforge/Dialect/Control/ControlOpsDialect.cpp.inc"

using namespace mlir;
using namespace mlir::graphforge::control;

void GraphForgeControlDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "graphforge/Dialect/Control/ControlOps.cpp.inc"
      >();
}
