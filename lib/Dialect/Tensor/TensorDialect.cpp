#include "graphforge/Dialect/Tensor/TensorDialect.h"

#include "graphforge/Dialect/Tensor/TensorOpsDialect.cpp.inc"

using namespace mlir;
using namespace mlir::graphforge::tensor;

void GraphForgeTensorDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "graphforge/Dialect/Tensor/TensorOps.cpp.inc"
      >();
}
