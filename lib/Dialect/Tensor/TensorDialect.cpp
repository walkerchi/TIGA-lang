#include "tiga/Dialect/Tensor/TensorDialect.h"

#include "tiga/Dialect/Tensor/TensorOpsDialect.cpp.inc"

using namespace mlir;
using namespace mlir::tiga::tensor;

void TigaTensorDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tiga/Dialect/Tensor/TensorOps.cpp.inc"
      >();
}
