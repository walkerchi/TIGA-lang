#include "tiga/Dialect/Iter/IterDialect.h"

#include "tiga/Dialect/Iter/IterOpsDialect.cpp.inc"

using namespace mlir;
using namespace mlir::tiga::iter;

void TigaIterDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tiga/Dialect/Iter/IterOps.cpp.inc"
      >();
}
