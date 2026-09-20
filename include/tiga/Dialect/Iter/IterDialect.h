#ifndef TIGA_DIALECT_ITER_ITERDIALECT_H
#define TIGA_DIALECT_ITER_ITERDIALECT_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"

#include "tiga/Dialect/Domain/DomainDialect.h"
#include "tiga/Dialect/Iter/IterOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "tiga/Dialect/Iter/IterOps.h.inc"

#endif
