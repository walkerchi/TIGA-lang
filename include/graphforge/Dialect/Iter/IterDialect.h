#ifndef GRAPHFORGE_DIALECT_ITER_ITERDIALECT_H
#define GRAPHFORGE_DIALECT_ITER_ITERDIALECT_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"

#include "graphforge/Dialect/Domain/DomainDialect.h"
#include "graphforge/Dialect/Iter/IterOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "graphforge/Dialect/Iter/IterOps.h.inc"

#endif
