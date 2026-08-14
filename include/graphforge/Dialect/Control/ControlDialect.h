#ifndef GRAPHFORGE_DIALECT_CONTROL_CONTROLDIALECT_H
#define GRAPHFORGE_DIALECT_CONTROL_CONTROLDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "graphforge/Dialect/Control/ControlOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "graphforge/Dialect/Control/ControlOps.h.inc"

#endif
