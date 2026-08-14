#ifndef GRAPHFORGE_DIALECT_KERNEL_KERNELDIALECT_H
#define GRAPHFORGE_DIALECT_KERNEL_KERNELDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"

#include "graphforge/Dialect/Kernel/KernelOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "graphforge/Dialect/Kernel/KernelOps.h.inc"

#endif
