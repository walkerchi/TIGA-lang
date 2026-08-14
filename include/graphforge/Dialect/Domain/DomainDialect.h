#ifndef GRAPHFORGE_DIALECT_DOMAIN_DOMAINDIALECT_H
#define GRAPHFORGE_DIALECT_DOMAIN_DOMAINDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "graphforge/Dialect/Domain/DomainOpsDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "graphforge/Dialect/Domain/DomainTypes.h.inc"

#define GET_OP_CLASSES
#include "graphforge/Dialect/Domain/DomainOps.h.inc"

#endif
