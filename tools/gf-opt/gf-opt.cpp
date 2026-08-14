#include "graphforge/Dialect/Control/ControlDialect.h"
#include "graphforge/Dialect/Domain/DomainDialect.h"
#include "graphforge/Dialect/Iter/IterDialect.h"
#include "graphforge/Dialect/Kernel/KernelDialect.h"
#include "graphforge/Dialect/Storage/StorageDialect.h"
#include "graphforge/Dialect/Task/TaskDialect.h"
#include "graphforge/Dialect/Tensor/TensorDialect.h"
#include "graphforge/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::graphforge::registerGraphForgePasses();
  mlir::registerTransformsPasses();
  mlir::DialectRegistry registry;
  registry.insert<mlir::graphforge::control::GraphForgeControlDialect,
                  mlir::graphforge::GraphForgeDomainDialect,
                  mlir::graphforge::iter::GraphForgeIterDialect,
                  mlir::graphforge::kernel::GraphForgeKernelDialect,
                  mlir::graphforge::storage::GraphForgeStorageDialect,
                  mlir::graphforge::task::GraphForgeTaskDialect,
                  mlir::graphforge::tensor::GraphForgeTensorDialect,
                  mlir::arith::ArithDialect, mlir::complex::ComplexDialect,
                  mlir::func::FuncDialect, mlir::math::MathDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                  mlir::vector::VectorDialect>();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "GraphForge optimizer\n", registry));
}
