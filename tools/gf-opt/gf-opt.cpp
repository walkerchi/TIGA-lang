#include "tiga/Dialect/Control/ControlDialect.h"
#include "tiga/Dialect/Domain/DomainDialect.h"
#include "tiga/Dialect/Iter/IterDialect.h"
#include "tiga/Dialect/Kernel/KernelDialect.h"
#include "tiga/Dialect/Storage/StorageDialect.h"
#include "tiga/Dialect/Task/TaskDialect.h"
#include "tiga/Dialect/Tensor/TensorDialect.h"
#include "tiga/Transforms/Passes.h"
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
  mlir::tiga::registerTigaPasses();
  mlir::registerTransformsPasses();
  mlir::DialectRegistry registry;
  registry.insert<mlir::tiga::control::TigaControlDialect,
                  mlir::tiga::TigaDomainDialect,
                  mlir::tiga::iter::TigaIterDialect,
                  mlir::tiga::kernel::TigaKernelDialect,
                  mlir::tiga::storage::TigaStorageDialect,
                  mlir::tiga::task::TigaTaskDialect,
                  mlir::tiga::tensor::TigaTensorDialect,
                  mlir::arith::ArithDialect, mlir::complex::ComplexDialect,
                  mlir::func::FuncDialect, mlir::math::MathDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                  mlir::vector::VectorDialect>();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Tiga optimizer\n", registry));
}
