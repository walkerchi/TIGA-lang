#include "graphforge/Target/Triton/Translate.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

int main(int argc, char **argv) {
  mlir::graphforge::registerKernelToTritonTranslation();
  mlir::graphforge::registerTensorToTritonTranslation();
  mlir::graphforge::registerTaskToBundleTranslation();
  return mlir::failed(
      mlir::mlirTranslateMain(argc, argv, "GraphForge translator"));
}
