#include "tiga/Target/Triton/Translate.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

int main(int argc, char **argv) {
  mlir::tiga::registerKernelToTritonTranslation();
  mlir::tiga::registerTensorToTritonTranslation();
  mlir::tiga::registerTaskToBundleTranslation();
  return mlir::failed(
      mlir::mlirTranslateMain(argc, argv, "Tiga translator"));
}
