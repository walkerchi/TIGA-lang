#ifndef TIGA_TENSOR_DIALECT_TD
#define TIGA_TENSOR_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Tiga_TensorDialect : Dialect {
  let name = "gf_tensor";
  let cppNamespace = "::mlir::tiga::tensor";
  let summary = "Canonical shape-aware tensor programs captured by Tiga";
  let description = [{
    The tensor dialect is the framework-independent, target-independent IR for
    Tiga Tensor programs.  It preserves broadcasting, view transforms,
    reductions and differentiation requests until compiler passes have enough
    information to select loops, layouts, memory spaces and communication.
  }];
}

class Tiga_TensorOp<string mnemonic, list<Trait> traits = []>
    : Op<Tiga_TensorDialect, mnemonic, !listconcat([Pure], traits)>;

#endif
