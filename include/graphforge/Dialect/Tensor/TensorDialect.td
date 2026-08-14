#ifndef GRAPHFORGE_TENSOR_DIALECT_TD
#define GRAPHFORGE_TENSOR_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def GraphForge_TensorDialect : Dialect {
  let name = "gf_tensor";
  let cppNamespace = "::mlir::graphforge::tensor";
  let summary = "Canonical shape-aware tensor programs captured by GraphForge";
  let description = [{
    The tensor dialect is the framework-independent, target-independent IR for
    GraphForge Tensor programs.  It preserves broadcasting, view transforms,
    reductions and differentiation requests until compiler passes have enough
    information to select loops, layouts, memory spaces and communication.
  }];
}

class GraphForge_TensorOp<string mnemonic, list<Trait> traits = []>
    : Op<GraphForge_TensorDialect, mnemonic, !listconcat([Pure], traits)>;

#endif
