#ifndef GRAPHFORGE_CONTROL_DIALECT_TD
#define GRAPHFORGE_CONTROL_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def GraphForge_ControlDialect : Dialect {
  let name = "gf_control";
  let cppNamespace = "::mlir::graphforge::control";
  let summary = "Structured control flow over GraphForge semantic values";
  let description = [{
    This dialect preserves loop-carried tensors and convergence semantics above
    provider kernels.  A provider may lower a loop to SCF, repeated launches,
    a command graph, or a persistent kernel when its synchronization contract
    permits it.
  }];
}

class GraphForge_ControlOp<string mnemonic, list<Trait> traits = []>
    : Op<GraphForge_ControlDialect, mnemonic, traits>;

#endif
