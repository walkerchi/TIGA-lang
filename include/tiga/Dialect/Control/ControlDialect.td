#ifndef TIGA_CONTROL_DIALECT_TD
#define TIGA_CONTROL_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Tiga_ControlDialect : Dialect {
  let name = "gf_control";
  let cppNamespace = "::mlir::tiga::control";
  let summary = "Structured control flow over Tiga semantic values";
  let description = [{
    This dialect preserves loop-carried tensors and convergence semantics above
    provider kernels.  A provider may lower a loop to SCF, repeated launches,
    a command graph, or a persistent kernel when its synchronization contract
    permits it.
  }];
}

class Tiga_ControlOp<string mnemonic, list<Trait> traits = []>
    : Op<Tiga_ControlDialect, mnemonic, traits>;

#endif
