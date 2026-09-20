#ifndef TIGA_TASK_DIALECT_TD
#define TIGA_TASK_DIALECT_TD

include "mlir/IR/OpBase.td"

def Tiga_TaskDialect : Dialect {
  let name = "gf_task";
  let cppNamespace = "::mlir::tiga::task";
  let summary = "Partitioned task and communication dependency semantics";
  let description = [{
    The task dialect represents ownership, ghost regions, halo requirements,
    collectives and event dependencies. It exposes communication/compute
    overlap to the compiler without exposing send/recv in user programs.
  }];
  let useDefaultTypePrinterParser = 1;
}

class Tiga_TaskOp<string mnemonic, list<Trait> traits = []>
    : Op<Tiga_TaskDialect, mnemonic, traits>;

#endif
