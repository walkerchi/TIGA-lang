#ifndef TIGA_ITER_DIALECT_TD
#define TIGA_ITER_DIALECT_TD

include "mlir/IR/OpBase.td"

def Tiga_IterDialect : Dialect {
  let name = "gf_iter";
  let cppNamespace = "::mlir::tiga::iter";
  let summary = "Coordinate iteration plans for Tiga relations";
  let description = [{
    The iteration dialect resolves a logical relation into a coordinate
    hierarchy and ordering. It remains independent of hardware workgroups,
    subgroup width, memory-space encodings and vendor assembly.
  }];
}

class Tiga_IterOp<string mnemonic, list<Trait> traits = []>
    : Op<Tiga_IterDialect, mnemonic, traits>;

#endif
