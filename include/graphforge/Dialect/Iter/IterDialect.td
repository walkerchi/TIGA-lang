#ifndef GRAPHFORGE_ITER_DIALECT_TD
#define GRAPHFORGE_ITER_DIALECT_TD

include "mlir/IR/OpBase.td"

def GraphForge_IterDialect : Dialect {
  let name = "gf_iter";
  let cppNamespace = "::mlir::graphforge::iter";
  let summary = "Coordinate iteration plans for GraphForge relations";
  let description = [{
    The iteration dialect resolves a logical relation into a coordinate
    hierarchy and ordering. It remains independent of hardware workgroups,
    subgroup width, memory-space encodings and vendor assembly.
  }];
}

class GraphForge_IterOp<string mnemonic, list<Trait> traits = []>
    : Op<GraphForge_IterDialect, mnemonic, traits>;

#endif
