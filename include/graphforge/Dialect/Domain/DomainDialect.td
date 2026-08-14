#ifndef GRAPHFORGE_DOMAIN_DIALECT_TD
#define GRAPHFORGE_DOMAIN_DIALECT_TD

include "mlir/IR/OpBase.td"

def GraphForge_DomainDialect : Dialect {
  let name = "gf";
  let cppNamespace = "::mlir::graphforge";
  let summary = "Relation semantics for GraphForge scientific programs";
  let description = [{
    The GraphForge domain dialect preserves logical relation identity, field
    snapshot versions, reducer semantics, and edge/node regions before any
    target traversal or storage placement is selected.
  }];
  let useDefaultTypePrinterParser = 1;
}

class GraphForge_Op<string mnemonic, list<Trait> traits = []>
    : Op<GraphForge_DomainDialect, mnemonic, traits>;

#endif

