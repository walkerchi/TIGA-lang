#ifndef TIGA_DOMAIN_DIALECT_TD
#define TIGA_DOMAIN_DIALECT_TD

include "mlir/IR/OpBase.td"

def Tiga_DomainDialect : Dialect {
  let name = "gf";
  let cppNamespace = "::mlir::tiga";
  let summary = "Relation semantics for Tiga scientific programs";
  let description = [{
    The Tiga domain dialect preserves logical relation identity, field
    snapshot versions, reducer semantics, and edge/node regions before any
    target traversal or storage placement is selected.
  }];
  let useDefaultTypePrinterParser = 1;
}

class Tiga_Op<string mnemonic, list<Trait> traits = []>
    : Op<Tiga_DomainDialect, mnemonic, traits>;

#endif
