#ifndef TIGA_STORAGE_DIALECT_TD
#define TIGA_STORAGE_DIALECT_TD

include "mlir/IR/OpBase.td"

def Tiga_StorageDialect : Dialect {
  let name = "gf_storage";
  let cppNamespace = "::mlir::tiga::storage";
  let summary = "Versioned physical storage and asynchronous transfer semantics";
  let description = [{
    The storage dialect separates logical regions from physical instances in
    register/shared/HBM/RAM/NVMe/remote spaces. Events retain transfer and
    lifetime dependencies before a target runtime is selected.
  }];
  let useDefaultTypePrinterParser = 1;
}

class Tiga_StorageOp<string mnemonic, list<Trait> traits = []>
    : Op<Tiga_StorageDialect, mnemonic, traits>;

#endif
