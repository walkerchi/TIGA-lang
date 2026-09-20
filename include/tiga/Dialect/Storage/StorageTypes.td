#ifndef TIGA_STORAGE_TYPES_TD
#define TIGA_STORAGE_TYPES_TD

include "tiga/Dialect/Storage/StorageDialect.td"
include "mlir/IR/AttrTypeBase.td"

class Tiga_StorageType<string name, string typeMnemonic,
                             list<Trait> traits = []>
    : TypeDef<Tiga_StorageDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Tiga_RegionType : Tiga_StorageType<"Region", "region"> {
  let summary = "A versioned logical subset of a Field or Relation";
  let assemblyFormat = "";
}

def Tiga_InstanceType : Tiga_StorageType<"Instance", "instance"> {
  let summary = "One physical realization of a logical region";
  let assemblyFormat = "";
}

def Tiga_EventType : Tiga_StorageType<"Event", "event"> {
  let summary = "An asynchronous copy, launch, collective, or lifetime event";
  let assemblyFormat = "";
}

#endif
