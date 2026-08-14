#ifndef GRAPHFORGE_STORAGE_TYPES_TD
#define GRAPHFORGE_STORAGE_TYPES_TD

include "graphforge/Dialect/Storage/StorageDialect.td"
include "mlir/IR/AttrTypeBase.td"

class GraphForge_StorageType<string name, string typeMnemonic,
                             list<Trait> traits = []>
    : TypeDef<GraphForge_StorageDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def GraphForge_RegionType : GraphForge_StorageType<"Region", "region"> {
  let summary = "A versioned logical subset of a Field or Relation";
  let assemblyFormat = "";
}

def GraphForge_InstanceType : GraphForge_StorageType<"Instance", "instance"> {
  let summary = "One physical realization of a logical region";
  let assemblyFormat = "";
}

def GraphForge_EventType : GraphForge_StorageType<"Event", "event"> {
  let summary = "An asynchronous copy, launch, collective, or lifetime event";
  let assemblyFormat = "";
}

#endif
