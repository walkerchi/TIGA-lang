#ifndef GRAPHFORGE_STORAGE_OPS_TD
#define GRAPHFORGE_STORAGE_OPS_TD

include "graphforge/Dialect/Storage/StorageTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def GraphForge_RegionOp : GraphForge_StorageOp<"region", [Pure]> {
  let summary = "Bind an external value to a versioned logical region";
  let arguments = (ins AnyType:$resource, StrAttr:$logical_id,
                       I64Attr:$version, I64Attr:$elements);
  let results = (outs GraphForge_RegionType:$result);
  let hasVerifier = 1;
}

def GraphForge_InstanceOp : GraphForge_StorageOp<"instance", [Pure]> {
  let summary = "Describe a physical instance and its memory space";
  let arguments = (ins GraphForge_RegionType:$region,
                       StrAttr:$memory_space, StrAttr:$layout,
                       StrAttr:$device, I64Attr:$capacity_bytes,
                       BoolAttr:$external);
  let results = (outs GraphForge_InstanceType:$result);
  let hasVerifier = 1;
}

def GraphForge_TransferOp : GraphForge_StorageOp<"transfer"> {
  let summary = "Asynchronously transfer one region between physical instances";
  let arguments = (ins GraphForge_InstanceType:$source,
                       GraphForge_InstanceType:$destination,
                       Variadic<GraphForge_EventType>:$depends_on,
                       I64Attr:$bytes, StrAttr:$engine,
                       I64Attr:$snapshot_version);
  let results = (outs GraphForge_EventType:$ready);
  let hasVerifier = 1;
}

def GraphForge_ReleaseOp : GraphForge_StorageOp<"release"> {
  let summary = "Release or recycle an instance after dependent events";
  let arguments = (ins GraphForge_InstanceType:$instance,
                       Variadic<GraphForge_EventType>:$depends_on,
                       I64Attr:$snapshot_version);
  let results = (outs GraphForge_EventType:$released);
  let hasVerifier = 1;
}

def GraphForge_JoinOp : GraphForge_StorageOp<"join"> {
  let summary = "Join asynchronous dependencies without forcing host sync";
  let arguments = (ins Variadic<GraphForge_EventType>:$events,
                       I64Attr:$snapshot_version);
  let results = (outs GraphForge_EventType:$joined);
  let hasVerifier = 1;
}

#endif
