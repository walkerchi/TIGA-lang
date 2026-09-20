#ifndef TIGA_STORAGE_OPS_TD
#define TIGA_STORAGE_OPS_TD

include "tiga/Dialect/Storage/StorageTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Tiga_RegionOp : Tiga_StorageOp<"region", [Pure]> {
  let summary = "Bind an external value to a versioned logical region";
  let arguments = (ins AnyType:$resource, StrAttr:$logical_id,
                       I64Attr:$version, I64Attr:$elements);
  let results = (outs Tiga_RegionType:$result);
  let hasVerifier = 1;
}

def Tiga_InstanceOp : Tiga_StorageOp<"instance", [Pure]> {
  let summary = "Describe a physical instance and its memory space";
  let arguments = (ins Tiga_RegionType:$region,
                       StrAttr:$memory_space, StrAttr:$layout,
                       StrAttr:$device, I64Attr:$capacity_bytes,
                       BoolAttr:$external);
  let results = (outs Tiga_InstanceType:$result);
  let hasVerifier = 1;
}

def Tiga_TransferOp : Tiga_StorageOp<"transfer"> {
  let summary = "Asynchronously transfer one region between physical instances";
  let arguments = (ins Tiga_InstanceType:$source,
                       Tiga_InstanceType:$destination,
                       Variadic<Tiga_EventType>:$depends_on,
                       I64Attr:$bytes, StrAttr:$engine,
                       I64Attr:$snapshot_version);
  let results = (outs Tiga_EventType:$ready);
  let hasVerifier = 1;
}

def Tiga_ReleaseOp : Tiga_StorageOp<"release"> {
  let summary = "Release or recycle an instance after dependent events";
  let arguments = (ins Tiga_InstanceType:$instance,
                       Variadic<Tiga_EventType>:$depends_on,
                       I64Attr:$snapshot_version);
  let results = (outs Tiga_EventType:$released);
  let hasVerifier = 1;
}

def Tiga_JoinOp : Tiga_StorageOp<"join"> {
  let summary = "Join asynchronous dependencies without forcing host sync";
  let arguments = (ins Variadic<Tiga_EventType>:$events,
                       I64Attr:$snapshot_version);
  let results = (outs Tiga_EventType:$joined);
  let hasVerifier = 1;
}

#endif
