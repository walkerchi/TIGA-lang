#ifndef TIGA_TASK_OPS_TD
#define TIGA_TASK_OPS_TD

include "tiga/Dialect/Domain/DomainTypes.td"
include "tiga/Dialect/Storage/StorageTypes.td"
include "tiga/Dialect/Task/TaskTypes.td"
include "mlir/IR/SymbolInterfaces.td"

def Tiga_PartitionOp : Tiga_TaskOp<"partition"> {
  let summary = "Attach destination ownership and ghost cardinality";
  let arguments = (ins Tiga_RelationType:$relation,
                       StrAttr:$scheme, DenseI64ArrayAttr:$mesh_shape,
                       I64Attr:$mesh_axis, I64Attr:$version,
                       I64Attr:$owned_entities, I64Attr:$ghost_entities);
  let results = (outs Tiga_PartitionType:$result);
  let hasVerifier = 1;
}

def Tiga_HaloOp : Tiga_TaskOp<"halo"> {
  let summary = "Derive halo requirements from a partition and Field accesses";
  let arguments = (ins Tiga_PartitionType:$partition,
                       I64Attr:$depth, ArrayAttr:$fields,
                       StrAttr:$strategy);
  let results = (outs Tiga_HaloPlanType:$result);
  let hasVerifier = 1;
}

def Tiga_HaloPackOp : Tiga_TaskOp<"halo_pack"> {
  let summary = "Pack one owned Field boundary according to a halo plan";
  let arguments = (ins Tiga_HaloPlanType:$plan,
                       Tiga_InstanceType:$source,
                       Variadic<Tiga_EventType>:$depends_on,
                       StrAttr:$field, I64Attr:$bytes,
                       I64Attr:$snapshot_version);
  let results = (outs Tiga_HaloBufferType:$send,
                       Tiga_EventType:$done);
  let hasVerifier = 1;
}

def Tiga_HaloExchangeOp : Tiga_TaskOp<"halo_exchange"> {
  let summary = "Exchange packed halo buffers with graph partition neighbors";
  let arguments = (ins Tiga_HaloPlanType:$plan,
                       Tiga_HaloBufferType:$send,
                       Variadic<Tiga_EventType>:$depends_on,
                       StrAttr:$group, I64Attr:$bytes,
                       I64Attr:$snapshot_version);
  let results = (outs Tiga_HaloBufferType:$receive,
                       Tiga_EventType:$done);
  let hasVerifier = 1;
}

def Tiga_HaloUnpackOp : Tiga_TaskOp<"halo_unpack"> {
  let summary = "Unpack received halo data into a ghost Field instance";
  let arguments = (ins Tiga_HaloPlanType:$plan,
                       Tiga_HaloBufferType:$receive,
                       Tiga_InstanceType:$destination,
                       Variadic<Tiga_EventType>:$depends_on,
                       StrAttr:$field, I64Attr:$snapshot_version);
  let results = (outs Tiga_EventType:$done);
  let hasVerifier = 1;
}

def Tiga_DegreeWorklistOp : Tiga_TaskOp<"degree_worklist"> {
  let summary = "Materialize disjoint destination-row degree buckets";
  let description = [{
    Builds one row-id worklist from a relation snapshot. `upper_bounds` are
    strictly increasing inclusive bounds. Bucket zero also contains degree
    zero; subsequent buckets denote `(upper_bounds[i-1], upper_bounds[i]]`.
    The physical buffer is an explicit non-external `gf_storage.instance`
    with layout `degree-row-worklist`. Its stable ABI is a packed i64 array
    `[bucket_offsets[B+1], bucket_cursors[B], row_ids[rows]]`, so no
    provider-private allocation
    or hidden metadata is needed to launch an individual bucket.
  }];
  let arguments = (ins Tiga_RelationType:$relation,
                       Tiga_InstanceType:$storage,
                       DenseI64ArrayAttr:$upper_bounds,
                       I64Attr:$rows, I64Attr:$snapshot_version);
  let results = (outs Tiga_DegreeWorklistType:$result,
                       Tiga_EventType:$ready);
  let hasVerifier = 1;
}

def Tiga_DegreeHistogramOp : Tiga_TaskOp<"degree_histogram"> {
  let summary = "Count destination rows in each degree bucket";
  let arguments = (ins Tiga_RelationType:$relation,
                       Tiga_InstanceType:$storage,
                       Tiga_EventType:$depends_on,
                       DenseI64ArrayAttr:$upper_bounds,
                       I64Attr:$rows, I64Attr:$snapshot_version);
  let results = (outs Tiga_EventType:$done);
  let hasVerifier = 1;
}

def Tiga_DegreeResetOp : Tiga_TaskOp<"degree_reset"> {
  let summary = "Reset reusable packed degree-worklist counters";
  let arguments = (ins Tiga_InstanceType:$storage,
                       I64Attr:$buckets, I64Attr:$snapshot_version);
  let results = (outs Tiga_EventType:$done);
  let hasVerifier = 1;
}

def Tiga_DegreePrefixOp : Tiga_TaskOp<"degree_prefix"> {
  let summary = "Prefix bucket counts into offsets and reset cursors";
  let arguments = (ins Tiga_InstanceType:$storage,
                       Tiga_EventType:$depends_on,
                       I64Attr:$buckets, I64Attr:$snapshot_version);
  let results = (outs Tiga_EventType:$done);
  let hasVerifier = 1;
}

def Tiga_DegreeScatterOp : Tiga_TaskOp<"degree_scatter"> {
  let summary = "Scatter destination row IDs into compact bucket segments";
  let arguments = (ins Tiga_RelationType:$relation,
                       Tiga_InstanceType:$storage,
                       Tiga_EventType:$depends_on,
                       DenseI64ArrayAttr:$upper_bounds,
                       I64Attr:$rows, I64Attr:$snapshot_version);
  let results = (outs Tiga_DegreeWorklistType:$result,
                       Tiga_EventType:$done);
  let hasVerifier = 1;
}

def Tiga_DegreeBucketLaunchOp
    : Tiga_TaskOp<"degree_bucket_launch"> {
  let summary = "Launch a kernel over one disjoint degree-worklist bucket";
  let description = [{
    `row_mapping = "worklist"` uses a bounded neighbor tile,
    `"direct-filter"` scans the logical row domain while masking other
    buckets, and `"worklist-chunked"` walks multiple bounded neighbor tiles
    for each compact high-degree row while retaining reducer state in the
    generated kernel.
  }];
  let arguments = (ins Tiga_DegreeWorklistType:$worklist,
                       Variadic<Tiga_EventType>:$depends_on,
                       FlatSymbolRefAttr:$callee,
                       I64Attr:$bucket_ordinal,
                       I64Attr:$degree_lower_exclusive,
                       I64Attr:$degree_upper_inclusive,
                       I64Attr:$rows_in_bucket,
                       StrAttr:$row_mapping,
                       ArrayAttr:$arguments,
                       ArrayAttr:$reads, ArrayAttr:$writes,
                       StrAttr:$partitioning, StrAttr:$output_partition,
                       I64Attr:$snapshot_version);
  let results = (outs Tiga_EventType:$done);
  let hasVerifier = 1;
}

def Tiga_RowSplitPartialOp : Tiga_TaskOp<"row_split_partial"> {
  let summary = "Reduce compact high-degree rows into explicit partial state";
  let description = [{
    Maps `(compact row slot, chunk ordinal)` to independent partial reducer
    states. The row slot is resolved through a degree worklist, so low-degree
    rows allocate neither partial state nor GPU programs. The compiler-owned
    storage uses dense row-major layout `[active_row, partial]`; no partial is
    allowed to update the final output.
  }];
  let arguments = (ins Tiga_RelationType:$relation,
                       Tiga_DegreeWorklistType:$worklist,
                       Tiga_InstanceType:$storage,
                       Tiga_EventType:$depends_on,
                       FlatSymbolRefAttr:$callee,
                       I64Attr:$chunk_edges, I64Attr:$partials_per_row,
                       I64Attr:$rows, I64Attr:$active_rows,
                       I64Attr:$bucket_ordinal, I64Attr:$buckets,
                       I64Attr:$state_bytes,
                       ArrayAttr:$arguments,
                       ArrayAttr:$reads, ArrayAttr:$writes,
                       I64Attr:$snapshot_version);
  let results = (outs Tiga_EventType:$done);
  let hasVerifier = 1;
}

def Tiga_RowSplitFinalizeOp : Tiga_TaskOp<"row_split_finalize"> {
  let summary = "Combine row partial states and run reducer/node finalization";
  let description = [{
    Consumes every partial produced for a destination row, combines them using
    the original reducer algebra, then runs finalize and the optional node
    region exactly once before writing the final output.
  }];
  let arguments = (ins Tiga_DegreeWorklistType:$worklist,
                       Tiga_InstanceType:$storage,
                       Tiga_EventType:$depends_on,
                       FlatSymbolRefAttr:$callee,
                       I64Attr:$partials_per_row, I64Attr:$rows,
                       I64Attr:$active_rows,
                       I64Attr:$bucket_ordinal, I64Attr:$buckets,
                       I64Attr:$state_bytes,
                       ArrayAttr:$arguments,
                       ArrayAttr:$reads, ArrayAttr:$writes,
                       StrAttr:$partitioning, StrAttr:$output_partition,
                       I64Attr:$snapshot_version);
  let results = (outs Tiga_EventType:$done);
  let hasVerifier = 1;
}

def Tiga_LaunchOp : Tiga_TaskOp<"launch", [AttrSizedOperandSegments]> {
  let summary = "Launch a local or communication task after asynchronous events";
  let arguments = (ins Optional<Tiga_PartitionType>:$partition,
                       Variadic<Tiga_EventType>:$depends_on,
                       StrAttr:$task_kind,
                       OptionalAttr<FlatSymbolRefAttr>:$callee,
                       ArrayAttr:$reads, ArrayAttr:$writes,
                       I64Attr:$snapshot_version);
  let results = (outs Tiga_EventType:$done);
  let hasVerifier = 1;
}

def Tiga_CollectiveOp : Tiga_TaskOp<"collective"> {
  let summary = "Run a collective over a graph partition after dependencies";
  let arguments = (ins Tiga_PartitionType:$partition,
                       Variadic<Tiga_EventType>:$depends_on,
                       StrAttr:$kind, StrAttr:$group, I64Attr:$bytes,
                       I64Attr:$snapshot_version);
  let results = (outs Tiga_EventType:$done);
  let hasVerifier = 1;
}

#endif
