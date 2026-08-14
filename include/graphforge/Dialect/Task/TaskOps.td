#ifndef GRAPHFORGE_TASK_OPS_TD
#define GRAPHFORGE_TASK_OPS_TD

include "graphforge/Dialect/Domain/DomainTypes.td"
include "graphforge/Dialect/Storage/StorageTypes.td"
include "graphforge/Dialect/Task/TaskTypes.td"
include "mlir/IR/SymbolInterfaces.td"

def GraphForge_PartitionOp : GraphForge_TaskOp<"partition"> {
  let summary = "Attach destination ownership and ghost cardinality";
  let arguments = (ins GraphForge_RelationType:$relation,
                       StrAttr:$scheme, DenseI64ArrayAttr:$mesh_shape,
                       I64Attr:$mesh_axis, I64Attr:$version,
                       I64Attr:$owned_entities, I64Attr:$ghost_entities);
  let results = (outs GraphForge_PartitionType:$result);
  let hasVerifier = 1;
}

def GraphForge_HaloOp : GraphForge_TaskOp<"halo"> {
  let summary = "Derive halo requirements from a partition and Field accesses";
  let arguments = (ins GraphForge_PartitionType:$partition,
                       I64Attr:$depth, ArrayAttr:$fields,
                       StrAttr:$strategy);
  let results = (outs GraphForge_HaloPlanType:$result);
  let hasVerifier = 1;
}

def GraphForge_HaloPackOp : GraphForge_TaskOp<"halo_pack"> {
  let summary = "Pack one owned Field boundary according to a halo plan";
  let arguments = (ins GraphForge_HaloPlanType:$plan,
                       GraphForge_InstanceType:$source,
                       Variadic<GraphForge_EventType>:$depends_on,
                       StrAttr:$field, I64Attr:$bytes,
                       I64Attr:$snapshot_version);
  let results = (outs GraphForge_HaloBufferType:$send,
                       GraphForge_EventType:$done);
  let hasVerifier = 1;
}

def GraphForge_HaloExchangeOp : GraphForge_TaskOp<"halo_exchange"> {
  let summary = "Exchange packed halo buffers with graph partition neighbors";
  let arguments = (ins GraphForge_HaloPlanType:$plan,
                       GraphForge_HaloBufferType:$send,
                       Variadic<GraphForge_EventType>:$depends_on,
                       StrAttr:$group, I64Attr:$bytes,
                       I64Attr:$snapshot_version);
  let results = (outs GraphForge_HaloBufferType:$receive,
                       GraphForge_EventType:$done);
  let hasVerifier = 1;
}

def GraphForge_HaloUnpackOp : GraphForge_TaskOp<"halo_unpack"> {
  let summary = "Unpack received halo data into a ghost Field instance";
  let arguments = (ins GraphForge_HaloPlanType:$plan,
                       GraphForge_HaloBufferType:$receive,
                       GraphForge_InstanceType:$destination,
                       Variadic<GraphForge_EventType>:$depends_on,
                       StrAttr:$field, I64Attr:$snapshot_version);
  let results = (outs GraphForge_EventType:$done);
  let hasVerifier = 1;
}

def GraphForge_DegreeWorklistOp : GraphForge_TaskOp<"degree_worklist"> {
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
  let arguments = (ins GraphForge_RelationType:$relation,
                       GraphForge_InstanceType:$storage,
                       DenseI64ArrayAttr:$upper_bounds,
                       I64Attr:$rows, I64Attr:$snapshot_version);
  let results = (outs GraphForge_DegreeWorklistType:$result,
                       GraphForge_EventType:$ready);
  let hasVerifier = 1;
}

def GraphForge_DegreeHistogramOp : GraphForge_TaskOp<"degree_histogram"> {
  let summary = "Count destination rows in each degree bucket";
  let arguments = (ins GraphForge_RelationType:$relation,
                       GraphForge_InstanceType:$storage,
                       GraphForge_EventType:$depends_on,
                       DenseI64ArrayAttr:$upper_bounds,
                       I64Attr:$rows, I64Attr:$snapshot_version);
  let results = (outs GraphForge_EventType:$done);
  let hasVerifier = 1;
}

def GraphForge_DegreeResetOp : GraphForge_TaskOp<"degree_reset"> {
  let summary = "Reset reusable packed degree-worklist counters";
  let arguments = (ins GraphForge_InstanceType:$storage,
                       I64Attr:$buckets, I64Attr:$snapshot_version);
  let results = (outs GraphForge_EventType:$done);
  let hasVerifier = 1;
}

def GraphForge_DegreePrefixOp : GraphForge_TaskOp<"degree_prefix"> {
  let summary = "Prefix bucket counts into offsets and reset cursors";
  let arguments = (ins GraphForge_InstanceType:$storage,
                       GraphForge_EventType:$depends_on,
                       I64Attr:$buckets, I64Attr:$snapshot_version);
  let results = (outs GraphForge_EventType:$done);
  let hasVerifier = 1;
}

def GraphForge_DegreeScatterOp : GraphForge_TaskOp<"degree_scatter"> {
  let summary = "Scatter destination row IDs into compact bucket segments";
  let arguments = (ins GraphForge_RelationType:$relation,
                       GraphForge_InstanceType:$storage,
                       GraphForge_EventType:$depends_on,
                       DenseI64ArrayAttr:$upper_bounds,
                       I64Attr:$rows, I64Attr:$snapshot_version);
  let results = (outs GraphForge_DegreeWorklistType:$result,
                       GraphForge_EventType:$done);
  let hasVerifier = 1;
}

def GraphForge_DegreeBucketLaunchOp
    : GraphForge_TaskOp<"degree_bucket_launch"> {
  let summary = "Launch a kernel over one disjoint degree-worklist bucket";
  let description = [{
    `row_mapping = "worklist"` uses a bounded neighbor tile,
    `"direct-filter"` scans the logical row domain while masking other
    buckets, and `"worklist-chunked"` walks multiple bounded neighbor tiles
    for each compact high-degree row while retaining reducer state in the
    generated kernel.
  }];
  let arguments = (ins GraphForge_DegreeWorklistType:$worklist,
                       Variadic<GraphForge_EventType>:$depends_on,
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
  let results = (outs GraphForge_EventType:$done);
  let hasVerifier = 1;
}

def GraphForge_RowSplitPartialOp : GraphForge_TaskOp<"row_split_partial"> {
  let summary = "Reduce compact high-degree rows into explicit partial state";
  let description = [{
    Maps `(compact row slot, chunk ordinal)` to independent partial reducer
    states. The row slot is resolved through a degree worklist, so low-degree
    rows allocate neither partial state nor GPU programs. The compiler-owned
    storage uses dense row-major layout `[active_row, partial]`; no partial is
    allowed to update the final output.
  }];
  let arguments = (ins GraphForge_RelationType:$relation,
                       GraphForge_DegreeWorklistType:$worklist,
                       GraphForge_InstanceType:$storage,
                       GraphForge_EventType:$depends_on,
                       FlatSymbolRefAttr:$callee,
                       I64Attr:$chunk_edges, I64Attr:$partials_per_row,
                       I64Attr:$rows, I64Attr:$active_rows,
                       I64Attr:$bucket_ordinal, I64Attr:$buckets,
                       I64Attr:$state_bytes,
                       ArrayAttr:$arguments,
                       ArrayAttr:$reads, ArrayAttr:$writes,
                       I64Attr:$snapshot_version);
  let results = (outs GraphForge_EventType:$done);
  let hasVerifier = 1;
}

def GraphForge_RowSplitFinalizeOp : GraphForge_TaskOp<"row_split_finalize"> {
  let summary = "Combine row partial states and run reducer/node finalization";
  let description = [{
    Consumes every partial produced for a destination row, combines them using
    the original reducer algebra, then runs finalize and the optional node
    region exactly once before writing the final output.
  }];
  let arguments = (ins GraphForge_DegreeWorklistType:$worklist,
                       GraphForge_InstanceType:$storage,
                       GraphForge_EventType:$depends_on,
                       FlatSymbolRefAttr:$callee,
                       I64Attr:$partials_per_row, I64Attr:$rows,
                       I64Attr:$active_rows,
                       I64Attr:$bucket_ordinal, I64Attr:$buckets,
                       I64Attr:$state_bytes,
                       ArrayAttr:$arguments,
                       ArrayAttr:$reads, ArrayAttr:$writes,
                       StrAttr:$partitioning, StrAttr:$output_partition,
                       I64Attr:$snapshot_version);
  let results = (outs GraphForge_EventType:$done);
  let hasVerifier = 1;
}

def GraphForge_LaunchOp : GraphForge_TaskOp<"launch", [AttrSizedOperandSegments]> {
  let summary = "Launch a local or communication task after asynchronous events";
  let arguments = (ins Optional<GraphForge_PartitionType>:$partition,
                       Variadic<GraphForge_EventType>:$depends_on,
                       StrAttr:$task_kind,
                       OptionalAttr<FlatSymbolRefAttr>:$callee,
                       ArrayAttr:$reads, ArrayAttr:$writes,
                       I64Attr:$snapshot_version);
  let results = (outs GraphForge_EventType:$done);
  let hasVerifier = 1;
}

def GraphForge_CollectiveOp : GraphForge_TaskOp<"collective"> {
  let summary = "Run a collective over a graph partition after dependencies";
  let arguments = (ins GraphForge_PartitionType:$partition,
                       Variadic<GraphForge_EventType>:$depends_on,
                       StrAttr:$kind, StrAttr:$group, I64Attr:$bytes,
                       I64Attr:$snapshot_version);
  let results = (outs GraphForge_EventType:$done);
  let hasVerifier = 1;
}

#endif
