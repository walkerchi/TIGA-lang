#ifndef GRAPHFORGE_ITER_OPS_TD
#define GRAPHFORGE_ITER_OPS_TD

include "graphforge/Dialect/Iter/IterDialect.td"
include "graphforge/Dialect/Domain/DomainTypes.td"

def GraphForge_TraverseOp : GraphForge_IterOp<"traverse"> {
  let summary = "Traverse one resolved relation coordinate hierarchy";
  let description = [{
    `coordinate_hierarchy` describes sparse/generated coordinates without
    committing to a thread mapping. `ordering` records the semantic traversal
    order required by the reducer and destination ownership.
  }];
  let arguments = (ins
    GraphForge_RelationType:$relation,
    Variadic<AnyType>:$inputs,
    StrAttr:$coordinate_hierarchy,
    StrAttr:$ordering,
    ArrayAttr:$reducers,
    DenseI64ArrayAttr:$region_kinds,
    DenseI64ArrayAttr:$input_segment_sizes,
    DenseI64ArrayAttr:$snapshot_versions,
    ArrayAttr:$effects,
    BoolAttr:$deterministic,
    OptionalAttr<ArrayAttr>:$input_roles,
    OptionalAttr<ArrayAttr>:$input_names,
    OptionalAttr<DenseI64ArrayAttr>:$node_input_indices,
    OptionalAttr<DenseI64ArrayAttr>:$node_input_segment_sizes,
    OptionalAttr<DenseI64ArrayAttr>:$iteration_lanes
  );
  let results = (outs Variadic<AnyType>:$outputs);
  let regions = (region VariadicRegion<AnyRegion>:$regions);
  let hasVerifier = 1;
}

def GraphForge_YieldOp : GraphForge_IterOp<"yield", [Terminator]> {
  let summary = "Yield a structured message or one node result";
  let arguments = (ins Variadic<AnyType>:$values);
  let hasVerifier = 1;
}

#endif
