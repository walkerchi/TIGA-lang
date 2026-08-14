#ifndef GRAPHFORGE_DOMAIN_TYPES_TD
#define GRAPHFORGE_DOMAIN_TYPES_TD

include "graphforge/Dialect/Domain/DomainDialect.td"
include "mlir/IR/AttrTypeBase.td"

class GraphForge_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<GraphForge_DomainDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def GraphForge_RelationType : GraphForge_Type<"Relation", "relation"> {
  let summary = "A versioned logical binary relation handle";
  let description = [{
    The runtime indices are operands of `gf.relation`; this type carries only
    the fact that an SSA value is a relation handle.  Origin, lifecycle,
    realization, endpoint cardinalities, and version remain attributes on the
    defining operation in M0.
  }];
  let assemblyFormat = "";
}

def GraphForge_ReducerType : GraphForge_Type<"Reducer", "reducer"> {
  let summary = "A symbolic reducer algebra definition";
  let assemblyFormat = "";
}

#endif
