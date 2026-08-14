#ifndef GRAPHFORGE_TASK_TYPES_TD
#define GRAPHFORGE_TASK_TYPES_TD

include "graphforge/Dialect/Task/TaskDialect.td"
include "mlir/IR/AttrTypeBase.td"

class GraphForge_TaskType<string name, string typeMnemonic,
                          list<Trait> traits = []>
    : TypeDef<GraphForge_TaskDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def GraphForge_PartitionType : GraphForge_TaskType<"Partition", "partition"> {
  let summary = "Owned/ghost partition metadata for one graph snapshot";
  let assemblyFormat = "";
}

def GraphForge_HaloPlanType : GraphForge_TaskType<"HaloPlan", "halo_plan"> {
  let summary = "A derived halo access and exchange plan";
  let assemblyFormat = "";
}

def GraphForge_HaloBufferType : GraphForge_TaskType<"HaloBuffer", "halo_buffer"> {
  let summary = "A packed send or receive buffer owned by one halo plan";
  let assemblyFormat = "";
}

def GraphForge_DegreeWorklistType
    : GraphForge_TaskType<"DegreeWorklist", "degree_worklist"> {
  let summary = "A stable partition of destination rows by degree";
  let assemblyFormat = "";
}

#endif
