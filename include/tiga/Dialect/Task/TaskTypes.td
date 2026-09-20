#ifndef TIGA_TASK_TYPES_TD
#define TIGA_TASK_TYPES_TD

include "tiga/Dialect/Task/TaskDialect.td"
include "mlir/IR/AttrTypeBase.td"

class Tiga_TaskType<string name, string typeMnemonic,
                          list<Trait> traits = []>
    : TypeDef<Tiga_TaskDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Tiga_PartitionType : Tiga_TaskType<"Partition", "partition"> {
  let summary = "Owned/ghost partition metadata for one graph snapshot";
  let assemblyFormat = "";
}

def Tiga_HaloPlanType : Tiga_TaskType<"HaloPlan", "halo_plan"> {
  let summary = "A derived halo access and exchange plan";
  let assemblyFormat = "";
}

def Tiga_HaloBufferType : Tiga_TaskType<"HaloBuffer", "halo_buffer"> {
  let summary = "A packed send or receive buffer owned by one halo plan";
  let assemblyFormat = "";
}

def Tiga_DegreeWorklistType
    : Tiga_TaskType<"DegreeWorklist", "degree_worklist"> {
  let summary = "A stable partition of destination rows by degree";
  let assemblyFormat = "";
}

#endif
