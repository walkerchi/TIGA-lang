#ifndef TIGA_TRANSFORMS_PASSES_TD
#define TIGA_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def GFVerifyDomain : Pass<"gf-verify-domain", "::mlir::ModuleOp"> {
  let summary = "Verify Tiga domain invariants";
  let description = [{
    Runs operation verification and rejects malformed relation/apply metadata
    before any analysis or fusion pass consumes it.
  }];
}

def GFFormApplyFusionGroups
    : Pass<"gf-form-apply-fusion-groups", "::mlir::ModuleOp"> {
  let summary = "Annotate adjacent applies that are semantically fusible";
  let description = [{
    Uses relation SSA identity, snapshot versions, effect summaries, and SSA
    dependencies to form deterministic candidate groups.  This pass makes no
    target profitability decision and is idempotent.
  }];
  let options = [
    Option<"emitRejections", "emit-rejections", "bool", /*default=*/"false",
           "Emit an optimization remark for every adjacent rejected pair">
  ];
}

def GFFuseCompatibleApplies
    : Pass<"gf-fuse-compatible-applies", "::mlir::ModuleOp"> {
  let summary = "Fuse compatible applies into one multi-region apply";
  let description = [{
    Concatenates independent apply regions over one relation snapshot.  Each
    reducer and result remains independent, giving a product reducer rather
    than reassociating floating-point operations.
  }];
}

def GFLowerDomainToIter
    : Pass<"gf-lower-domain-to-iter", "::mlir::ModuleOp"> {
  let summary = "Resolve Tiga relations into coordinate iteration";
  let description = [{
    Converts logical applies into destination-major compressed-row or
    generated-neighborhood traversal while preserving local regions and
    reducer/effect metadata. No hardware mapping is selected.
  }];
}

def GFLowerIterToKernel
    : Pass<"gf-lower-iter-to-kernel", "::mlir::ModuleOp"> {
  let summary = "Select provider-neutral kernel traversal skeletons";
  let description = [{
    Converts coordinate hierarchies into kernel launch skeletons. Vendor
    subgroup, layout, memory-space encoding and assembly remain unspecified.
  }];
}

def GFSelectKernelSchedule
    : Pass<"gf-select-kernel-schedule", "::mlir::ModuleOp"> {
  let summary = "Select an explicit provider-neutral tile schedule";
  let description = [{
    Annotates gf_kernel launches with the selected row/neighbor tile and warp
    count after iteration lowering.  The decision uses degree bounds and
    scalar field shape, remains inspectable in Kernel IR, and is consumed by
    provider translation rather than being rediscovered in a backend emitter.
  }];
}

def GFPlanDegreeBuckets
    : Pass<"gf-plan-degree-buckets", "::mlir::ModuleOp"> {
  let summary = "Materialize degree worklists and disjoint local task bundles";
  let description = [{
    Consumes the provider-neutral `degree-bucketing-required` schedule decision
    and emits an explicit compiler-owned row-id instance, a degree worklist,
    parallel bucket launches, and a completion join. Provider translation may
    only select a multi-kernel bucket implementation after this physical ABI
    exists in IR.
  }];
}

def GFPlanSplitRows
    : Pass<"gf-plan-split-rows", "::mlir::ModuleOp"> {
  let summary = "Plan compact partial-state reductions for oversized CSR rows";
  let description = [{
    Consumes the high-degree partition of a compiler-owned degree worklist,
    allocates partial state only for compacted tail rows, and replaces that
    partition with a partial/finalize Event chain. The pass only plans the
    currently proven scalar f32 single-state reducer ABI; provider lowering
    must consume this task plan rather than rediscovering row splits.
  }];
}

def GFDecomposeDegreeWorklists
    : Pass<"gf-decompose-degree-worklists", "::mlir::ModuleOp"> {
  let summary = "Lower logical degree-worklist materialization to task primitives";
  let description = [{
    Rewrites each logical materialization to an explicit histogram, small
    prefix/reset and row scatter Event chain. The three tasks share the packed
    `[offsets, cursors, row_ids]` PhysicalInstance and remain provider-neutral.
  }];
}

def GFPlanDistributedTasks
    : Pass<"gf-plan-distributed-tasks", "::mlir::ModuleOp"> {
  let summary = "Derive halo communication and overlap tasks from placement";
  let description = [{
    Finds CSR kernel launches whose source relation carries a distributed
    placement, derives partition and halo plans, and emits an explicit
    pack/exchange/unpack plus interior/boundary Event DAG. The task graph is a
    provider-neutral scheduling sidecar consumed by a distributed runtime.
  }];
}

def GFTensorVJP : Pass<"gf-tensor-vjp", "::mlir::ModuleOp"> {
  let summary = "Lower gf_tensor.grad requests to reverse-mode tensor IR";
  let description = [{
    Constructs a vector-Jacobian product in reverse SSA order. Broadcasting,
    reductions, relation gather/segment-reduce, view transforms and complex
    conjugate-Wirtinger multiplication remain explicit gf_tensor operations
    for later fusion and target lowering.
  }];
}

def GFPlanTensorCheckpoints
    : Pass<"gf-plan-tensor-checkpoints", "::mlir::ModuleOp"> {
  let summary = "Select backward checkpoints under a static memory budget";
  let description = [{
    Rewrites `gf_tensor.checkpoint_candidate` identities to explicit
    `gf_tensor.checkpoint` values when their producer is an irregular gather
    and their static allocation fits the deployment budget. Rejected
    candidates become recomputation identities. The deterministic decision
    vector and selected byte count are attached to the module for runtime
    physicalization and debugging.
  }];
  let options = [
    Option<"memoryBudgetBytes", "memory-budget-bytes", "int64_t",
           /*default=*/"-1",
           "Maximum peak device checkpoint bytes; -1 means no explicit limit">,
    Option<"spillBudgetBytes", "spill-budget-bytes", "int64_t",
           /*default=*/"0",
           "Maximum peak host-pinned spill bytes; -1 means no explicit limit">
  ];
}

def GFLowerTensorToCPU
    : Pass<"gf-lower-tensor-to-cpu", "::mlir::ModuleOp"> {
  let summary = "Lower gf_tensor programs to explicit CPU loops and memrefs";
  let description = [{
    Rewrites a verified static gf_tensor function into a flat storage ABI with
    SCF loops, scalar arithmetic and memref loads/stores. Subsequent standard
    MLIR conversion lowers that program to the LLVM dialect; no C or C++
    source emitter participates in this pipeline.
  }];
}

#endif
