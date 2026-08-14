// RUN: gf-opt %s -gf-lower-domain-to-iter | FileCheck %s --check-prefix=ITER
// RUN: gf-opt %s -gf-lower-domain-to-iter -gf-lower-iter-to-kernel | FileCheck %s --check-prefix=KERNEL
// RUN: gf-opt %s -gf-lower-domain-to-iter -gf-lower-iter-to-kernel -gf-select-kernel-schedule | FileCheck %s --check-prefix=SCHEDULE

func.func @weighted_sum(%row: tensor<?xi64>, %col: tensor<?xi64>,
                        %x: tensor<?xf32>) -> tensor<?xf32> {
  %r = "gf.relation"(%row, %col) {
    origin = "external", lifecycle = "frozen",
    realization = "materialized", relation_id = "r0",
    version = 0 : i64, num_src = 16 : i64, num_dst = 16 : i64
  } : (tensor<?xi64>, tensor<?xi64>) -> !gf.relation
  %0 = "gf.apply"(%r, %x) ({
  ^bb0(%src: f32, %dst: f32):
    %message = arith.mulf %src, %dst : f32
    "gf.yield"(%message) : (f32) -> ()
  }) {
    reducers = ["sum"], region_kinds = array<i64: 0>,
    input_segment_sizes = array<i64: 1>,
    snapshot_versions = array<i64: 0>, effects = ["read"],
    deterministic = false
  } : (!gf.relation, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// ITER: "gf_iter.traverse"
// ITER-SAME: coordinate_hierarchy = "compressed-row"
// ITER-SAME: ordering = "destination-major"
// ITER: arith.mulf
// ITER: "gf_iter.yield"
// ITER-NOT: "gf.apply"

// KERNEL: "gf_kernel.launch"
// KERNEL-SAME: traversal = "csr-row"
// KERNEL: arith.mulf
// KERNEL: "gf_kernel.yield"
// KERNEL-NOT: "gf_iter.traverse"
// KERNEL-NOT: "gf.apply"

// SCHEDULE: "gf_kernel.launch"
// SCHEDULE: block_neighbors = 1 : i64
// SCHEDULE-SAME: block_rows = 1 : i64
// SCHEDULE-SAME: num_warps = 4 : i64
// SCHEDULE-SAME: pipeline_stages = 1 : i64
// SCHEDULE-SAME: schedule_handoffs = ["load.global-to-register", "store.register-to-global"]
// SCHEDULE-SAME: schedule_kind = "scalar-row-loop"
// SCHEDULE-SAME: schedule_resources = ["relation.global.read", "input.global.read", "state.register.private", "output.global.write"]
// SCHEDULE-SAME: target_contract = "provider-neutral-v1"
