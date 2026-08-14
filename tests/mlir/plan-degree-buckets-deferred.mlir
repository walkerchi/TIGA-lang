// RUN: gf-opt %s -gf-lower-domain-to-iter -gf-lower-iter-to-kernel -gf-select-kernel-schedule -gf-plan-degree-buckets | FileCheck %s

// A max degree alone does not reveal how much compact tail storage is needed.
// Keep the semantic launch available for provider fallback instead of emitting
// a task bundle whose >64 bucket has no supported generated kernel.
func.func @unprofiled_tail(%row: tensor<?xi64>, %col: tensor<?xi64>,
                           %x: tensor<?xf32>) -> tensor<?xf32> {
  %r = "gf.relation"(%row, %col) {origin = "external", lifecycle = "frozen",
    realization = "materialized", relation_id = "unprofiled", version = 1 : i64,
    num_src = 128 : i64, num_dst = 128 : i64,
    degree_min = 0 : i64, degree_max = 256 : i64,
    degree_sum = 2048 : i64} :
    (tensor<?xi64>, tensor<?xi64>) -> !gf.relation
  %out = "gf.apply"(%r, %x) ({
  ^bb0(%value: f32):
    "gf.yield"(%value) : (f32) -> ()
  }) {reducers = ["sum"], region_kinds = array<i64: 0>,
    input_segment_sizes = array<i64: 1>, snapshot_versions = array<i64: 1>,
    effects = ["read"], deterministic = false} :
    (!gf.relation, tensor<?xf32>) -> tensor<?xf32>
  return %out : tensor<?xf32>
}

// CHECK-NOT: gf_task.degree_worklist
// CHECK: "gf_kernel.launch"
// CHECK: gf_task.degree_rejected
// CHECK-SAME: load_balance_plan = "provider-deferred-high-degree"
