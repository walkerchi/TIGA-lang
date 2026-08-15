// RUN: gf-opt %s -gf-lower-domain-to-iter -gf-lower-iter-to-kernel -gf-select-kernel-schedule -gf-plan-degree-buckets -gf-plan-split-rows | FileCheck %s

// A scalar additive message has no node epilogue, so the exact degree CDF can
// collapse the power-law plan to two launches: a bounded short-row kernel and
// one register-resident chunked worklist tail.  No partial-state buffer or
// finalize launch is required.
"gf.reducer"() <{sym_name = "additive", kind = "algebraic",
  message_types = [f32], state_types = [f32], result_types = [f32],
  associative = true, commutative = true}> ({
  %zero = arith.constant 0.0 : f32
  "gf.reducer_yield"(%zero) : (f32) -> ()
}, {
^bb0(%message: f32):
  "gf.reducer_yield"(%message) : (f32) -> ()
}, {
^bb0(%left: f32, %right: f32):
  %sum = arith.addf %left, %right : f32
  "gf.reducer_yield"(%sum) : (f32) -> ()
}, {
^bb0(%state: f32):
  "gf.reducer_yield"(%state) : (f32) -> ()
}) : () -> ()

func.func @power_law(%row: tensor<?xi64>, %col: tensor<?xi64>,
                     %x: tensor<?xf32>) -> tensor<?xf32> {
  %r = "gf.relation"(%row, %col) {origin = "external", lifecycle = "frozen",
    realization = "materialized", relation_id = "power-law", version = 9 : i64,
    num_src = 128 : i64, num_dst = 128 : i64,
    degree_min = 8 : i64, degree_max = 256 : i64, degree_sum = 1664 : i64,
    degree_histogram = array<i64:
      0, 0, 0, 0, 0, 0, 0, 0, 120, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1>} :
    (tensor<?xi64>, tensor<?xi64>) -> !gf.relation
  %out = "gf.apply"(%r, %x) ({
  ^bb0(%value: f32):
    "gf.yield"(%value) : (f32) -> ()
  }) {reducers = [@additive], region_kinds = array<i64: 0>,
    input_segment_sizes = array<i64: 1>, snapshot_versions = array<i64: 9>,
    input_roles = ["src"], input_names = ["x"],
    effects = ["read"], deterministic = false} :
    (!gf.relation, tensor<?xf32>) -> tensor<?xf32>
  return %out : tensor<?xf32>
}

// CHECK: %[[WORKLIST:.*]], %[[READY:.*]] = "gf_task.degree_worklist"({{.*}}) <{rows = 128 : i64, snapshot_version = 9 : i64, upper_bounds = array<i64: 8, 256>}>
// CHECK: %[[SHORT:.*]] = "gf_task.degree_bucket_launch"(%[[WORKLIST]], %[[READY]]) <{{.*}}bucket_ordinal = 0 : i64{{.*}}degree_upper_inclusive = 8 : i64{{.*}}row_mapping = "direct-filter"{{.*}}rows_in_bucket = 120 : i64
// CHECK: %[[TAIL:.*]] = "gf_task.degree_bucket_launch"(%[[WORKLIST]], %[[READY]]) <{{.*}}bucket_ordinal = 1 : i64{{.*}}degree_lower_exclusive = 8 : i64{{.*}}degree_upper_inclusive = 256 : i64{{.*}}row_mapping = "worklist-chunked"{{.*}}rows_in_bucket = 8 : i64
// CHECK: "gf_storage.join"(%[[SHORT]], %[[TAIL]])
// CHECK-NOT: "gf_task.row_split_partial"
