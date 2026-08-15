// RUN: gf-opt %s -gf-lower-domain-to-iter -gf-lower-iter-to-kernel -gf-select-kernel-schedule -gf-plan-degree-buckets -gf-plan-split-rows | FileCheck %s --check-prefix=IR
// RUN: gf-opt %s -gf-lower-domain-to-iter -gf-lower-iter-to-kernel -gf-select-kernel-schedule -gf-plan-degree-buckets -gf-plan-split-rows | gf-translate -gf-task-to-bundle | FileCheck %s --check-prefix=PLAN

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

func.func @heavy(%row: tensor<?xi64>, %col: tensor<?xi64>,
                 %x: tensor<?xf32>) -> tensor<?xf32> {
  %r = "gf.relation"(%row, %col) {origin = "external", lifecycle = "frozen",
    realization = "materialized", relation_id = "heavy", version = 5 : i64,
    num_src = 128 : i64, num_dst = 128 : i64,
    degree_min = 8 : i64, degree_max = 256 : i64,
    degree_sum = 1664 : i64,
    source_index_span_ratio = 3.300000e-01 : f64,
    degree_histogram = array<i64:
      0, 0, 0, 0, 0, 0, 0, 0, 120,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1>} :
    (tensor<?xi64>, tensor<?xi64>) -> !gf.relation
  %out = "gf.apply"(%r, %x) ({
  ^bb0(%value: f32):
    "gf.yield"(%value) : (f32) -> ()
  }, {
  ^bb0(%aggregate: f32):
    "gf.yield"(%aggregate) : (f32) -> ()
  }) {reducers = [@additive], region_kinds = array<i64: 0, 1>,
    input_segment_sizes = array<i64: 1>, snapshot_versions = array<i64: 5>,
    input_roles = ["src"], input_names = ["x"],
    node_input_indices = array<i64>,
    node_input_segment_sizes = array<i64: 0>,
    effects = ["read"], deterministic = false} :
    (!gf.relation, tensor<?xf32>) -> tensor<?xf32>
  return %out : tensor<?xf32>
}

// IR: %[[WORKLIST:.*]], %[[READY:.*]] = "gf_task.degree_worklist"({{.*}}) <{rows = 128 : i64, snapshot_version = 5 : i64, upper_bounds = array<i64: 8, 64, 256>}>
// IR: %[[SHORT:.*]] = "gf_task.degree_bucket_launch"(%[[WORKLIST]], %[[READY]]) <{{.*}}degree_upper_inclusive = 8 : i64, output_partition = "bucket:0"
// IR: %[[MIDDLE:.*]] = "gf_task.degree_bucket_launch"(%[[WORKLIST]], %[[READY]]) <{{.*}}degree_lower_exclusive = 8 : i64, degree_upper_inclusive = 64 : i64, output_partition = "bucket:1"
// IR: %[[REGION:.*]] = "gf_storage.region"({{.*}}) <{elements = 4 : i64, logical_id = "heavy.row-partials", version = 5 : i64}>
// IR: %[[INSTANCE:.*]] = "gf_storage.instance"(%[[REGION]]) <{capacity_bytes = 16 : i64, device = "provider:0", external = false, layout = "row-partial-f32", memory_space = "device"}>
// IR: %[[PARTIAL:.*]] = "gf_task.row_split_partial"({{.*}}, %[[WORKLIST]], %[[INSTANCE]], %[[READY]]) <{active_rows = 1 : i64, arguments = {{.*}}, bucket_ordinal = 2 : i64, buckets = 3 : i64, callee = @heavy, chunk_edges = 64 : i64, partials_per_row = 4 : i64
// IR-SAME: reads = ["row_ptr", "col_idx", "src:x", "row_worklist"], rows = 128 : i64, snapshot_version = 5 : i64, state_bytes = 4 : i64, writes = ["row_partials"]
// IR: %[[FINAL:.*]] = "gf_task.row_split_finalize"(%[[WORKLIST]], %[[INSTANCE]], %[[PARTIAL]]) <{active_rows = 1 : i64, arguments = {{.*}}, bucket_ordinal = 2 : i64, buckets = 3 : i64, callee = @heavy, output_partition = "bucket:2", partials_per_row = 4 : i64, partitioning = "degree-worklist", reads = ["row_worklist", "row_partials"]
// IR-SAME: writes = ["output"]
// IR: "gf_storage.release"(%[[INSTANCE]], %[[FINAL]]) <{snapshot_version = 5 : i64}>
// IR: "gf_storage.join"(%[[SHORT]], %[[MIDDLE]], %[[FINAL]])
// IR: load_balance_plan = "high-degree-tail-split"

// PLAN: "chunk_edges": 64
// PLAN: "partials_per_row": 4
// PLAN: "task_kind": "row-split-partial"
// PLAN: "active_rows": 1
// PLAN: "grid_x": 1
// PLAN: "task_kind": "row-split-finalize"
// PLAN: "task_kind": "release"
// PLAN: "capacity_bytes": 16
// PLAN: "layout": "row-partial-f32"
// PLAN: "name": "row_partials"
// PLAN: "terminals": [
// PLAN-NEXT: {{ *}}"release:0"
