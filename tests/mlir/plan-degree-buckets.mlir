// RUN: gf-opt %s -gf-lower-domain-to-iter -gf-lower-iter-to-kernel -gf-select-kernel-schedule -gf-plan-degree-buckets | FileCheck %s

func.func @skewed(%row: tensor<?xi64>, %col: tensor<?xi64>,
                  %x: tensor<?xf32>) -> tensor<?xf32> {
  %r = "gf.relation"(%row, %col) {origin = "external", lifecycle = "frozen",
    realization = "materialized", relation_id = "skewed", version = 3 : i64,
    num_src = 128 : i64, num_dst = 128 : i64,
    degree_min = 0 : i64, degree_max = 64 : i64,
    degree_sum = 1024 : i64} :
    (tensor<?xi64>, tensor<?xi64>) -> !gf.relation
  %out = "gf.apply"(%r, %x) ({
  ^bb0(%value: f32):
    "gf.yield"(%value) : (f32) -> ()
  }) {reducers = ["sum"], region_kinds = array<i64: 0>,
    input_segment_sizes = array<i64: 1>, snapshot_versions = array<i64: 3>,
    effects = ["read"], deterministic = false} :
    (!gf.relation, tensor<?xf32>) -> tensor<?xf32>
  return %out : tensor<?xf32>
}

// CHECK: %[[REGION:.*]] = "gf_storage.region"({{.*}}) <{elements = 137 : i64, logical_id = "skewed.degree-row-ids", version = 3 : i64}>
// CHECK: %[[INSTANCE:.*]] = "gf_storage.instance"(%[[REGION]]) <{capacity_bytes = 1096 : i64, device = "provider:0", external = false, layout = "degree-row-worklist", memory_space = "device"}>
// CHECK: %[[WORKLIST:.*]], %[[READY:.*]] = "gf_task.degree_worklist"({{.*}}, %[[INSTANCE]]) <{rows = 128 : i64, snapshot_version = 3 : i64, upper_bounds = array<i64: 8, 16, 32, 64>}>
// CHECK: %[[B0:.*]] = "gf_task.degree_bucket_launch"(%[[WORKLIST]], %[[READY]]) <{arguments = {{.*}}, bucket_ordinal = 0 : i64, callee = @skewed, degree_lower_exclusive = -1 : i64, degree_upper_inclusive = 8 : i64
// CHECK-SAME: output_partition = "bucket:0", partitioning = "degree-worklist", reads = {{.*}}, row_mapping = "worklist", rows_in_bucket = 128 : i64
// CHECK: %[[B1:.*]] = "gf_task.degree_bucket_launch"(%[[WORKLIST]], %[[READY]]) <{arguments = {{.*}}, bucket_ordinal = 1 : i64
// CHECK: %[[B2:.*]] = "gf_task.degree_bucket_launch"(%[[WORKLIST]], %[[READY]]) <{arguments = {{.*}}, bucket_ordinal = 2 : i64
// CHECK: %[[B3:.*]] = "gf_task.degree_bucket_launch"(%[[WORKLIST]], %[[READY]]) <{arguments = {{.*}}, bucket_ordinal = 3 : i64
// CHECK: "gf_storage.join"(%[[B0]], %[[B1]], %[[B2]], %[[B3]]) <{snapshot_version = 3 : i64}>
