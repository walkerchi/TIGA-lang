// RUN: gf-opt %s -gf-fuse-compatible-applies | FileCheck %s

// A consumer that reads a finalized producer result crosses a reduction
// barrier.  It must remain a two-stage program.
func.func @dependent(%row: tensor<?xi64>, %col: tensor<?xi64>,
                     %x: tensor<?xf32>) -> tensor<?xf32> {
  %r = "gf.relation"(%row, %col) {
    origin = "external", lifecycle = "frozen",
    realization = "materialized", relation_id = "r0",
    version = 0 : i64, num_src = 16 : i64, num_dst = 16 : i64
  } : (tensor<?xi64>, tensor<?xi64>) -> !gf.relation
  %flux = "gf.apply"(%r, %x) ({
  ^bb0(%src: f32, %dst: f32):
    %message = arith.subf %src, %dst : f32
    "gf.yield"(%message) : (f32) -> ()
  }) {
    reducers = ["sum"], region_kinds = array<i64: 0>,
    input_segment_sizes = array<i64: 1>,
    snapshot_versions = array<i64: 0>, effects = ["read"],
    deterministic = false
  } : (!gf.relation, tensor<?xf32>) -> tensor<?xf32>
  %result = "gf.apply"(%r, %flux) ({
  ^bb0(%src: f32, %dst: f32):
    %message = arith.addf %src, %dst : f32
    "gf.yield"(%message) : (f32) -> ()
  }) {
    reducers = ["sum"], region_kinds = array<i64: 0>,
    input_segment_sizes = array<i64: 1>,
    snapshot_versions = array<i64: 1>, effects = ["read"],
    deterministic = false
  } : (!gf.relation, tensor<?xf32>) -> tensor<?xf32>
  return %result : tensor<?xf32>
}

// CHECK-LABEL: func.func @dependent
// CHECK-COUNT-2: "gf.apply"
// CHECK-NOT: fusion_count

func.func @snapshot_mismatch(%row: tensor<?xi64>, %col: tensor<?xi64>,
                             %x: tensor<?xf32>)
    -> (tensor<?xf32>, tensor<?xf32>) {
  %r = "gf.relation"(%row, %col) {
    origin = "external", lifecycle = "frozen",
    realization = "materialized", relation_id = "r1",
    version = 0 : i64, num_src = 16 : i64, num_dst = 16 : i64
  } : (tensor<?xi64>, tensor<?xi64>) -> !gf.relation
  %a = "gf.apply"(%r, %x) ({
  ^bb0(%src: f32, %dst: f32):
    %message = arith.subf %src, %dst : f32
    "gf.yield"(%message) : (f32) -> ()
  }) {
    reducers = ["sum"], region_kinds = array<i64: 0>,
    input_segment_sizes = array<i64: 1>,
    snapshot_versions = array<i64: 0>, effects = ["read"],
    deterministic = false
  } : (!gf.relation, tensor<?xf32>) -> tensor<?xf32>
  %b = "gf.apply"(%r, %x) ({
  ^bb0(%src: f32, %dst: f32):
    %message = arith.addf %src, %dst : f32
    "gf.yield"(%message) : (f32) -> ()
  }) {
    reducers = ["max"], region_kinds = array<i64: 0>,
    input_segment_sizes = array<i64: 1>,
    snapshot_versions = array<i64: 1>, effects = ["read"],
    deterministic = false
  } : (!gf.relation, tensor<?xf32>) -> tensor<?xf32>
  return %a, %b : tensor<?xf32>, tensor<?xf32>
}

// CHECK-LABEL: func.func @snapshot_mismatch
// CHECK-COUNT-2: "gf.apply"
// CHECK-NOT: fusion_count
