// RUN: gf-opt %s -gf-verify-domain -split-input-file | FileCheck %s

func.func @valid(%row: tensor<?xi64>, %col: tensor<?xi64>,
                 %x: tensor<?xf32>) -> tensor<?xf32> {
  %r = "gf.relation"(%row, %col) {
    origin = "external", lifecycle = "frozen",
    realization = "materialized", relation_id = "r0",
    version = 0 : i64, num_src = 16 : i64, num_dst = 16 : i64
  } : (tensor<?xi64>, tensor<?xi64>) -> !gf.relation
  %0 = "gf.apply"(%r, %x) ({
  ^bb0(%src: f32, %dst: f32):
    %message = arith.subf %src, %dst : f32
    "gf.yield"(%message) : (f32) -> ()
  }) {
    reducers = ["sum"], region_kinds = array<i64: 0>,
    input_segment_sizes = array<i64: 1>,
    snapshot_versions = array<i64: 0>, effects = ["read"],
    deterministic = false
  } : (!gf.relation, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK: "gf.relation"
// CHECK: "gf.apply"

func.func @valid_optional_node(%row: tensor<?xi64>, %col: tensor<?xi64>,
                               %x: tensor<?xf32>) -> tensor<?xf32> {
  %r = "gf.relation"(%row, %col) {
    origin = "external", lifecycle = "frozen",
    realization = "materialized", relation_id = "r-node",
    version = 0 : i64, num_src = 16 : i64, num_dst = 16 : i64
  } : (tensor<?xi64>, tensor<?xi64>) -> !gf.relation
  %0 = "gf.apply"(%r, %x) ({
  ^bb0(%src: f32, %dst: f32):
    %message = arith.subf %src, %dst : f32
    "gf.yield"(%message) : (f32) -> ()
  }, {
  ^bb0(%old: f32, %aggregate: f32):
    %updated = arith.addf %old, %aggregate : f32
    "gf.yield"(%updated) : (f32) -> ()
  }) {
    reducers = ["sum"], region_kinds = array<i64: 0, 1>,
    input_segment_sizes = array<i64: 1>,
    snapshot_versions = array<i64: 0>, effects = ["read"],
    deterministic = false
  } : (!gf.relation, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK: region_kinds = array<i64: 0, 1>
