// RUN: not gf-opt %s -gf-verify-domain 2>&1 | FileCheck %s

func.func @bad_reducer_count(%row: tensor<?xi64>, %col: tensor<?xi64>,
                             %x: tensor<?xf32>) -> tensor<?xf32> {
  %r = "gf.relation"(%row, %col) {
    origin = "external", lifecycle = "frozen",
    realization = "materialized", relation_id = "r0",
    version = 0 : i64, num_src = 16 : i64, num_dst = 16 : i64
  } : (tensor<?xi64>, tensor<?xi64>) -> !gf.relation
  // expected-error @+1 {{requires one reducer per edge region}}
  %0 = "gf.apply"(%r, %x) ({
  ^bb0(%src: f32, %dst: f32):
    %message = arith.subf %src, %dst : f32
    "gf.yield"(%message) : (f32) -> ()
  }) {
    reducers = [], region_kinds = array<i64: 0>,
    input_segment_sizes = array<i64: 1>,
    snapshot_versions = array<i64: 0>, effects = ["read"],
    deterministic = false
  } : (!gf.relation, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK: requires one reducer per edge region
