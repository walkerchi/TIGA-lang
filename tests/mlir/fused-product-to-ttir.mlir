// RUN: gf-opt %s -gf-fuse-compatible-applies -gf-lower-domain-to-iter -gf-lower-iter-to-kernel -gf-select-kernel-schedule | gf-translate -gf-kernel-to-ttir | FileCheck %s

"gf.reducer"() <{sym_name = "sum", kind = "algebraic",
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

func.func @horizontal(%row: tensor<?xi64>, %col: tensor<?xi64>,
                      %x: tensor<?xf32>, %w0: tensor<?xf32>,
                      %w1: tensor<?xf32>)
    -> (tensor<?xf32>, tensor<?xf32>) {
  %r = "gf.relation"(%row, %col) {
    origin = "external", lifecycle = "frozen", realization = "materialized",
    relation_id = "r0", version = 0 : i64,
    num_src = 4096 : i64, num_dst = 4096 : i64,
    degree_min = 16 : i64, degree_max = 16 : i64
  } : (tensor<?xi64>, tensor<?xi64>) -> !gf.relation
  %out0 = "gf.apply"(%r, %x, %w0) ({
  ^bb0(%src: f32, %weight: f32):
    %message = arith.mulf %src, %weight : f32
    "gf.yield"(%message) : (f32) -> ()
  }) {reducers = [@sum], region_kinds = array<i64: 0>,
    input_segment_sizes = array<i64: 2>, snapshot_versions = array<i64: 0, 0>,
    input_roles = ["src", "edge"], input_names = ["x", "w0"],
    effects = ["read", "read"], deterministic = false} :
    (!gf.relation, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %out1 = "gf.apply"(%r, %x, %w1) ({
  ^bb0(%src: f32, %weight: f32):
    %message = arith.mulf %src, %weight : f32
    "gf.yield"(%message) : (f32) -> ()
  }) {reducers = [@sum], region_kinds = array<i64: 0>,
    input_segment_sizes = array<i64: 2>, snapshot_versions = array<i64: 0, 0>,
    input_roles = ["src", "edge"], input_names = ["x", "w1"],
    effects = ["read", "read"], deterministic = false} :
    (!gf.relation, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %out0, %out1 : tensor<?xf32>, tensor<?xf32>
}

// CHECK: graphforge.launch entry=gf_csr_product_additive_tile
// CHECK-SAME: abi=row_ptr,col_idx,x,w0,{{x_[0-9]+}},w1,out0,out1
// CHECK: tt.func public @gf_csr_product_additive_tile
// CHECK-SAME: tt.divisibility = 16
// CHECK: %[[COL:.*]] = tt.load %gf_col_ptr
// CHECK-NOT: %gf_input_base2
// CHECK: "tt.reduce"
// CHECK: tt.store %gf_out_ptr0
// CHECK: "tt.reduce"
// CHECK: tt.store %gf_out_ptr1
// CHECK-NOT: @sum
