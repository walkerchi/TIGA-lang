// RUN: gf-opt %s -gf-lower-domain-to-iter -gf-lower-iter-to-kernel -gf-select-kernel-schedule | gf-translate -gf-kernel-to-ttir | FileCheck %s

"gf.reducer"() <{
  sym_name = "arbitrary_tuple_reducer", kind = "algebraic",
  message_types = [f32], state_types = [f32, f32], result_types = [f32],
  associative = true, commutative = true
}> ({
  %zero = arith.constant 0.0 : f32
  %count = arith.constant 0.0 : f32
  "gf.reducer_yield"(%zero, %count) : (f32, f32) -> ()
}, {
^bb0(%message: f32):
  %one = arith.constant 1.0 : f32
  "gf.reducer_yield"(%message, %one) : (f32, f32) -> ()
}, {
^bb0(%left_sum: f32, %left_count: f32,
     %right_sum: f32, %right_count: f32):
  %sum = arith.addf %left_sum, %right_sum : f32
  %count = arith.addf %left_count, %right_count : f32
  "gf.reducer_yield"(%sum, %count) : (f32, f32) -> ()
}, {
^bb0(%sum: f32, %count: f32):
  %mean = arith.divf %sum, %count : f32
  "gf.reducer_yield"(%mean) : (f32) -> ()
}) : () -> ()

func.func @generic_dense(%x: tensor<?xf32>) -> tensor<?xf32> {
  %relation = "gf.cartesian"() {
    num_src = 8 : i64, num_dst = 8 : i64, version = 0 : i64
  } : () -> !gf.relation
  %result = "gf.apply"(%relation, %x) ({
  ^bb0(%value: f32):
    "gf.yield"(%value) : (f32) -> ()
  }) {
    reducers = [@arbitrary_tuple_reducer],
    region_kinds = array<i64: 0>, input_segment_sizes = array<i64: 1>,
    snapshot_versions = array<i64: 0>, effects = ["read"],
    input_roles = ["src"], input_names = ["x"], deterministic = false
  } : (!gf.relation, tensor<?xf32>) -> tensor<?xf32>
  return %result : tensor<?xf32>
}

// CHECK: graphforge.launch entry=gf_dense_scalar_reduce
// CHECK-SAME: abi=x,out
// CHECK: tt.func public @gf_dense_scalar_reduce
// CHECK: %[[STATE:.*]]:2 = scf.for
// CHECK: arith.addf
// CHECK: arith.addf
// CHECK: arith.divf %[[STATE]]#0, %[[STATE]]#1
// CHECK: tt.store
// CHECK-NOT: mean
