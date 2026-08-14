// RUN: gf-opt %s --gf-lower-domain-to-iter | FileCheck %s --check-prefix=ITER
// RUN: gf-opt %s --gf-lower-domain-to-iter --gf-lower-iter-to-kernel | FileCheck %s --check-prefix=KERNEL

module {
  "gf.reducer"() ({
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
  }) {sym_name = "sum_f32", kind = "sum", message_types = [f32],
      state_types = [f32], result_types = [f32],
      associative = true, commutative = true} : () -> ()

  func.func @reduce(%row_ptr: tensor<?xi64>, %col_idx: tensor<?xi64>,
                    %x: tensor<?xf32>, %weight: tensor<?xf32>)
      -> tensor<?xf32> {
    %relation = "gf.relation"(%row_ptr, %col_idx) {
      origin = "external", lifecycle = "frozen", realization = "materialized",
      relation_id = "r", version = 0 : i64,
      num_src = 16 : i64, num_dst = 16 : i64,
      degree_min = 4 : i64, degree_max = 4 : i64
    } : (tensor<?xi64>, tensor<?xi64>) -> !gf.relation
    %out = "gf.apply"(%relation, %x, %weight) ({
    ^bb0(%source: f32, %edge_weight: f32):
      %message = arith.mulf %source, %edge_weight : f32
      "gf.yield"(%message) : (f32) -> ()
    }) {reducers = [@sum_f32], region_kinds = array<i64: 0>,
        input_segment_sizes = array<i64: 2>, snapshot_versions = array<i64: 0, 0>,
        effects = ["read", "read"], deterministic = false}
        : (!gf.relation, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    return %out : tensor<?xf32>
  }
}

// ITER: "gf.reducer"
// ITER: "gf_iter.traverse"
// ITER-SAME: reducers = [@sum_f32]
// KERNEL: "gf.reducer"
// KERNEL: "gf_kernel.launch"
// KERNEL-SAME: reducers = [@sum_f32]
