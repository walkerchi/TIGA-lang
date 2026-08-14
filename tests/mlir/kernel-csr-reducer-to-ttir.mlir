// RUN: gf-opt %s -gf-lower-domain-to-iter -gf-lower-iter-to-kernel -gf-select-kernel-schedule | gf-translate -gf-kernel-to-ttir | FileCheck %s

"gf.reducer"() <{sym_name = "tuple_algebra", kind = "algebraic",
  message_types = [f32], state_types = [f32, f32], result_types = [f32],
  associative = true, commutative = true}> ({
  %a = arith.constant 0.0 : f32
  %b = arith.constant 0.0 : f32
  "gf.reducer_yield"(%a, %b) : (f32, f32) -> ()
}, {
^bb0(%x: f32):
  %one = arith.constant 1.0 : f32
  "gf.reducer_yield"(%x, %one) : (f32, f32) -> ()
}, {
^bb0(%a: f32, %b: f32, %c: f32, %d: f32):
  %sum = arith.addf %a, %c : f32
  %count = arith.addf %b, %d : f32
  "gf.reducer_yield"(%sum, %count) : (f32, f32) -> ()
}, {
^bb0(%sum: f32, %count: f32):
  %mean = arith.divf %sum, %count : f32
  "gf.reducer_yield"(%mean) : (f32) -> ()
}) : () -> ()

func.func @csr_tuple(%row: tensor<?xi64>, %col: tensor<?xi64>,
                     %x: tensor<?xf32>) -> tensor<?xf32> {
  %r = "gf.relation"(%row, %col) {origin = "external", lifecycle = "frozen",
    realization = "materialized", relation_id = "r", version = 0 : i64,
    num_src = 8 : i64, num_dst = 8 : i64,
    degree_min = 2 : i64, degree_max = 2 : i64} :
    (tensor<?xi64>, tensor<?xi64>) -> !gf.relation
  %out = "gf.apply"(%r, %x) ({
  ^bb0(%value: f32):
    "gf.yield"(%value) : (f32) -> ()
  }) {reducers = [@tuple_algebra], region_kinds = array<i64: 0>,
    input_segment_sizes = array<i64: 1>, snapshot_versions = array<i64: 0>,
    input_roles = ["src"], input_names = ["x"], effects = ["read"],
    deterministic = true} :
    (!gf.relation, tensor<?xf32>) -> tensor<?xf32>
  return %out : tensor<?xf32>
}

// CHECK: graphforge.launch entry=gf_csr_scalar_reduce
// CHECK: tt.func public @gf_csr_scalar_reduce
// CHECK: %[[STATE:.*]]:2 = scf.for
// CHECK: tt.load %gf_col_ptr
// CHECK: arith.divf %[[STATE]]#0, %[[STATE]]#1
// CHECK-NOT: tuple_algebra

// A deterministic request must preserve the serial ordering even though the
// relation is bounded and the tuple state is structurally zero/additive.
