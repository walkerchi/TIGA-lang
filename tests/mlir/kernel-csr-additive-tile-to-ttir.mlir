// RUN: gf-opt %s -gf-lower-domain-to-iter -gf-lower-iter-to-kernel -gf-select-kernel-schedule | gf-translate -gf-kernel-to-ttir | FileCheck %s

"gf.reducer"() <{sym_name = "anonymous_additive", kind = "algebraic",
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

func.func @bounded(%row: tensor<?xi64>, %col: tensor<?xi64>,
                   %x: tensor<?xf32>, %scale: f32) -> tensor<?xf32> {
  %r = "gf.relation"(%row, %col) {origin = "external", lifecycle = "frozen",
    realization = "materialized", relation_id = "r", version = 0 : i64,
    num_src = 8 : i64, num_dst = 8 : i64,
    degree_min = 16 : i64, degree_max = 16 : i64} :
    (tensor<?xi64>, tensor<?xi64>) -> !gf.relation
  %out = "gf.apply"(%r, %x, %scale) ({
  ^bb0(%value: f32, %factor: f32):
    %message = arith.mulf %value, %factor : f32
    "gf.yield"(%message) : (f32) -> ()
  }) {reducers = [@anonymous_additive], region_kinds = array<i64: 0>,
    input_segment_sizes = array<i64: 2>, snapshot_versions = array<i64: 0, 0>,
    input_roles = ["src", "param"], input_names = ["x", "scale"],
    effects = ["read", "read"], deterministic = false} :
    (!gf.relation, tensor<?xf32>, f32) -> tensor<?xf32>
  return %out : tensor<?xf32>
}

// CHECK: graphforge.launch entry=gf_csr_additive_tile
// CHECK-SAME: block_rows=16
// CHECK: tt.func public @gf_csr_additive_tile
// CHECK: tt.make_range
// CHECK: "tt.reduce"
// CHECK-NOT: anonymous_additive
