// RUN: gf-opt %s --gf-lower-domain-to-iter --gf-lower-iter-to-kernel --gf-select-kernel-schedule | FileCheck %s
// RUN: gf-opt %s --gf-lower-domain-to-iter --gf-lower-iter-to-kernel --gf-select-kernel-schedule | gf-translate --gf-kernel-to-ttir | FileCheck %s --check-prefix=PRUNE

module {
  "gf.reducer"() ({
    %neg_inf = arith.constant -1.000000e+30 : f32
    %zero = arith.constant 0.0 : f32
    %zero_v = arith.constant dense<0.0> : vector<64xf32>
    "gf.reducer_yield"(%neg_inf, %zero, %zero_v)
      : (f32, f32, vector<64xf32>) -> ()
  }, {
  ^bb0(%score: f32, %value: vector<64xf16>):
    %one = arith.constant 1.0 : f32
    %value_f32 = arith.extf %value : vector<64xf16> to vector<64xf32>
    "gf.reducer_yield"(%score, %one, %value_f32)
      : (f32, f32, vector<64xf32>) -> ()
  }, {
  ^bb0(%m0: f32, %l0: f32, %o0: vector<64xf32>,
       %m1: f32, %l1: f32, %o1: vector<64xf32>):
    %m = arith.maximumf %m0, %m1 : f32
    %dm0 = arith.subf %m0, %m : f32
    %dm1 = arith.subf %m1, %m : f32
    %a0 = math.exp %dm0 : f32
    %a1 = math.exp %dm1 : f32
    %sl0 = arith.mulf %a0, %l0 : f32
    %sl1 = arith.mulf %a1, %l1 : f32
    %l = arith.addf %sl0, %sl1 : f32
    %a0v = vector.broadcast %a0 : f32 to vector<64xf32>
    %a1v = vector.broadcast %a1 : f32 to vector<64xf32>
    %so0 = arith.mulf %a0v, %o0 : vector<64xf32>
    %so1 = arith.mulf %a1v, %o1 : vector<64xf32>
    %o = arith.addf %so0, %so1 : vector<64xf32>
    "gf.reducer_yield"(%m, %l, %o) : (f32, f32, vector<64xf32>) -> ()
  }, {
  ^bb0(%m: f32, %l: f32, %o: vector<64xf32>):
    %lv = vector.broadcast %l : f32 to vector<64xf32>
    %normalized = arith.divf %o, %lv : vector<64xf32>
    %result = arith.truncf %normalized : vector<64xf32> to vector<64xf16>
    "gf.reducer_yield"(%result) : (vector<64xf16>) -> ()
  }) {sym_name = "stable_weighted_v64", kind = "streaming",
      message_types = [f32, vector<64xf16>],
      state_types = [f32, f32, vector<64xf32>],
      result_types = [vector<64xf16>], associative = true,
      commutative = true,
      block_prune_threshold = 1.562500e-02 : f64} : () -> ()

  func.func @dense_reduction(
      %query: tensor<?x16x64xf16>, %key: tensor<?x16x64xf16>,
      %value: tensor<?x16x64xf16>, %scale: f32) -> tensor<?x16x64xf16> {
    %relation = "gf.cartesian"() {
      num_src = 4096 : i64, num_dst = 4096 : i64, version = 0 : i64,
      boundary = "lower_inclusive"
    } : () -> !gf.relation
    %out = "gf.apply"(%relation, %query, %key, %value, %scale) ({
    ^bb0(%q: vector<64xf16>, %k: vector<64xf16>, %v: vector<64xf16>,
         %s: f32):
      %product = arith.mulf %q, %k : vector<64xf16>
      %score = vector.reduction <add>, %product : vector<64xf16> into f16
      %score_f32 = arith.extf %score : f16 to f32
      %scaled = arith.mulf %score_f32, %s : f32
      "gf.yield"(%scaled, %v) : (f32, vector<64xf16>) -> ()
    }) {reducers = [@stable_weighted_v64], region_kinds = array<i64: 0>,
        input_segment_sizes = array<i64: 4>,
        snapshot_versions = array<i64: 0, 0, 0, 0>,
        effects = ["read", "read", "read", "read"], deterministic = false,
        input_roles = ["dst", "src", "src", "param"],
        iteration_lanes = array<i64: 16>}
        : (!gf.relation, tensor<?x16x64xf16>, tensor<?x16x64xf16>,
           tensor<?x16x64xf16>, f32) -> tensor<?x16x64xf16>
    return %out : tensor<?x16x64xf16>
  }
}

// CHECK: "gf.reducer"
// CHECK: "gf_kernel.dense_launch"
// CHECK-SAME: reducers = [@stable_weighted_v64]
// CHECK-SAME: schedule_kind = "dense-query-key-tile"
// CHECK: vector.reduction <add>
// CHECK: "gf_kernel.yield"(%{{.*}}, %{{.*}}) : (f32, vector<64xf16>) -> ()
// CHECK: boundary = "lower_inclusive"

// PRUNE: tiga.launch entry=gf_dense_streaming_reduce block_rows=64
// PRUNE: tiga.reducer block_prune=0.01562500
// PRUNE-SAME: boundary=lower_inclusive
// PRUNE: %causal_end = arith.minsi
// PRUNE: %[[STATE:.*]]:4 = scf.for
// PRUNE: %boundary_mask = arith.cmpi sle
// PRUNE: %[[SKIP:.*]] = arith.cmpf olt
// PRUNE: %[[ACCEPTED:.*]]:4 = scf.if %[[SKIP]]
// PRUNE: } else {
// PRUNE: %payload_tile = tt.load
// PRUNE: scf.yield %[[ACCEPTED]]#0, %[[ACCEPTED]]#1, %[[ACCEPTED]]#2, %[[ACCEPTED]]#3
