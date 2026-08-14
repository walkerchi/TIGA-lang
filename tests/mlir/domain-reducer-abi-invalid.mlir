// RUN: not gf-opt %s -split-input-file 2>&1 | FileCheck %s

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
  }) {sym_name = "sum_f32", kind = "algebraic", message_types = [f32],
      state_types = [f32], result_types = [f32],
      associative = true, commutative = true} : () -> ()

  func.func @bad_edge_message(%row: tensor<?xi64>, %col: tensor<?xi64>,
                              %x: tensor<?xf64>) -> tensor<?xf32> {
    %relation = "gf.relation"(%row, %col) {
      origin = "external", lifecycle = "frozen", realization = "materialized",
      relation_id = "r", version = 0 : i64, num_src = 4 : i64,
      num_dst = 4 : i64
    } : (tensor<?xi64>, tensor<?xi64>) -> !gf.relation
    // expected-error @+1 {{edge region yielded types do not match reducer message_types}}
    %out = "gf.apply"(%relation, %x) ({
    ^bb0(%value: f64):
      "gf.yield"(%value) : (f64) -> ()
    }) {reducers = [@sum_f32], region_kinds = array<i64: 0>,
        input_segment_sizes = array<i64: 1>, snapshot_versions = array<i64: 0>,
        effects = ["read"], deterministic = false}
        : (!gf.relation, tensor<?xf64>) -> tensor<?xf32>
    return %out : tensor<?xf32>
  }
}

// CHECK: edge region yielded types do not match reducer message_types

// -----

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
  }) {sym_name = "sum_f32", kind = "algebraic", message_types = [f32],
      state_types = [f32], result_types = [f32],
      associative = true, commutative = true} : () -> ()

  func.func @bad_node_aggregate(%row: tensor<?xi64>, %col: tensor<?xi64>,
                                %x: tensor<?xf32>) -> tensor<?xf32> {
    %relation = "gf.relation"(%row, %col) {
      origin = "external", lifecycle = "frozen", realization = "materialized",
      relation_id = "r", version = 0 : i64, num_src = 4 : i64,
      num_dst = 4 : i64
    } : (tensor<?xi64>, tensor<?xi64>) -> !gf.relation
    // expected-error @+1 {{node region must receive the finalized reducer result as its last argument}}
    %out = "gf.apply"(%relation, %x) ({
    ^bb0(%value: f32):
      "gf.yield"(%value) : (f32) -> ()
    }, {
    ^bb0(%wrong_aggregate: f64):
      %zero = arith.constant 0.0 : f32
      "gf.yield"(%zero) : (f32) -> ()
    }) {reducers = [@sum_f32], region_kinds = array<i64: 0, 1>,
        input_segment_sizes = array<i64: 1>, snapshot_versions = array<i64: 0>,
        effects = ["read"], deterministic = false}
        : (!gf.relation, tensor<?xf32>) -> tensor<?xf32>
    return %out : tensor<?xf32>
  }
}

// CHECK: node region must receive the finalized reducer result as its last argument

// -----

module {
  func.func @bad_node_input_index(%row: tensor<?xi64>, %col: tensor<?xi64>,
                                  %x: tensor<?xf32>) -> tensor<?xf32> {
    %relation = "gf.relation"(%row, %col) {
      origin = "external", lifecycle = "frozen", realization = "materialized",
      relation_id = "r", version = 0 : i64, num_src = 4 : i64,
      num_dst = 4 : i64
    } : (tensor<?xi64>, tensor<?xi64>) -> !gf.relation
    // expected-error @+1 {{node input index is outside flattened inputs}}
    %out = "gf.apply"(%relation, %x) ({
    ^bb0(%value: f32):
      "gf.yield"(%value) : (f32) -> ()
    }, {
    ^bb0(%old: f32, %aggregate: f32):
      "gf.yield"(%aggregate) : (f32) -> ()
    }) {reducers = ["sum"], region_kinds = array<i64: 0, 1>,
        input_segment_sizes = array<i64: 1>, snapshot_versions = array<i64: 0>,
        node_input_indices = array<i64: 3>,
        node_input_segment_sizes = array<i64: 1>,
        effects = ["read"], deterministic = false}
        : (!gf.relation, tensor<?xf32>) -> tensor<?xf32>
    return %out : tensor<?xf32>
  }
}

// CHECK: node input index is outside flattened inputs
