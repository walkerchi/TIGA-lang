// RUN: not gf-opt %s -split-input-file 2>&1 | FileCheck %s

func.func @self_requires_alias(%query: tensor<?x3xf32>,
                               %candidate: tensor<?x3xf32>) {
  // expected-error @+1 {{a shared entity domain must reuse the same position SSA value}}
  %r = "gf.ranked_relation"(%query, %candidate) {
    num_queries = 8 : i64, num_candidates = 8 : i64, dimensions = 3 : i64,
    k = 2 : i64, metric = "squared_euclidean", selection = "smallest",
    tie_break = "source_index", exclude_self = true,
    same_entity_domain = true, exact = true, version = 0 : i64
  } : (tensor<?x3xf32>, tensor<?x3xf32>) -> !gf.relation
  return
}

// CHECK: a shared entity domain must reuse the same position SSA value

// -----

func.func @k_exceeds_available(%query: tensor<?x3xf32>) {
  // expected-error @+1 {{k exceeds the candidates available to each query}}
  %r = "gf.ranked_relation"(%query, %query) {
    num_queries = 8 : i64, num_candidates = 8 : i64, dimensions = 3 : i64,
    k = 8 : i64, metric = "squared_euclidean", selection = "smallest",
    tie_break = "source_index", exclude_self = true,
    same_entity_domain = true, exact = true, version = 0 : i64
  } : (tensor<?x3xf32>, tensor<?x3xf32>) -> !gf.relation
  return
}

// CHECK: k exceeds the candidates available to each query
