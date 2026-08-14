// RUN: not gf-opt %s -gf-verify-domain 2>&1 | FileCheck %s

func.func @bad(%row: tensor<?xi64>, %col: tensor<?xi64>) {
  %r = "gf.relation"(%row, %col) {origin = "external", lifecycle = "frozen",
    realization = "materialized", relation_id = "bad", version = 0 : i64,
    num_src = 8 : i64, num_dst = 8 : i64,
    degree_min = 2 : i64, degree_max = 4 : i64,
    degree_sum = 40 : i64} :
    (tensor<?xi64>, tensor<?xi64>) -> !gf.relation
  return
}

// CHECK: degree_sum is inconsistent with row count and bounds
