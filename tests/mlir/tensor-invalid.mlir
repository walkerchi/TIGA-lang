// RUN: gf-opt %s -split-input-file -verify-diagnostics

func.func @bad_broadcast(%lhs: tensor<2x3xf32>,
                         %rhs: tensor<4xf32>) -> tensor<2x4xf32> {
  // expected-error@+1 {{operands are not broadcast-compatible}}
  %0 = "gf_tensor.add"(%lhs, %rhs)
    : (tensor<2x3xf32>, tensor<4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

func.func @bad_reshape(%input: tensor<2x3xf32>) -> tensor<7xf32> {
  // expected-error@+1 {{cannot change the number of elements}}
  %0 = "gf_tensor.reshape"(%input) {shape = array<i64: 7>}
    : (tensor<2x3xf32>) -> tensor<7xf32>
  return %0 : tensor<7xf32>
}

// -----

func.func @bad_permute(%input: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // expected-error@+1 {{axes must be a permutation}}
  %0 = "gf_tensor.permute"(%input) {axes = array<i64: 0, 0>}
    : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// -----

func.func @bad_reduce(%input: tensor<2x3xf32>) -> tensor<2xf32> {
  // expected-error@+1 {{axes must be unique}}
  %0 = "gf_tensor.reduce_sum"(%input) {
    axes = array<i64: 1, 1>, keep_dims = false
  } : (tensor<2x3xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// -----

func.func @bad_matmul(%lhs: tensor<2x3xf32>,
                      %rhs: tensor<4x5xf32>) -> tensor<2x5xf32> {
  // expected-error@+1 {{contraction dimensions must match}}
  %0 = "gf_tensor.matmul"(%lhs, %rhs)
    : (tensor<2x3xf32>, tensor<4x5xf32>) -> tensor<2x5xf32>
  return %0 : tensor<2x5xf32>
}
