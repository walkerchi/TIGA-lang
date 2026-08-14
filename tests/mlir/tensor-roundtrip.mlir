// RUN: gf-opt %s | FileCheck %s

// CHECK-LABEL: func.func @tensor_program
// CHECK: "gf_tensor.input"
// CHECK: "gf_tensor.mul"
// CHECK: "gf_tensor.add"
// CHECK: "gf_tensor.permute"
// CHECK: "gf_tensor.reshape"
// CHECK: "gf_tensor.broadcast"
// CHECK: "gf_tensor.reduce_sum"
func.func @tensor_program(%arg0: tensor<2x3xf32>,
                          %arg1: tensor<3xf32>) -> tensor<4x1x2xf32> {
  %x = "gf_tensor.input"(%arg0) {
    offset = 0 : i64, strides = array<i64: 3, 1>
  } : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %y = "gf_tensor.input"(%arg1) {
    offset = 0 : i64, strides = array<i64: 1>
  } : (tensor<3xf32>) -> tensor<3xf32>
  %0 = "gf_tensor.mul"(%x, %y)
    : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
  %1 = "gf_tensor.add"(%0, %x)
    : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  %2 = "gf_tensor.permute"(%1) {axes = array<i64: 1, 0>}
    : (tensor<2x3xf32>) -> tensor<3x2xf32>
  %3 = "gf_tensor.reshape"(%2) {shape = array<i64: 3, 2, 1>}
    : (tensor<3x2xf32>) -> tensor<3x2x1xf32>
  %4 = "gf_tensor.broadcast"(%3) {shape = array<i64: 3, 2, 4>}
    : (tensor<3x2x1xf32>) -> tensor<3x2x4xf32>
  %5 = "gf_tensor.reduce_sum"(%4) {
    axes = array<i64: 0, 2>, keep_dims = true
  } : (tensor<3x2x4xf32>) -> tensor<1x2x1xf32>
  %6 = "gf_tensor.reshape"(%5) {shape = array<i64: 1, 2>}
    : (tensor<1x2x1xf32>) -> tensor<1x2xf32>
  %7 = "gf_tensor.broadcast"(%6) {shape = array<i64: 4, 1, 2>}
    : (tensor<1x2xf32>) -> tensor<4x1x2xf32>
  return %7 : tensor<4x1x2xf32>
}

// CHECK-LABEL: func.func @matmul_program
// CHECK: "gf_tensor.matmul"
func.func @matmul_program(%lhs: tensor<5x7xf16>,
                          %rhs: tensor<7x3xf16>) -> tensor<5x3xf16> {
  %0 = "gf_tensor.input"(%lhs) {
    offset = 0 : i64, strides = array<i64: 7, 1>
  } : (tensor<5x7xf16>) -> tensor<5x7xf16>
  %1 = "gf_tensor.input"(%rhs) {
    offset = 0 : i64, strides = array<i64: 3, 1>
  } : (tensor<7x3xf16>) -> tensor<7x3xf16>
  %2 = "gf_tensor.matmul"(%0, %1)
    : (tensor<5x7xf16>, tensor<7x3xf16>) -> tensor<5x3xf16>
  return %2 : tensor<5x3xf16>
}
