// RUN: gf-opt --gf-lower-tensor-to-cpu %s | FileCheck %s

// CHECK-LABEL: func.func @scatter_forward
// CHECK: scf.if
// CHECK: memref.load
// CHECK: arith.constant 0.000000e+00 : f32
// CHECK: memref.store
func.func @scatter_forward(
    %input: tensor<2x2xf32>, %destination: tensor<2xi64>,
    %inverse: tensor<4xi64>) -> tensor<4x2xf32> {
  %0 = "gf_tensor.input"(%destination) <{offset = 0 : i64,
      strides = array<i64: 1>}> : (tensor<2xi64>) -> tensor<2xi64>
  %1 = "gf_tensor.input"(%inverse) <{offset = 0 : i64,
      strides = array<i64: 1>}> : (tensor<4xi64>) -> tensor<4xi64>
  %2 = "gf_tensor.input"(%input) <{offset = 0 : i64,
      strides = array<i64: 2, 1>}> : (tensor<2x2xf32>) -> tensor<2x2xf32>
  %3 = "gf_tensor.scatter_rows"(%2, %0, %1) <{num_rows = 4 : i64}>
      : (tensor<2x2xf32>, tensor<2xi64>, tensor<4xi64>) -> tensor<4x2xf32>
  return %3 : tensor<4x2xf32>
}
