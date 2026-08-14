// RUN: gf-opt --gf-tensor-vjp %s | FileCheck %s --check-prefix=VJP

// VJP-LABEL: func.func @scatter_vjp
// VJP: %[[DX:.+]] = "gf_tensor.gather"(%arg3, %0)
// VJP-NOT: "gf_tensor.grad"
// VJP: return %[[DX]]
func.func @scatter_vjp(
    %input: tensor<2x2xf32>, %destination: tensor<2xi64>,
    %inverse: tensor<4xi64>, %cotangent: tensor<4x2xf32>)
    -> tensor<2x2xf32> {
  %0 = "gf_tensor.input"(%destination) <{offset = 0 : i64,
      strides = array<i64: 1>}> : (tensor<2xi64>) -> tensor<2xi64>
  %1 = "gf_tensor.input"(%inverse) <{offset = 0 : i64,
      strides = array<i64: 1>}> : (tensor<4xi64>) -> tensor<4xi64>
  %2 = "gf_tensor.input"(%input) <{offset = 0 : i64,
      strides = array<i64: 2, 1>}> : (tensor<2x2xf32>) -> tensor<2x2xf32>
  %3 = "gf_tensor.scatter_rows"(%2, %0, %1) <{num_rows = 4 : i64}>
      : (tensor<2x2xf32>, tensor<2xi64>, tensor<4xi64>) -> tensor<4x2xf32>
  %4 = "gf_tensor.grad"(%3, %2, %cotangent)
      : (tensor<4x2xf32>, tensor<2x2xf32>, tensor<4x2xf32>) -> tensor<2x2xf32>
  return %4 : tensor<2x2xf32>
}
