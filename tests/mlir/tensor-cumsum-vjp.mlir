// RUN: gf-opt --gf-tensor-vjp %s | FileCheck %s --check-prefix=VJP

// VJP-LABEL: func.func @cumsum_vjp
// VJP: %[[SCAN:.+]] = "gf_tensor.cumsum"(%0) <{axis = 1 : i64, reverse = false}>
// VJP: %[[DX:.+]] = "gf_tensor.cumsum"(%arg1) <{axis = 1 : i64, reverse = true}>
// VJP-NOT: "gf_tensor.grad"
// VJP: return %[[DX]]
func.func @cumsum_vjp(%input: tensor<3x4xf32>, %cot: tensor<3x4xf32>)
    -> tensor<3x4xf32> {
  %0 = "gf_tensor.input"(%input) {
    offset = 0 : i64, strides = array<i64: 4, 1>
  } : (tensor<3x4xf32>) -> tensor<3x4xf32>
  %1 = "gf_tensor.cumsum"(%0) <{axis = 1 : i64, reverse = false}>
    : (tensor<3x4xf32>) -> tensor<3x4xf32>
  %2 = "gf_tensor.grad"(%1, %0, %cot)
    : (tensor<3x4xf32>, tensor<3x4xf32>, tensor<3x4xf32>)
      -> tensor<3x4xf32>
  return %2 : tensor<3x4xf32>
}
