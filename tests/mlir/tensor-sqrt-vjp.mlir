// RUN: gf-opt --gf-tensor-vjp %s | FileCheck %s

// CHECK-LABEL: func.func @sqrt_vjp
// CHECK: %[[ROOT:.+]] = "gf_tensor.sqrt"(%0)
// CHECK: %[[TWICE:.+]] = "gf_tensor.add"(%[[ROOT]], %[[ROOT]])
// CHECK: %[[DX:.+]] = "gf_tensor.div"(%arg1, %[[TWICE]])
// CHECK-NOT: "gf_tensor.grad"
// CHECK: return %[[DX]]
func.func @sqrt_vjp(%input: tensor<4xf32>, %cot: tensor<4xf32>)
    -> tensor<4xf32> {
  %0 = "gf_tensor.input"(%input) {
    offset = 0 : i64, strides = array<i64: 1>
  } : (tensor<4xf32>) -> tensor<4xf32>
  %1 = "gf_tensor.sqrt"(%0) : (tensor<4xf32>) -> tensor<4xf32>
  %2 = "gf_tensor.grad"(%1, %0, %cot)
    : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %2 : tensor<4xf32>
}
