// RUN: gf-opt --gf-tensor-vjp %s | FileCheck %s

// CHECK-LABEL: func.func @broadcast_mul_vjp
// CHECK: %[[KEEP:.+]] = "gf_tensor.reshape"(%arg2)
// CHECK: %[[UP:.+]] = "gf_tensor.broadcast"(%[[KEEP]])
// CHECK: %[[DX:.+]] = "gf_tensor.mul"(%[[UP]], %1)
// CHECK-NOT: "gf_tensor.grad"
// CHECK: return %[[DX]]
func.func @broadcast_mul_vjp(%arg0: tensor<2x3xf32>,
                             %arg1: tensor<3xf32>,
                             %cot: tensor<2xf32>) -> tensor<2x3xf32> {
  %x = "gf_tensor.input"(%arg0) {
    offset = 0 : i64, strides = array<i64: 3, 1>
  } : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %y = "gf_tensor.input"(%arg1) {
    offset = 0 : i64, strides = array<i64: 1>
  } : (tensor<3xf32>) -> tensor<3xf32>
  %product = "gf_tensor.mul"(%x, %y)
    : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
  %out = "gf_tensor.reduce_sum"(%product) {
    axes = array<i64: 1>, keep_dims = false
  } : (tensor<2x3xf32>) -> tensor<2xf32>
  %dx = "gf_tensor.grad"(%out, %x, %cot)
    : (tensor<2xf32>, tensor<2x3xf32>, tensor<2xf32>) -> tensor<2x3xf32>
  return %dx : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @complex_square_vjp
// CHECK: %[[C0:.+]] = "gf_tensor.conj"(%0)
// CHECK: %[[M0:.+]] = "gf_tensor.mul"(%arg1, %[[C0]])
// CHECK: %[[C1:.+]] = "gf_tensor.conj"(%0)
// CHECK: %[[M1:.+]] = "gf_tensor.mul"(%arg1, %[[C1]])
// CHECK: %[[SUM:.+]] = "gf_tensor.add"(%[[M0]], %[[M1]])
// CHECK: return %[[SUM]]
func.func @complex_square_vjp(%arg0: tensor<4xcomplex<f32>>,
                              %cot: tensor<4xcomplex<f32>>)
    -> tensor<4xcomplex<f32>> {
  %x = "gf_tensor.input"(%arg0) {
    offset = 0 : i64, strides = array<i64: 1>
  } : (tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>>
  %out = "gf_tensor.mul"(%x, %x)
    : (tensor<4xcomplex<f32>>, tensor<4xcomplex<f32>>)
      -> tensor<4xcomplex<f32>>
  %dx = "gf_tensor.grad"(%out, %x, %cot)
    : (tensor<4xcomplex<f32>>, tensor<4xcomplex<f32>>,
       tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>>
  return %dx : tensor<4xcomplex<f32>>
}

// CHECK-LABEL: func.func @matmul_lhs_vjp
// CHECK: %[[BT:.+]] = "gf_tensor.permute"(%1)
// CHECK: %[[DX:.+]] = "gf_tensor.matmul"(%arg2, %[[BT]])
// CHECK-NOT: "gf_tensor.grad"
// CHECK: return %[[DX]]
func.func @matmul_lhs_vjp(%arg0: tensor<2x3xf32>,
                          %arg1: tensor<3x4xf32>,
                          %cot: tensor<2x4xf32>) -> tensor<2x3xf32> {
  %lhs = "gf_tensor.input"(%arg0) {
    offset = 0 : i64, strides = array<i64: 3, 1>
  } : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %rhs = "gf_tensor.input"(%arg1) {
    offset = 0 : i64, strides = array<i64: 4, 1>
  } : (tensor<3x4xf32>) -> tensor<3x4xf32>
  %out = "gf_tensor.matmul"(%lhs, %rhs)
    : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
  %dx = "gf_tensor.grad"(%out, %lhs, %cot)
    : (tensor<2x4xf32>, tensor<2x3xf32>, tensor<2x4xf32>) -> tensor<2x3xf32>
  return %dx : tensor<2x3xf32>
}
