// RUN: gf-opt --gf-tensor-vjp %s | FileCheck %s

// CHECK-LABEL: func.func @gather_segment_vjp
// CHECK: %[[UP:.+]] = "gf_tensor.gather"(%arg3, %1)
// CHECK: %[[DX:.+]] = "gf_tensor.segment_sum"(%[[UP]], %0) <{num_segments = 3 : i64}>
// CHECK-NOT: "gf_tensor.grad"
// CHECK: return %[[DX]]
func.func @gather_segment_vjp(
    %src: tensor<3xf32>, %src_index: tensor<5xi64>,
    %dst_index: tensor<5xi64>, %cot: tensor<3xf32>) -> tensor<3xf32> {
  %0 = "gf_tensor.input"(%src_index) {
    offset = 0 : i64, strides = array<i64: 1>
  } : (tensor<5xi64>) -> tensor<5xi64>
  %1 = "gf_tensor.input"(%dst_index) {
    offset = 0 : i64, strides = array<i64: 1>
  } : (tensor<5xi64>) -> tensor<5xi64>
  %2 = "gf_tensor.input"(%src) {
    offset = 0 : i64, strides = array<i64: 1>
  } : (tensor<3xf32>) -> tensor<3xf32>
  %3 = "gf_tensor.gather"(%2, %0)
    : (tensor<3xf32>, tensor<5xi64>) -> tensor<5xf32>
  %4 = "gf_tensor.segment_sum"(%3, %1) <{num_segments = 3 : i64}>
    : (tensor<5xf32>, tensor<5xi64>) -> tensor<3xf32>
  %5 = "gf_tensor.grad"(%4, %2, %cot)
    : (tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  return %5 : tensor<3xf32>
}

// CHECK-LABEL: func.func @csr_segment_vjp
// CHECK: %[[DX:.+]] = "gf_tensor.csr_expand_rows"(%arg2, %1) <{num_edges = 5 : i64}>
// CHECK-NOT: "gf_tensor.grad"
// CHECK: return %[[DX]]
func.func @csr_segment_vjp(
    %message: tensor<5xf32>, %row_ptr: tensor<4xi64>,
    %cot: tensor<3xf32>) -> tensor<5xf32> {
  %0 = "gf_tensor.input"(%message) {
    offset = 0 : i64, strides = array<i64: 1>
  } : (tensor<5xf32>) -> tensor<5xf32>
  %1 = "gf_tensor.input"(%row_ptr) {
    offset = 0 : i64, strides = array<i64: 1>
  } : (tensor<4xi64>) -> tensor<4xi64>
  %2 = "gf_tensor.csr_segment_sum"(%0, %1) <{num_rows = 3 : i64}>
    : (tensor<5xf32>, tensor<4xi64>) -> tensor<3xf32>
  %3 = "gf_tensor.grad"(%2, %0, %cot)
    : (tensor<3xf32>, tensor<5xf32>, tensor<3xf32>) -> tensor<5xf32>
  return %3 : tensor<5xf32>
}

// CHECK-LABEL: func.func @division_vjp
// CHECK: "gf_tensor.div"
// CHECK: "gf_tensor.neg"
// CHECK-NOT: "gf_tensor.grad"
func.func @division_vjp(
    %lhs: tensor<4xf32>, %rhs: tensor<4xf32>,
    %cot: tensor<4xf32>) -> tensor<4xf32> {
  %0 = "gf_tensor.input"(%lhs) {
    offset = 0 : i64, strides = array<i64: 1>
  } : (tensor<4xf32>) -> tensor<4xf32>
  %1 = "gf_tensor.input"(%rhs) {
    offset = 0 : i64, strides = array<i64: 1>
  } : (tensor<4xf32>) -> tensor<4xf32>
  %2 = "gf_tensor.div"(%0, %1)
    : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %3 = "gf_tensor.grad"(%2, %1, %cot)
    : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %3 : tensor<4xf32>
}
