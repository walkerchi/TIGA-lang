// RUN: gf-translate --gf-tensor-to-ttir %s | FileCheck %s

// The source IR deliberately contains only ordinary tensor algebra.  This
// regression prevents the linear-recurrence benchmark from depending on a
// workload-named operator or a handwritten Python/Triton kernel.
// CHECK: tiga.tensor entry=gf_tensor_scan_contract
// CHECK-SAME: num_warps=1
// CHECK: tt.func public @gf_tensor_scan_contract
// CHECK: scf.for
// CHECK: tt.reduce
// CHECK: tt.store
// CHECK-NOT: attention
func.func @map_scan_contract(%q: tensor<2x4x3xf32>,
                             %k: tensor<2x4x3xf32>,
                             %v: tensor<2x4x5xf32>) -> tensor<2x4x5xf32> {
  %0 = "gf_tensor.input"(%q) <{offset = 0 : i64,
      strides = array<i64: 12, 3, 1>}>
      : (tensor<2x4x3xf32>) -> tensor<2x4x3xf32>
  %2 = "gf_tensor.reshape"(%0) <{shape = array<i64: 2, 4, 3, 1>}>
      : (tensor<2x4x3xf32>) -> tensor<2x4x3x1xf32>
  %3 = "gf_tensor.broadcast"(%2) <{shape = array<i64: 2, 4, 3, 5>}>
      : (tensor<2x4x3x1xf32>) -> tensor<2x4x3x5xf32>
  %4 = "gf_tensor.input"(%k) <{offset = 0 : i64,
      strides = array<i64: 12, 3, 1>}>
      : (tensor<2x4x3xf32>) -> tensor<2x4x3xf32>
  %6 = "gf_tensor.reshape"(%4) <{shape = array<i64: 2, 4, 3, 1>}>
      : (tensor<2x4x3xf32>) -> tensor<2x4x3x1xf32>
  %7 = "gf_tensor.broadcast"(%6) <{shape = array<i64: 2, 4, 3, 5>}>
      : (tensor<2x4x3x1xf32>) -> tensor<2x4x3x5xf32>
  %8 = "gf_tensor.input"(%v) <{offset = 0 : i64,
      strides = array<i64: 20, 5, 1>}>
      : (tensor<2x4x5xf32>) -> tensor<2x4x5xf32>
  %10 = "gf_tensor.reshape"(%8) <{shape = array<i64: 2, 4, 1, 5>}>
      : (tensor<2x4x5xf32>) -> tensor<2x4x1x5xf32>
  %11 = "gf_tensor.broadcast"(%10) <{shape = array<i64: 2, 4, 3, 5>}>
      : (tensor<2x4x1x5xf32>) -> tensor<2x4x3x5xf32>
  %12 = "gf_tensor.mul"(%7, %11)
      : (tensor<2x4x3x5xf32>, tensor<2x4x3x5xf32>)
        -> tensor<2x4x3x5xf32>
  %13 = "gf_tensor.cumsum"(%12) <{axis = 1 : i64, reverse = false}>
      : (tensor<2x4x3x5xf32>) -> tensor<2x4x3x5xf32>
  %14 = "gf_tensor.mul"(%3, %13)
      : (tensor<2x4x3x5xf32>, tensor<2x4x3x5xf32>)
        -> tensor<2x4x3x5xf32>
  %15 = "gf_tensor.reduce_sum"(%14)
      <{axes = array<i64: 2>, keep_dims = false}>
      : (tensor<2x4x3x5xf32>) -> tensor<2x4x5xf32>
  return %15 : tensor<2x4x5xf32>
}
