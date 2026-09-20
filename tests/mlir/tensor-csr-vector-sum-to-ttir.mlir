// RUN: gf-translate --gf-tensor-to-ttir %s | FileCheck %s
module {
  func.func @sum(%a: tensor<131072x16xf32>, %r: tensor<8193xi64>) -> tensor<8192x16xf32> {
    %x = "gf_tensor.input"(%a) {offset = 0 : i64, strides = array<i64: 16, 1>} : (tensor<131072x16xf32>) -> tensor<131072x16xf32>
    %rows = "gf_tensor.input"(%r) {offset = 0 : i64, strides = array<i64: 1>} : (tensor<8193xi64>) -> tensor<8193xi64>
    %out = "gf_tensor.csr_segment_sum"(%x, %rows) {num_rows = 8192 : i64, degree_min = 16 : i64, degree_max = 16 : i64} : (tensor<131072x16xf32>, tensor<8193xi64>) -> tensor<8192x16xf32>
    return %out : tensor<8192x16xf32>
  }
}
// CHECK: tiga.tensor entry=gf_tensor_csr_sum block_rows=1 block_elements=16
// CHECK: dense<0.000000e+00>
// CHECK: arith.divui
// CHECK: arith.addf %a, %b
// CHECK: tt.store
