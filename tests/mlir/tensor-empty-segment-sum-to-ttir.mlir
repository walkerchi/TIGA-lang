// RUN: gf-translate --gf-tensor-to-ttir %s | FileCheck %s
module {
  func.func @empty_sum(%a: tensor<0x2xf32>, %i: tensor<0xi64>) -> tensor<3x2xf32> {
    %x = "gf_tensor.input"(%a) {offset = 0 : i64, strides = array<i64: 2, 1>} : (tensor<0x2xf32>) -> tensor<0x2xf32>
    %index = "gf_tensor.input"(%i) {offset = 0 : i64, strides = array<i64: 1>} : (tensor<0xi64>) -> tensor<0xi64>
    %out = "gf_tensor.segment_sum"(%x, %index) {num_segments = 3 : i64} : (tensor<0x2xf32>, tensor<0xi64>) -> tensor<3x2xf32>
    return %out : tensor<3x2xf32>
  }
}
// CHECK: tiga.tensor entry=gf_tensor_segment_sum block_rows=1 block_elements=1
// CHECK: %edge_bound = arith.constant dense<0>
// CHECK: %valid = arith.cmpi slt
// CHECK: tt.load %input_ptr, %valid, %zero
// CHECK: tt.store
