// RUN: gf-opt --gf-lower-tensor-to-cpu %s | FileCheck %s

// CHECK-LABEL: func.func @repeat_pointwise
// CHECK-SAME: tiga.cpu.loop_buffers = 2
// CHECK-SAME: tiga.cpu.serial_control
// CHECK-COUNT-2: memref.alloc
// CHECK: scf.for
// CHECK: scf.for {{.*}} iter_args(
// CHECK: scf.for
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: scf.yield
// CHECK: memref.dealloc
// CHECK: memref.dealloc
func.func @repeat_pointwise(%initial: tensor<32xf32>, %scale: tensor<f32>,
                            %bias: tensor<f32>) -> tensor<32xf32> {
  %0 = "gf_tensor.input"(%initial) <{offset = 0 : i64,
      strides = array<i64: 1>}> : (tensor<32xf32>) -> tensor<32xf32>
  %1 = "gf_tensor.input"(%scale) <{offset = 0 : i64,
      strides = array<i64>}> : (tensor<f32>) -> tensor<f32>
  %2 = "gf_tensor.input"(%bias) <{offset = 0 : i64,
      strides = array<i64>}> : (tensor<f32>) -> tensor<f32>
  %3 = "gf_control.repeat"(%0, %1, %2) <{iterations = 7 : i64,
      num_carried = 1 : i64}> ({
  ^bb0(%state: tensor<32xf32>, %captured_scale: tensor<f32>,
       %captured_bias: tensor<f32>):
    %4 = "gf_tensor.mul"(%state, %captured_scale)
        : (tensor<32xf32>, tensor<f32>) -> tensor<32xf32>
    %5 = "gf_tensor.add"(%4, %captured_bias)
        : (tensor<32xf32>, tensor<f32>) -> tensor<32xf32>
    "gf_control.yield"(%5) : (tensor<32xf32>) -> ()
  }) : (tensor<32xf32>, tensor<f32>, tensor<f32>) -> tensor<32xf32>
  return %3 : tensor<32xf32>
}
