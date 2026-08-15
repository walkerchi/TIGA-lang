// RUN: gf-opt --gf-lower-tensor-to-cpu %s | FileCheck %s

// CHECK-LABEL: func.func @repeat_multi
// CHECK-SAME: graphforge.cpu.loop_buffers = 4
// CHECK-SAME: graphforge.cpu.loop_scalar_temporaries = 1
// CHECK-SAME: graphforge.cpu.serial_control
// CHECK-COUNT-4: memref.alloc
// CHECK: memref.alloca
// CHECK: scf.for {{.*}} iter_args(
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK-COUNT-4: memref.dealloc
func.func @repeat_multi(%vector: tensor<32xf32>, %scalar: tensor<f32>)
    -> tensor<32xf32> {
  %0 = "gf_tensor.input"(%vector) <{offset = 0 : i64,
      strides = array<i64: 1>}> : (tensor<32xf32>) -> tensor<32xf32>
  %1 = "gf_tensor.input"(%scalar) <{offset = 0 : i64,
      strides = array<i64>}> : (tensor<f32>) -> tensor<f32>
  %2:2 = "gf_control.repeat"(%0, %1)
      <{iterations = 5 : i64, num_carried = 2 : i64}> ({
  ^bb0(%current_vector: tensor<32xf32>, %current_scalar: tensor<f32>):
    %3 = "gf_tensor.add"(%current_vector, %current_scalar)
        : (tensor<32xf32>, tensor<f32>) -> tensor<32xf32>
    %4 = "gf_tensor.mul"(%current_scalar, %current_scalar)
        : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "gf_control.yield"(%3, %4)
        : (tensor<32xf32>, tensor<f32>) -> ()
  }) : (tensor<32xf32>, tensor<f32>) -> (tensor<32xf32>, tensor<f32>)
  return %2#0 : tensor<32xf32>
}
