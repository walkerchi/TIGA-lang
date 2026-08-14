// RUN: gf-opt %s -split-input-file -verify-diagnostics

func.func @wrong_body_arity(%initial: tensor<4xf32>, %capture: tensor<f32>)
    -> tensor<4xf32> {
  // expected-error@+1 {{requires one body argument per input}}
  %0 = "gf_control.repeat"(%initial, %capture) <{iterations = 2 : i64}> ({
  ^bb0(%state: tensor<4xf32>):
    "gf_control.yield"(%state) : (tensor<4xf32>) -> ()
  }) : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

func.func @wrong_initial_type(%initial: tensor<4xf32>) -> tensor<5xf32> {
  // expected-error@+1 {{initial and result tensor types must match}}
  %0 = "gf_control.repeat"(%initial) <{iterations = 2 : i64}> ({
  ^bb0(%state: tensor<4xf32>):
    "gf_control.yield"(%state) : (tensor<4xf32>) -> ()
  }) : (tensor<4xf32>) -> tensor<5xf32>
  return %0 : tensor<5xf32>
}
