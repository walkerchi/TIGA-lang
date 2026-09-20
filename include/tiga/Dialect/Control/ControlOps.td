#ifndef TIGA_CONTROL_OPS_TD
#define TIGA_CONTROL_OPS_TD

include "tiga/Dialect/Control/ControlDialect.td"

def Tiga_RepeatOp : Tiga_ControlOp<"repeat", [Pure]> {
  let summary = "Repeat a multi-state tensor region a fixed number of times";
  let description = [{
    The first `num_carried` operands are initial loop-carried tensors and have
    one matching result each. Remaining operands are immutable captures. The
    body receives one block argument for every operand and yields one next
    value per carried tensor. Keeping captures explicit makes storage,
    distributed ownership, and executable-cache dependencies visible to later
    planning passes.
  }];
  let arguments = (ins Variadic<AnyRankedTensor>:$inputs,
                       ConfinedAttr<I64Attr, [IntPositive]>:$num_carried,
                       ConfinedAttr<I64Attr, [IntNonNegative]>:$iterations);
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
}

def Tiga_WhileOp : Tiga_ControlOp<"while", [Pure]> {
  let summary = "Run a multi-state tensor region under a bounded condition";
  let description = [{
    The first `num_carried` operands are loop-carried tensors and have one
    matching result each. Remaining operands are immutable captures. Both
    regions receive every operand as a block argument. The condition must
    terminate with a rank-zero tensor<i1> `gf_control.condition`; the body
    yields one next value per carried tensor. `max_iterations` is mandatory,
    making resource use finite even when data-dependent convergence fails.
  }];
  let arguments = (ins Variadic<AnyRankedTensor>:$inputs,
                       ConfinedAttr<I64Attr, [IntPositive]>:$num_carried,
                       ConfinedAttr<I64Attr, [IntNonNegative]>:$max_iterations);
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let regions = (region SizedRegion<1>:$condition,
                        SizedRegion<1>:$body);
  let hasVerifier = 1;
}

def Tiga_ConditionOp
    : Tiga_ControlOp<"condition", [Pure, Terminator]> {
  let arguments = (ins AnyRankedTensor:$value);
  let hasVerifier = 1;
}

def Tiga_ControlYieldOp
    : Tiga_ControlOp<"yield", [Pure, Terminator]> {
  let arguments = (ins Variadic<AnyRankedTensor>:$values);
  let hasVerifier = 1;
}

#endif
