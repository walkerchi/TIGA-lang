#ifndef GRAPHFORGE_CONTROL_OPS_TD
#define GRAPHFORGE_CONTROL_OPS_TD

include "graphforge/Dialect/Control/ControlDialect.td"

def GraphForge_RepeatOp : GraphForge_ControlOp<"repeat", [Pure]> {
  let summary = "Repeat one loop-carried tensor region a fixed number of times";
  let description = [{
    Operand zero is the initial loop-carried tensor. Remaining operands are
    immutable captures. The body receives one block argument for every
    operand and yields the next value of operand zero. Keeping captures
    explicit makes storage, distributed ownership, and executable-cache
    dependencies visible to later planning passes.
  }];
  let arguments = (ins Variadic<AnyRankedTensor>:$inputs,
                       ConfinedAttr<I64Attr, [IntNonNegative]>:$iterations);
  let results = (outs AnyRankedTensor:$result);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
}

def GraphForge_ControlYieldOp
    : GraphForge_ControlOp<"yield", [Pure, Terminator]> {
  let arguments = (ins AnyRankedTensor:$value);
  let hasVerifier = 1;
}

#endif
