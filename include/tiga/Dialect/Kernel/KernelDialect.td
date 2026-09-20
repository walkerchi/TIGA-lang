#ifndef TIGA_KERNEL_DIALECT_TD
#define TIGA_KERNEL_DIALECT_TD

include "mlir/IR/OpBase.td"

def Tiga_KernelDialect : Dialect {
  let name = "gf_kernel";
  let cppNamespace = "::mlir::tiga::kernel";
  let summary = "Target-independent executable traversal plans for Tiga";
  let description = [{
    The kernel dialect is the stable provider boundary. It fixes traversal,
    masks, local reducer regions and abstract mapping decisions without
    embedding a vendor TTIR layout or machine instruction.
  }];
}

class Tiga_KernelOp<string mnemonic, list<Trait> traits = []>
    : Op<Tiga_KernelDialect, mnemonic, traits>;

#endif
