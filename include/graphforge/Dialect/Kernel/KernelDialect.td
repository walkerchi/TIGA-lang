#ifndef GRAPHFORGE_KERNEL_DIALECT_TD
#define GRAPHFORGE_KERNEL_DIALECT_TD

include "mlir/IR/OpBase.td"

def GraphForge_KernelDialect : Dialect {
  let name = "gf_kernel";
  let cppNamespace = "::mlir::graphforge::kernel";
  let summary = "Target-independent executable traversal plans for GraphForge";
  let description = [{
    The kernel dialect is the stable provider boundary. It fixes traversal,
    masks, local reducer regions and abstract mapping decisions without
    embedding a vendor TTIR layout or machine instruction.
  }];
}

class GraphForge_KernelOp<string mnemonic, list<Trait> traits = []>
    : Op<GraphForge_KernelDialect, mnemonic, traits>;

#endif
