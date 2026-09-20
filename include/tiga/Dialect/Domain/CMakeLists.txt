add_mlir_dialect(DomainOps gf)
add_mlir_doc(DomainOps TigaDomain Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS DomainTypes.td)
mlir_tablegen(DomainTypes.h.inc -gen-typedef-decls)
mlir_tablegen(DomainTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(TigaDomainTypesIncGen)

