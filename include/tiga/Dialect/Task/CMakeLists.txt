add_mlir_dialect(TaskOps gf_task)
add_mlir_doc(TaskOps TigaTask Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS TaskTypes.td)
mlir_tablegen(TaskTypes.h.inc -gen-typedef-decls)
mlir_tablegen(TaskTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(TigaTaskTypesIncGen)
