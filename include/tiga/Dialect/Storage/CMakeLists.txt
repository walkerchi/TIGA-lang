add_mlir_dialect(StorageOps gf_storage)
add_mlir_doc(StorageOps TigaStorage Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS StorageTypes.td)
mlir_tablegen(StorageTypes.h.inc -gen-typedef-decls)
mlir_tablegen(StorageTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(TigaStorageTypesIncGen)
