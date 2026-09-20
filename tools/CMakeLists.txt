add_subdirectory(gf-opt)
add_subdirectory(gf-translate)

# add_mlir_tool keeps developer tools out of the default target in some LLVM
# distributions. Wheels require both executables, so make that dependency
# explicit without changing the individual tools' upstream-style definitions.
add_custom_target(tiga-python-tools ALL DEPENDS gf-opt gf-translate)
