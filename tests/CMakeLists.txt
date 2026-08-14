set(
  GRAPHFORGE_LLVM_TEST_TOOLS_DIR "${LLVM_TOOLS_DIR}"
  CACHE PATH
  "Directory containing external FileCheck/not utilities for binary SDKs")

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(GRAPHFORGE_TEST_DEPENDS
  gf-opt
  gf-translate
)
if(TARGET FileCheck)
  list(APPEND GRAPHFORGE_TEST_DEPENDS FileCheck)
endif()

add_lit_testsuite(check-graphforge
  "Running GraphForge MLIR regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${GRAPHFORGE_TEST_DEPENDS}
)
set_target_properties(check-graphforge PROPERTIES FOLDER "GraphForge tests")
