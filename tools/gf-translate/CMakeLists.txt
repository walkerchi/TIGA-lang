add_mlir_tool(gf-translate gf-translate.cpp)

set_target_properties(gf-translate PROPERTIES
  INSTALL_RPATH "${GRAPHFORGE_TOOL_RPATH}")

target_link_libraries(gf-translate PRIVATE
  GraphForgeTargetTriton
  MLIRTranslateLib
)

install(
  TARGETS gf-translate
  RUNTIME DESTINATION graphforge/bin
  COMPONENT GraphForgeWheel)
