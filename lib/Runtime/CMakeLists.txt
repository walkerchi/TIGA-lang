add_library(GraphForgeRuntime SHARED Runtime.cpp CudaDriver.cpp)
target_link_libraries(GraphForgeRuntime PRIVATE ${CMAKE_DL_LIBS})
target_compile_definitions(GraphForgeRuntime PRIVATE GRAPHFORGE_RUNTIME_EXPORTS)
target_include_directories(GraphForgeRuntime
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
set_target_properties(GraphForgeRuntime PROPERTIES
  OUTPUT_NAME graphforge_runtime
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES)

install(
  TARGETS GraphForgeRuntime
  LIBRARY DESTINATION graphforge/lib
    COMPONENT GraphForgeWheel
  RUNTIME DESTINATION graphforge/lib
    COMPONENT GraphForgeWheel)
