add_library(TigaRuntime SHARED Runtime.cpp CudaDriver.cpp)
target_link_libraries(TigaRuntime PRIVATE ${CMAKE_DL_LIBS})
target_compile_definitions(TigaRuntime PRIVATE TIGA_RUNTIME_EXPORTS)
target_include_directories(TigaRuntime
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
set_target_properties(TigaRuntime PROPERTIES
  OUTPUT_NAME tiga_runtime
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES)

install(
  TARGETS TigaRuntime
  LIBRARY DESTINATION tiga/lib
    COMPONENT TigaWheel
  RUNTIME DESTINATION tiga/lib
    COMPONENT TigaWheel)
