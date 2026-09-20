cmake_minimum_required(VERSION 3.21)
if(POLICY CMP0219)
  cmake_policy(SET CMP0219 NEW)
endif()
project(Tiga LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

include(cmake/LLVMRevision.cmake)
option(
  TIGA_STRICT_LLVM_VERSION
  "Reject an MLIR SDK that differs from the pinned Tiga release"
  ON)
option(TIGA_INCLUDE_TESTS "Build Tiga lit test targets" ON)
option(
  TIGA_BUNDLE_LLVM_RUNTIME
  "Install the MLIR and LLVM shared libraries needed by a standalone wheel"
  ON)

if(APPLE)
  set(TIGA_EXTENSION_RPATH "@loader_path/lib")
  set(TIGA_TOOL_RPATH "@loader_path/../lib")
elseif(UNIX)
  set(TIGA_EXTENSION_RPATH "$ORIGIN/lib")
  set(TIGA_TOOL_RPATH "$ORIGIN/../lib")
else()
  set(TIGA_EXTENSION_RPATH "")
  set(TIGA_TOOL_RPATH "")
endif()

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake from ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake from ${LLVM_DIR}")

# The official Linux LLVM binary SDK exports LLVMSupport with an unconditional
# dependency on zstd::libzstd_static.  On distributions where that system
# archive was not compiled as PIC, linking Tiga's Python extension then
# fails even though Findzstd also found a usable shared library.  Imported
# targets are only build metadata, so point LLVM's static alias at the shared
# artifact for shared Tiga consumers.  LLVM itself does not require zstd
# to be linked statically, and wheel repair can bundle this dynamic dependency.
if(UNIX AND NOT APPLE AND
   TARGET zstd::libzstd_static AND TARGET zstd::libzstd_shared)
  get_target_property(
    _tiga_shared_zstd zstd::libzstd_shared IMPORTED_LOCATION)
  if(_tiga_shared_zstd)
    set_property(
      TARGET zstd::libzstd_static PROPERTY IMPORTED_LOCATION
      "${_tiga_shared_zstd}")
  endif()
  unset(_tiga_shared_zstd)
endif()

if(TIGA_STRICT_LLVM_VERSION AND
   NOT LLVM_PACKAGE_VERSION VERSION_EQUAL TIGA_LLVM_VERSION)
  message(FATAL_ERROR
    "Tiga is pinned to LLVM ${TIGA_LLVM_VERSION}, but found "
    "${LLVM_PACKAGE_VERSION}. Pass -DTIGA_STRICT_LLVM_VERSION=OFF only "
    "for an intentional compatibility experiment.")
endif()

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Some distro SDKs ship lit.py but omit the llvm-lit wrapper.  Keep this as a
# packaging fallback; official SDKs continue to use LLVM_EXTERNAL_LIT.
if(NOT EXISTS "${LLVM_EXTERNAL_LIT}" AND
   EXISTS "${LLVM_BINARY_DIR}/build/utils/lit/lit.py")
  set(LLVM_EXTERNAL_LIT
      "${LLVM_BINARY_DIR}/build/utils/lit/lit.py"
      CACHE FILEPATH "Path to llvm-lit" FORCE)
endif()

set(LLVM_RUNTIME_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/bin")
set(LLVM_LIBRARY_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/lib")
set(MLIR_BINARY_DIR "${CMAKE_BINARY_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories("${PROJECT_BINARY_DIR}/include")
link_directories(${LLVM_BUILD_LIBRARY_DIR})
# Official binary SDKs publish LLVM_DEFINITIONS as one shell-style string,
# while distro packages may expose a CMake list. Normalize it before passing
# definitions to targets; otherwise CMake turns the entire tail into the value
# of _GLIBCXX_USE_CXX11_ABI.
separate_arguments(
  TIGA_LLVM_DEFINITIONS NATIVE_COMMAND "${LLVM_DEFINITIONS}")
add_definitions(${TIGA_LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(python_bindings)
install(FILES THIRD_PARTY_NOTICES.md DESTINATION tiga/licenses COMPONENT TigaWheel)
install(DIRECTORY third_party/licenses/ DESTINATION tiga/licenses COMPONENT TigaWheel)
if(TIGA_INCLUDE_TESTS)
  add_subdirectory(tests)
endif()

# Distro and official LLVM SDKs may expose the monolithic MLIR/LLVM libraries
# as shared imported targets.  A Python wheel cannot rely on the SDK remaining
# installed at the path used to build it, so make those runtime artifacts part
# of the wheel component.  Static SDK builds have no runtime artifact here and
# need no extra installation step.
if(TIGA_BUNDLE_LLVM_RUNTIME)
  set(_tiga_llvm_runtime_targets)
  foreach(_target IN ITEMS MLIR LLVM)
    if(TARGET ${_target})
      get_target_property(_target_type ${_target} TYPE)
      if(_target_type STREQUAL "SHARED_LIBRARY")
        list(APPEND _tiga_llvm_runtime_targets ${_target})
      endif()
    endif()
  endforeach()
  if(_tiga_llvm_runtime_targets)
    if(UNIX)
      # install(IMPORTED_RUNTIME_ARTIFACTS) preserves an imported library's
      # symlink chain.  Some distro LLVM packages point that chain outside the
      # SDK prefix (for example into a multi-arch directory), and wheel formats
      # cannot carry such an external target.  Resolve it at configure time and
      # install a regular file using the ELF/Mach-O install name instead.
      foreach(_target IN LISTS _tiga_llvm_runtime_targets)
        get_target_property(_configs ${_target} IMPORTED_CONFIGURATIONS)
        get_target_property(_location ${_target} IMPORTED_LOCATION)
        get_target_property(_soname ${_target} IMPORTED_SONAME)
        if(NOT _location)
          foreach(_config IN LISTS _configs)
            string(TOUPPER "${_config}" _config_upper)
            get_target_property(
              _candidate ${_target} "IMPORTED_LOCATION_${_config_upper}")
            if(_candidate)
              set(_location "${_candidate}")
              get_target_property(
                _soname ${_target} "IMPORTED_SONAME_${_config_upper}")
              break()
            endif()
          endforeach()
        endif()
        if(NOT _location)
          message(FATAL_ERROR
            "Cannot locate imported shared library target ${_target}")
        endif()
        file(REAL_PATH "${_location}" _real_location)
        if(NOT _soname)
          get_filename_component(_soname "${_real_location}" NAME)
        endif()
        install(
          FILES "${_real_location}"
          DESTINATION tiga/lib
          RENAME "${_soname}"
          COMPONENT TigaWheel)
      endforeach()
    else()
      install(
        IMPORTED_RUNTIME_ARTIFACTS ${_tiga_llvm_runtime_targets}
        LIBRARY DESTINATION tiga/lib
          COMPONENT TigaWheel
        RUNTIME DESTINATION tiga
          COMPONENT TigaWheel)
    endif()
  endif()
endif()
