cmake_minimum_required(VERSION 3.20)
project(VAiSt LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(WIN32)
  set(VAIST_PLATFORM windows)
else()
  set(VAIST_PLATFORM linux)
endif()

set(VAIST_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/vulkan/${VAIST_PLATFORM}")
set(VAIST_C99 "${VAIST_ROOT}/c99")
set(VAIST_CPP "${VAIST_ROOT}/c++")
set(VAIST_PYTHON "${VAIST_ROOT}/python")
set(VAIST_MODULES core runtime tensor compute quant graph model nn llm engine ai distributed blas linalg attn)

option(VAIST_ENABLE_VULKAN "Pull in the Vulkan SDK headers for the Vulkan compute backend" OFF)
if(VAIST_ENABLE_VULKAN)
  if(DEFINED ENV{VULKAN_SDK})
    set(VAIST_VULKAN_INCLUDE "$ENV{VULKAN_SDK}/Include")
  elseif(DEFINED VULKAN_SDK)
    set(VAIST_VULKAN_INCLUDE "${VULKAN_SDK}/Include")
  else()
    # Try a find_path fallback (no link dependency — symbols are resolved
    # dynamically through libvulkan at runtime, matching the weak-GPU stance).
    find_path(VAIST_VULKAN_INCLUDE vulkan/vulkan.h PATH_SUFFIXES Include NO_DEFAULT_PATH)
    if(NOT VAIST_VULKAN_INCLUDE)
      find_path(VAIST_VULKAN_INCLUDE vulkan/vulkan.h)
    endif()
  endif()
  if(NOT EXISTS "${VAIST_VULKAN_INCLUDE}/vulkan/vulkan.h")
    message(WARNING "VAIST_ENABLE_VULKAN is ON but vulkan/vulkan.h was not found; "
                    "Vulkan backends will compile with stubs. Set VULKAN_SDK or install the SDK.")
    set(VAIST_VULKAN_INCLUDE "")
  endif()
   message(STATUS "VAiSt Vulkan SDK include: ${VAIST_VULKAN_INCLUDE}")
   # Detect the SDK's vkAllocateCommandBuffers convention at configure time.
   # Modern Vulkan 1.4: VkCommandBufferAllocateInfo.pCommandBuffers exists and
   # the PFN is 2-arg. Some SDK headers omit the struct field (3-arg PFN).
   # A file scan is deterministic (no test-compile needed) and works headless.
   if(VAIST_VULKAN_INCLUDE)
     set(_vkcore "${VAIST_VULKAN_INCLUDE}/vulkan/vulkan_core.h")
     if(NOT EXISTS "${_vkcore}")
       set(_vkcore "${VAIST_VULKAN_INCLUDE}/vulkan/vulkan.h")
     endif()
     if(EXISTS "${_vkcore}")
       file(STRINGS "${_vkcore}" _vk_alloc_strings REGEX "VkCommandBufferAllocateInfo")
       set(_vaist_vk_3arg FALSE)
       foreach(_line IN LISTS _vk_alloc_strings)
         if(_line MATCHES "pCommandBuffers")
           set(_vaist_vk_3arg FALSE)
         endif()
       endforeach()
       # Distinguish 3-arg PFN: the struct-with-field variant is 2-arg.
       file(STRINGS "${_vkcore}" _vk_alloc_pn REGEX "PFN_vkAllocateCommandBuffers")
       foreach(_line IN LISTS _vk_alloc_pn)
         if(_line MATCHES "VkCommandBuffer\\* *pCommandBuffers")
           set(_vaist_vk_3arg TRUE)
         endif()
       endforeach()
       if(_vaist_vk_3arg)
         set(VAIST_VK_ALLOC_3ARG_COMPILE TRUE)
         message(STATUS "vaist: SDK exposes 3-arg vkAllocateCommandBuffers (defective struct)")
       else()
         set(VAIST_VK_ALLOC_3ARG_COMPILE FALSE)
         message(STATUS "vaist: SDK has VkCommandBufferAllocateInfo.pCommandBuffers (Vulkan 1.4)")
       endif()
     endif()
   endif()
 endif()
add_library(vaist_vulkan_include INTERFACE)
if(VAIST_VULKAN_INCLUDE)
  target_include_directories(vaist_vulkan_include INTERFACE "${VAIST_VULKAN_INCLUDE}")
endif()

function(vaist_c_warning_flags target)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANG_AND_ID:C,GNU>:-Wall;-Wextra;-Werror>
    $<$<COMPILE_LANG_AND_ID:C,Clang>:-Wall;-Wextra;-Werror>
    $<$<COMPILE_LANG_AND_ID:C,MSVC>:/W4;/WX>)
endfunction()

function(vaist_cpp_warning_flags target)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wall;-Wextra;-Werror>
    $<$<COMPILE_LANG_AND_ID:CXX,Clang>:-Wall;-Wextra;-Werror>
    $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/W4;/WX>)
endfunction()

# -----------------------------------------------------------------------------
# C99: exactly one independently buildable shared library per component.
# -----------------------------------------------------------------------------
add_library(vaist_core SHARED "${VAIST_C99}/core/src/vaist_core.c")
# Canonical header root: include/vaist/. All component headers live here,
# so build never reads the per-platform vulkan/<plat>/c99/<mod>/include copies.
target_include_directories(vaist_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/vaist>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/vaist>)
if(WIN32)
    set_target_properties(vaist_core PROPERTIES PREFIX "")
    target_compile_definitions(vaist_core PRIVATE VAIST_CORE_BUILD _CRT_SECURE_NO_WARNINGS)
endif()
vaist_c_warning_flags(vaist_core)

foreach(D IN LISTS VAIST_MODULES)
  if(D STREQUAL "core")
    continue()
  endif()
  set(M "vaist_${D}")
  add_library(${M} SHARED "${VAIST_C99}/${D}/src/${M}.c")
  target_include_directories(${M} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/vaist>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/vaist>)
  target_link_libraries(${M} PRIVATE vaist_core)
  if(D STREQUAL "blas")
    target_link_libraries(${M} PRIVATE vaist_runtime vaist_compute vaist_quant)
  endif()
  if(D STREQUAL "compute")
    target_link_libraries(${M} PRIVATE vaist_runtime)
  endif()
  if(D STREQUAL "linalg")
    target_link_libraries(${M} PRIVATE vaist_blas vaist_runtime)
  endif()
  if(D STREQUAL "attn")
    target_link_libraries(${M} PRIVATE vaist_runtime vaist_compute vaist_quant)
    if(VAIST_ENABLE_VULKAN)
      target_link_libraries(${M} PRIVATE vaist_vulkan_include)
      target_compile_definitions(${M} PRIVATE VAIST_VULKAN_HDR=1 VAIST_ENABLE_VULKAN=1 VAIST_HAVE_VK_HDR=1 VAIST_SPIRV_EMBEDDED=1)
      target_include_directories(${M} PRIVATE "${VAIST_SHADER_OUT_DIR}")
    endif()
  endif()
  vaist_c_warning_flags(${M})
  if(WIN32)
    set_target_properties(${M} PROPERTIES PREFIX "")
    target_compile_definitions(${M} PRIVATE _CRT_SECURE_NO_WARNINGS)
  endif()
  if(UNIX AND NOT APPLE AND D STREQUAL "runtime")
    if(VAIST_ENABLE_VULKAN)
      target_link_libraries(${M} PRIVATE dl vaist_vulkan_include)
      target_compile_definitions(${M} PRIVATE VAIST_VULKAN_HDR=1 VAIST_ENABLE_VULKAN=1)
    else()
      target_link_libraries(${M} PRIVATE dl)
    endif()
  elseif(WIN32 AND D STREQUAL "runtime" AND VAIST_ENABLE_VULKAN)
    target_link_libraries(${M} PRIVATE vaist_vulkan_include)
    target_compile_definitions(${M} PRIVATE VAIST_VULKAN_HDR=1 VAIST_ENABLE_VULKAN=1)
  endif()
  if(UNIX AND NOT APPLE)
    set_target_properties(${M} PROPERTIES INSTALL_RPATH "$ORIGIN")
  endif()
endforeach()
if(UNIX AND NOT APPLE)
  set_target_properties(vaist_core PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()

# -----------------------------------------------------------------------------
# C++: twelve independently buildable component libraries.
# Each wrapper links only its corresponding C99 component target.
# -----------------------------------------------------------------------------
foreach(D IN LISTS VAIST_MODULES)
  set(M "vaist_${D}")
  set(CPP_TARGET "vaist_cpp_${D}")
  add_library(${CPP_TARGET} STATIC "${VAIST_CPP}/${D}/vaist_${D}.cpp")
  target_include_directories(${CPP_TARGET} PUBLIC
    $<BUILD_INTERFACE:${VAIST_CPP}/${D}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/vaist>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/vaist>)
  target_link_libraries(${CPP_TARGET} PUBLIC ${M})
  if(D STREQUAL "linalg")
    target_link_libraries(${CPP_TARGET} PUBLIC vaist_blas)
  endif()
  vaist_cpp_warning_flags(${CPP_TARGET})
endforeach()

foreach(D IN LISTS VAIST_MODULES)
  set(TEST_TARGET "vaist_cpp_${D}_test")
  add_executable(${TEST_TARGET} "${VAIST_CPP}/${D}/test_${D}.cpp")
  target_include_directories(${TEST_TARGET} PRIVATE
    "${VAIST_CPP}/${D}"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/vaist")
  target_link_libraries(${TEST_TARGET} PRIVATE vaist_cpp_${D} vaist_runtime)
  vaist_cpp_warning_flags(${TEST_TARGET})
  add_test(NAME ${TEST_TARGET} COMMAND ${TEST_TARGET})
endforeach()

# Optional aggregate convenience target. It does not replace individual targets.
add_library(vaist_cpp_wrappers INTERFACE)
foreach(D IN LISTS VAIST_MODULES)
  target_link_libraries(vaist_cpp_wrappers INTERFACE vaist_cpp_${D})
endforeach()

# -----------------------------------------------------------------------------
# SPIR-V shaders for vaist_blas. Compiled with glslangValidator when present in
# the Vulkan SDK; validated with spirv-val. Output lands in build/shaders/.
# Mirrors scripts/run_vulkan_checks.sh (ctest + spirv-val must pass).
# -----------------------------------------------------------------------------
set(VAIST_SHADERS
  vkattn/baseline/attn_flash_decode.comp
  vkblas/baseline/qgemm_q8_0.comp
  vkblas/baseline/qgemm_q4_0.comp
  vkblas/baseline/qgemm_q4_1.comp
  vkblas/baseline/qgemm_q5_0.comp
  vkblas/baseline/qgemm_q5_1.comp
  vkblas/baseline/qgemm_q2_K.comp
  vkblas/baseline/qgemm_q3k.comp
  vkblas/baseline/qgemm_q4k.comp
  vkblas/baseline/qgemm_q5k.comp
  vkblas/baseline/qgemm_q6k.comp
  vkblas/baseline/qgemm_iq4xs.comp
  vkblas/baseline/moe_route.comp)
# Pre-existing SPIR-V blobs with no source .comp (half-migrated repo).
# Their base names map to the symbol names vaist_blas.c references.
set(VAIST_PRECOMPILED_SPV
  vkblas/gemm_f32_tiled.comp.spv
  vkblas/matvec_ternary.comp.spv
  vkblas/matvec_binar.comp.spv)
set(VAIST_SHADER_OUT_DIR "${CMAKE_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY "${VAIST_SHADER_OUT_DIR}")
find_program(GLSLANG_VALIDATOR glslangValidator PATHS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}")
find_program(SPIRV_VAL spirv-val PATHS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}")
if(GLSLANG_VALIDATOR)
  set(VAIST_SPV "")
  foreach(S IN LISTS VAIST_SHADERS)
    set(out "${VAIST_SHADER_OUT_DIR}/${S}.spv")
    get_filename_component(_sd "${out}" DIRECTORY)
    file(MAKE_DIRECTORY "${_sd}")
    add_custom_command(
      OUTPUT "${out}"
      COMMAND ${GLSLANG_VALIDATOR} -V --target-env vulkan1.3
              "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${S}"
              -o "${out}"
      COMMAND ${SPIRV_VAL} "${out}"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${S}"
      COMMENT "glslang: ${S}" VERBATIM)
    list(APPEND VAIST_SPV "${out}")
  endforeach()
  add_custom_target(vaist_shaders ALL DEPENDS ${VAIST_SPV})
  if(SPIRV_VAL)
    add_custom_target(vaist_spirv_validate
      COMMAND ${CMAKE_COMMAND} -E echo "spirv-val: validating..."
      COMMENT "spirv-val all shaders")
    add_dependencies(vaist_spirv_validate vaist_shaders)
  endif()
endif()

# -----------------------------------------------------------------------------
# SPIR-V embeddings for vaist_blas (shaders/vkblas/*.comp). glslang already
# compiled them to ${VAIST_SHADER_OUT_DIR}/vkblas/*.spv above. When Vulkan is
# enabled, fold those binaries into a generated header (consumed only by
# vaist_blas.c under VAIST_HAVE_VK_HDR) so the library stays self-contained.
# -----------------------------------------------------------------------------
find_package(Python3 COMPONENTS Interpreter QUIET)
if(VAIST_ENABLE_VULKAN AND GLSLANG_VALIDATOR AND Python3_Interpreter_FOUND)
  set(VAIST_BLAS_SPV_HDR "${VAIST_SHADER_OUT_DIR}/vaist_blas_spv.h")
  add_custom_command(
    OUTPUT "${VAIST_BLAS_SPV_HDR}"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/spirv_to_header.py" "${VAIST_BLAS_SPV_HDR}"
             "vaist_spv_gemm_f32_tiled" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/vkblas/gemm_f32_tiled.comp.spv"
             "vaist_spv_matvec_ternary" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/vkblas/matvec_ternary.comp.spv"
             "vaist_spv_matvec_binar"   "${CMAKE_CURRENT_SOURCE_DIR}/shaders/vkblas/matvec_binar.comp.spv"
             "vaist_spv_qgemm_q8_0"     "${VAIST_SHADER_OUT_DIR}/vkblas/baseline/qgemm_q8_0.comp.spv"
             "vaist_spv_qgemm_q4_0"     "${VAIST_SHADER_OUT_DIR}/vkblas/baseline/qgemm_q4_0.comp.spv"
             "vaist_spv_qgemm_q4_1"     "${VAIST_SHADER_OUT_DIR}/vkblas/baseline/qgemm_q4_1.comp.spv"
             "vaist_spv_qgemm_q5_0"     "${VAIST_SHADER_OUT_DIR}/vkblas/baseline/qgemm_q5_0.comp.spv"
             "vaist_spv_qgemm_q5_1"     "${VAIST_SHADER_OUT_DIR}/vkblas/baseline/qgemm_q5_1.comp.spv"
             "vaist_spv_qgemm_q2_K"     "${VAIST_SHADER_OUT_DIR}/vkblas/baseline/qgemm_q2_K.comp.spv"
             "vaist_spv_qgemm_q3k"      "${VAIST_SHADER_OUT_DIR}/vkblas/baseline/qgemm_q3k.comp.spv"
             "vaist_spv_qgemm_q4k"      "${VAIST_SHADER_OUT_DIR}/vkblas/baseline/qgemm_q4k.comp.spv"
             "vaist_spv_qgemm_q5k"      "${VAIST_SHADER_OUT_DIR}/vkblas/baseline/qgemm_q5k.comp.spv"
             "vaist_spv_qgemm_q6k"      "${VAIST_SHADER_OUT_DIR}/vkblas/baseline/qgemm_q6k.comp.spv"
             "vaist_spv_qgemm_iq4xs"    "${VAIST_SHADER_OUT_DIR}/vkblas/baseline/qgemm_iq4xs.comp.spv"
             "vaist_spv_moe_route"      "${VAIST_SHADER_OUT_DIR}/vkblas/baseline/moe_route.comp.spv"
             "vaist_spv_attn_flash_decode" "${VAIST_SHADER_OUT_DIR}/vkattn/baseline/attn_flash_decode.comp.spv"
    DEPENDS ${VAIST_SPV}
    COMMENT "embed SPIR-V -> vaist_blas_spv.h" VERBATIM)
  add_custom_target(vaist_blas_spv_hdr DEPENDS "${VAIST_BLAS_SPV_HDR}")
endif()

# Apply the Vulkan compute configuration to the already-created vaist_blas target.
if(VAIST_ENABLE_VULKAN)
  if(TARGET vaist_blas)
    if(TARGET vaist_vulkan_include)
      target_link_libraries(vaist_blas PRIVATE vaist_vulkan_include)
    endif()
    target_compile_definitions(vaist_blas PRIVATE
        VAIST_VULKAN_HDR=1 VAIST_ENABLE_VULKAN=1 VAIST_HAVE_VK_HDR=1)
    if(VAIST_VK_ALLOC_3ARG_COMPILE)
      target_compile_definitions(vaist_blas PRIVATE VAIST_VK_ALLOC_3ARG=1)
    endif()
    target_include_directories(vaist_blas PRIVATE "${VAIST_SHADER_OUT_DIR}")
    if(TARGET vaist_blas_spv_hdr)
      add_dependencies(vaist_blas vaist_blas_spv_hdr)
    endif()
  endif()
endif()

# Install the SPIR-V artifacts (shaders/*.spv) for Vulkan backends.
if(VAIST_ENABLE_VULKAN AND GLSLANG_VALIDATOR)
  install(FILES ${VAIST_SPV} DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/vaist/shaders)
endif()

# Apply the Vulkan compute configuration to vaist_linalg (GPU tiles are stubs
# that drop to the CPU path; the include + defines keep it ABI-ready for the
# later tile-dispatch pass without forcing a Vulkan SDK on weak/CPU boxes).
if(VAIST_ENABLE_VULKAN)
  if(TARGET vaist_linalg)
    if(TARGET vaist_vulkan_include)
      target_link_libraries(vaist_linalg PRIVATE vaist_vulkan_include)
    endif()
    target_compile_definitions(vaist_linalg PRIVATE
        VAIST_VULKAN_HDR=1 VAIST_ENABLE_VULKAN=1 VAIST_HAVE_VK_HDR=1)
  endif()
endif()

# -----------------------------------------------------------------------------
# Tests: one native C99 stack test, one C++ test covering all twelve targets,

# plus a Python load test for all twelve components.
# -----------------------------------------------------------------------------
enable_testing()

add_executable(vaist_stack_smoke tests/stack_smoke.c)
target_include_directories(vaist_stack_smoke PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include/vaist")
foreach(D IN LISTS VAIST_MODULES)
  target_link_libraries(vaist_stack_smoke PRIVATE vaist_${D})
endforeach()
if(UNIX)
  target_link_libraries(vaist_stack_smoke PRIVATE m dl)
endif()
add_test(NAME stack_smoke COMMAND vaist_stack_smoke)

add_executable(vaist_linalg_test tests/linalg_test.c)
target_include_directories(vaist_linalg_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include/vaist")
target_link_libraries(vaist_linalg_test PRIVATE vaist_linalg vaist_blas vaist_runtime)
if(UNIX)
  target_link_libraries(vaist_linalg_test PRIVATE m)
endif()
add_test(NAME linalg_test COMMAND vaist_linalg_test)

add_executable(vaist_blas_test tests/blas_test.c)
target_include_directories(vaist_blas_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include/vaist")
target_link_libraries(vaist_blas_test PRIVATE vaist_blas vaist_runtime)
if(UNIX)
  target_link_libraries(vaist_blas_test PRIVATE m)
endif()
add_test(NAME blas_test COMMAND vaist_blas_test)

# vaist_blas_vulkan_test: exercises the vaist_blas Vulkan dispatch path when a
# real Vulkan device is present, and the CPU fallback otherwise (weak-box
# friendly: always passes without a device).
add_executable(vaist_blas_vulkan_test tests/vaist_blas_vulkan_test.c)
target_include_directories(vaist_blas_vulkan_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include/vaist")
target_link_libraries(vaist_blas_vulkan_test PRIVATE vaist_blas vaist_runtime)
if(UNIX)
  target_link_libraries(vaist_blas_vulkan_test PRIVATE m)
endif()
add_test(NAME vaist_blas_vulkan_test COMMAND vaist_blas_vulkan_test)

add_executable(vaist_cpp_smoke tests/cpp_smoke_all.cpp)
foreach(D IN LISTS VAIST_MODULES)
  target_link_libraries(vaist_cpp_smoke PRIVATE vaist_cpp_${D})
endforeach()
if(UNIX)
  target_link_libraries(vaist_cpp_smoke PRIVATE m dl)
endif()
add_test(NAME cpp_smoke COMMAND vaist_cpp_smoke)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  add_test(NAME python_component_load
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/python_components.py" "${CMAKE_BINARY_DIR}" "${VAIST_PLATFORM}")
  set_tests_properties(python_component_load PROPERTIES ENVIRONMENT "PYTHONPATH=${VAIST_PYTHON}")
endif()

if(Python3_Interpreter_FOUND)
  add_test(NAME clean_staging_load
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/clean_staging.py" "${CMAKE_BINARY_DIR}" "${VAIST_PLATFORM}" "${CMAKE_COMMAND}")
endif()

if(Python3_Interpreter_FOUND AND UNIX AND NOT APPLE)
  add_test(NAME linux_c99_symbol_check
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/symbol_check.py" "${CMAKE_BINARY_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

# -----------------------------------------------------------------------------
# Install/export contract.
# -----------------------------------------------------------------------------
include(GNUInstallDirs)
install(TARGETS vaist_core vaist_runtime vaist_tensor vaist_compute vaist_quant vaist_graph vaist_model vaist_nn vaist_llm vaist_engine vaist_ai vaist_distributed vaist_blas vaist_linalg vaist_attn
  EXPORT VAiStTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

foreach(D IN LISTS VAIST_MODULES)
  install(TARGETS vaist_cpp_${D}
    EXPORT VAiStTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endforeach()

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
# Per-module C++ wrappers live under vulkan/<plat>/c++/<mod> (platform-specific).
foreach(D IN LISTS VAIST_MODULES)
  install(DIRECTORY "${VAIST_CPP}/${D}/" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/vaist/cpp/${D}
    FILES_MATCHING PATTERN "*.hpp")
endforeach()

foreach(D IN LISTS VAIST_MODULES)
  install(DIRECTORY "${VAIST_PYTHON}/${D}/" DESTINATION "${CMAKE_INSTALL_DATADIR}/vaist/python/${D}")
endforeach()

install(EXPORT VAiStTargets
  FILE VAiStTargets.cmake
  NAMESPACE VAiSt::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/VAiSt")
