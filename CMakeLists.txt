cmake_minimum_required(VERSION 3.20)
project(VAiT VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# HIP-touching TU's compile as C++ (see -D below); set a modern C++ standard.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Prevent in-source builds.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "In-source builds are not allowed. Use a separate build directory.")
endif()

find_package(Vulkan REQUIRED)

# ── ROCm root (HIP / rocm-libraries path) ────────────────────────────────────
# Defaults to F:/ROCM-7.14.0-Windows; override with -DROCM_ROOT=/path
set(ROCM_ROOT "F:/ROCM-7.14.0-Windows" CACHE PATH "ROCm installation root")
set(ROCM_LIB_DIR "${ROCM_ROOT}/lib" CACHE PATH "ROCm import library directory")
set(ROCM_INC_DIR "${ROCM_ROOT}/include" CACHE PATH "ROCm include directory")
set(ROCM_BIN_DIR "${ROCM_ROOT}/bin" CACHE PATH "ROCm DLL directory (for test ENVIRONMENT PATH)")

# Warning flags: MSVC accepts neither -Wall semantics nor -Wextra; map to /W4.
set(VAIT_WARN_FLAGS "")
if(MSVC)
    set(VAIT_WARN_FLAGS /W4)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    set(VAIT_WARN_FLAGS -Wall -Wextra)
endif()

# ── HIP / VJITC bridge ────────────────────────────────────────────────────────
# HIP-touching TU's (vkblas_sparse/lapack/jit/conv*) include HIP headers and
# require a C++ compiler + ROCm libraries. When VAIT_HIP=OFF, those TU's are
# replaced by stubs (src/vkblas/vkblas_stub.c) that return
# VK_ERROR_FEATURE_NOT_PRESENT, and no ROCm toolchain is required.
option(VAIT_HIP "Enable HIP/VJITC bridge (sparse, LAPACK, conv, JIT)" OFF)
# HIP-touching TU's compile as C++ when VAIT_HIP is ON.
option(VAIT_FORCE_CLANGCXX "Force all CXX sources through clang++ (HIP workaround)" OFF)
if(VAIT_FORCE_CLANGCXX AND CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    message(FATAL_ERROR "VAIT_FORCE_CLANGCXX requires CMAKE_CXX_COMPILER to be clang++ (ROCm toolchain)")
endif()

# ── Shader header regeneration ──────────────────────────────────────────────
# The embedded SPIR-V headers (src/*/shaders_spv.h) are checked into the repo,
# but can be regenerated with compile_shaders.ps1. A custom target makes this
# available as:  cmake --build . --target shaders
if(WIN32 AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/compile_shaders.ps1")
    add_custom_target(shaders
        COMMAND powershell -NoProfile -ExecutionPolicy Bypass
                -File "${CMAKE_CURRENT_SOURCE_DIR}/shaders/compile_shaders.ps1"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMENT "Regenerating embedded SPIR-V headers (compile_shaders.ps1)")
endif()

# ── VKRuntime library (base layer; every other library sits on it) ──────────
add_library(vkruntime STATIC src/vkruntime/vkruntime.c)
target_include_directories(vkruntime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/vkruntime
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vkruntime
)
target_link_libraries(vkruntime PUBLIC Vulkan::Vulkan)
target_compile_options(vkruntime PRIVATE ${VAIT_WARN_FLAGS})

# ── VKMath library ──────────────────────────────────────────────────────────
add_library(vkmath STATIC
    src/vkmath/vkmath.c
)
target_include_directories(vkmath PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/vkmath
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vkmath
    ${CMAKE_CURRENT_SOURCE_DIR}/include/vkruntime
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vkruntime
)
target_link_libraries(vkmath PUBLIC Vulkan::Vulkan vkruntime)
target_compile_options(vkmath PRIVATE ${VAIT_WARN_FLAGS})

# ── VKBLAS library ──────────────────────────────────────────────────────────
# HIP C++ headers (hip/hip_runtime.h, hipsparse, hiprtc) require a C++
# compiler. The MSVC toolchain's cl.exe (C mode) cannot parse them, so the
# HIP-touching translation units below are forced to C++ and compiled with the
# ROCm clang++ driver (see -DCMAKE_CXX_COMPILER=... in README). vkblas.c itself
# stays pure C / MSVC. The resulting COFF objects link together under the MSVC
# linker because ROCm clang++ on Windows targets the MSVC ABI.
if(VAIT_HIP)
    set(VKBLAS_HIP_SOURCES
        src/vkblas/vkblas_sparse.c
        src/vkblas/vkblas_lapack.c
        src/vkblas/vkblas_conv3d.c
        src/vkblas/vkblas_conv12d.c
        src/vkblas/vkblas_jit.c
    )
    set_source_files_properties(${VKBLAS_HIP_SOURCES}
        PROPERTIES LANGUAGE CXX)
    set(VKBLAS_ROCM_INC ${ROCM_INC_DIR})
    set(VKBLAS_ROCM_LIBS
        ${ROCM_LIB_DIR}/amdhip64.lib
        ${ROCM_LIB_DIR}/hipsparse.lib
        ${ROCM_LIB_DIR}/rocblas.lib
        ${ROCM_LIB_DIR}/rocsolver.lib
        ${ROCM_LIB_DIR}/MIOpen.lib)
else()
    set(VKBLAS_HIP_SOURCES "")
    set(VKBLAS_ROCM_INC "")
    set(VKBLAS_ROCM_LIBS "")
endif()

add_library(vkblas STATIC
    src/vkblas/vkblas.c
    src/vkblas/vkblas_conv3d.c
    src/vkblas/vkblas_conv12d.c
    ${VKBLAS_HIP_SOURCES}
)
target_include_directories(vkblas PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/vkblas
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vkblas
    ${CMAKE_CURRENT_SOURCE_DIR}/include/vkruntime
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vkruntime
    ${VKBLAS_ROCM_INC}
)
# VAIT_JIT: enable hipRTC + shaderc runtime compilation (off by default)
option(VAIT_JIT "Enable JIT compilation via hipRTC and shaderc" OFF)
if(VAIT_JIT)
    if(NOT VAIT_HIP)
        message(FATAL_ERROR "VAIT_JIT requires VAIT_HIP=ON")
    endif()
    target_compile_definitions(vkblas PUBLIC VAIT_JIT)
    list(APPEND VKBLAS_ROCM_LIBS ${ROCM_LIB_DIR}/hiprtc.lib)
endif()
target_link_libraries(vkblas PUBLIC Vulkan::Vulkan vkruntime ${VKBLAS_ROCM_LIBS})
target_compile_options(vkblas PRIVATE ${VAIT_WARN_FLAGS})

# ── Test harnesses ──────────────────────────────────────────────────────────
include(CTest)
if(BUILD_TESTING)
    enable_testing()

    add_executable(test_vkmath tests/test_vkmath.c)
    target_include_directories(test_vkmath PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vkmath
    )
    target_link_libraries(test_vkmath PRIVATE vkmath Vulkan::Vulkan)
    add_test(NAME test_vkmath COMMAND test_vkmath)

    add_executable(test_vkmath_conv2d tests/test_vkmath_conv2d.c)
    target_include_directories(test_vkmath_conv2d PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vkmath
    )
    target_link_libraries(test_vkmath_conv2d PRIVATE vkmath Vulkan::Vulkan)
    add_test(NAME test_vkmath_conv2d COMMAND test_vkmath_conv2d)

    add_executable(test_vkmath_pool2d tests/test_vkmath_pool2d.c)
    target_include_directories(test_vkmath_pool2d PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vkmath
    )
    target_link_libraries(test_vkmath_pool2d PRIVATE vkmath Vulkan::Vulkan)
    add_test(NAME test_vkmath_pool2d COMMAND test_vkmath_pool2d)

    add_executable(test_vkmath_batchnorm tests/test_vkmath_batchnorm.c)
    target_include_directories(test_vkmath_batchnorm PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vkmath
    )
    target_link_libraries(test_vkmath_batchnorm PRIVATE vkmath Vulkan::Vulkan)
    add_test(NAME test_vkmath_batchnorm COMMAND test_vkmath_batchnorm)

    add_executable(test_vkmath_transpose tests/test_vkmath_transpose.c)
    target_include_directories(test_vkmath_transpose PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vkmath
    )
    target_link_libraries(test_vkmath_transpose PRIVATE vkmath Vulkan::Vulkan)
    add_test(NAME test_vkmath_transpose COMMAND test_vkmath_transpose)

    add_executable(test_vkmath_conv1d tests/test_vkmath_conv1d.c)
    target_include_directories(test_vkmath_conv1d PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vkmath
    )
    target_link_libraries(test_vkmath_conv1d PRIVATE vkmath Vulkan::Vulkan)
    add_test(NAME test_vkmath_conv1d COMMAND test_vkmath_conv1d)

    add_executable(test_vkblas tests/test_vkblas.c)
    target_include_directories(test_vkblas PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vkblas
    )
    target_link_libraries(test_vkblas PRIVATE vkblas Vulkan::Vulkan)
    add_test(NAME test_vkblas COMMAND test_vkblas)

# Sparse GEMM VJITC bridge test (requires HIP runtime — C++ compile for HIP headers)
    if(VAIT_HIP)
    add_executable(test_vkblas_sparse tests/test_vkblas_sparse.c)
    set_source_files_properties(tests/test_vkblas_sparse.c PROPERTIES LANGUAGE CXX)
    set_target_properties(test_vkblas_sparse PROPERTIES CXX_STANDARD 20)
    target_include_directories(test_vkblas_sparse PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vkblas
        ${ROCM_INC_DIR}
    )
    target_link_libraries(test_vkblas_sparse PRIVATE
        vkblas vkruntime Vulkan::Vulkan
        ${ROCM_LIB_DIR}/amdhip64.lib
    )
    add_test(NAME test_vkblas_sparse COMMAND test_vkblas_sparse)

    # JIT compilation test (requires hipRTC + shaderc — C++ compile for HIP headers)
    add_executable(test_vkblas_jit tests/test_vkblas_jit.c)
    set_source_files_properties(tests/test_vkblas_jit.c PROPERTIES LANGUAGE CXX)
    set_target_properties(test_vkblas_jit PROPERTIES CXX_STANDARD 20)
    target_include_directories(test_vkblas_jit PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vkblas
        ${ROCM_INC_DIR}
        ${Vulkan_SDK}/Include
    )
    target_link_libraries(test_vkblas_jit PRIVATE
        vkblas vkruntime Vulkan::Vulkan
        ${ROCM_LIB_DIR}/amdhip64.lib
        ${ROCM_LIB_DIR}/hiprtc.lib
    )
    add_test(NAME test_vkblas_jit COMMAND test_vkblas_jit)
    set_tests_properties(test_vkblas_jit PROPERTIES
        ENVIRONMENT "PATH=${ROCM_BIN_DIR}\\;C:/VulkanSDK/1.4.357.0/Bin"
    )

    # LAPACK VJITC bridge test (requires HIP runtime — C++ compile for HIP headers)
    add_executable(test_vkblas_lapack tests/test_vkblas_lapack.c)
    set_source_files_properties(tests/test_vkblas_lapack.c PROPERTIES LANGUAGE CXX)
    set_target_properties(test_vkblas_lapack PROPERTIES CXX_STANDARD 20)
    target_include_directories(test_vkblas_lapack PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vkblas
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vkmath
        ${ROCM_INC_DIR}
    )
    target_link_libraries(test_vkblas_lapack PRIVATE
        vkblas vkruntime Vulkan::Vulkan
        ${ROCM_LIB_DIR}/amdhip64.lib
    )
    add_test(NAME test_vkblas_lapack COMMAND test_vkblas_lapack)
    set_tests_properties(test_vkblas_lapack PROPERTIES
        ENVIRONMENT "PATH=${ROCM_BIN_DIR}\\;C:/VulkanSDK/1.4.357.0/Bin"
    )
    set_tests_properties(test_vkblas_sparse PROPERTIES
        ENVIRONMENT "PATH=${ROCM_BIN_DIR}\\;C:/VulkanSDK/1.4.357.0/Bin"
    )

    # Conv1d/2d VJITC bridge test (requires MIOpen via HIP runtime — C++ compile)
    add_executable(test_vkblas_conv12d tests/test_vkblas_conv12d.c)
    set_source_files_properties(tests/test_vkblas_conv12d.c PROPERTIES LANGUAGE CXX)
    set_target_properties(test_vkblas_conv12d PROPERTIES CXX_STANDARD 20)
    target_include_directories(test_vkblas_conv12d PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vkblas
        ${ROCM_INC_DIR}
    )
    target_link_libraries(test_vkblas_conv12d PRIVATE
        vkblas vkruntime Vulkan::Vulkan
        ${ROCM_LIB_DIR}/amdhip64.lib
    )
    add_test(NAME test_vkblas_conv12d COMMAND test_vkblas_conv12d)
    set_tests_properties(test_vkblas_conv12d PROPERTIES
        ENVIRONMENT "PATH=${ROCM_BIN_DIR}\\;C:/VulkanSDK/1.4.357.0/Bin"
    )
    endif() # VAIT_HIP tests

    # ── Native Vulkan convolution tests (always built, no HIP) ─────────────
    add_executable(test_vkblas_conv3d tests/test_vkblas_conv3d.c)
    target_include_directories(test_vkblas_conv3d PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vkblas
    )
    target_link_libraries(test_vkblas_conv3d PRIVATE vkblas vkruntime Vulkan::Vulkan)
    add_test(NAME test_vkblas_conv3d COMMAND test_vkblas_conv3d)

    add_executable(test_vkblas_conv12d tests/test_vkblas_conv12d.c)
    target_include_directories(test_vkblas_conv12d PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vkblas
    )
    target_link_libraries(test_vkblas_conv12d PRIVATE vkblas vkruntime Vulkan::Vulkan)
    add_test(NAME test_vkblas_conv12d COMMAND test_vkblas_conv12d)

# ── VKQuant library ──────────────────────────────────────────────────
    add_library(vkquant STATIC src/vkquant/vkquant.c)
    target_include_directories(vkquant PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vkquant
        ${CMAKE_CURRENT_SOURCE_DIR}/src/vkquant
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vkruntime
        ${CMAKE_CURRENT_SOURCE_DIR}/src/vkruntime
    )
    target_link_libraries(vkquant PUBLIC Vulkan::Vulkan vkruntime)
    target_compile_options(vkquant PRIVATE ${VAIT_WARN_FLAGS})

    add_executable(test_vkquant tests/test_vkquant.c)
    target_include_directories(test_vkquant PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vkquant
    )
    target_link_libraries(test_vkquant PRIVATE vkquant Vulkan::Vulkan)
    add_test(NAME test_vkquant COMMAND test_vkquant)

    # ── VKRAND library ───────────────────────────────────────────────────
    add_library(vkrand STATIC src/vkrand/vkrand.c)
    target_include_directories(vkrand PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vkrand
        ${CMAKE_CURRENT_SOURCE_DIR}/src/vkrand
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vkruntime
        ${CMAKE_CURRENT_SOURCE_DIR}/src/vkruntime
    )
    target_link_libraries(vkrand PUBLIC Vulkan::Vulkan vkruntime)
    target_compile_options(vkrand PRIVATE ${VAIT_WARN_FLAGS})

    add_executable(test_vkrand tests/test_vkrand.c)
    target_include_directories(test_vkrand PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vkrand
    )
    target_link_libraries(test_vkrand PRIVATE vkrand Vulkan::Vulkan)
    add_test(NAME test_vkrand COMMAND test_vkrand)

    # ── VKFFT library ────────────────────────────────────────────────────
    add_library(vkfft STATIC src/vkfft/vkfft.c)
    target_include_directories(vkfft PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vkfft
        ${CMAKE_CURRENT_SOURCE_DIR}/src/vkfft
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vkruntime
        ${CMAKE_CURRENT_SOURCE_DIR}/src/vkruntime
    )
    target_link_libraries(vkfft PUBLIC Vulkan::Vulkan vkruntime)
    target_compile_options(vkfft PRIVATE ${VAIT_WARN_FLAGS})

    add_executable(test_vkfft tests/test_vkfft.c)
    target_include_directories(test_vkfft PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vkfft
    )
    target_link_libraries(test_vkfft PRIVATE vkfft Vulkan::Vulkan)
    add_test(NAME test_vkfft COMMAND test_vkfft)

    add_executable(test_vkruntime tests/test_vkruntime.c)
    target_include_directories(test_vkruntime PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vkruntime
    )
    target_link_libraries(test_vkruntime PRIVATE vkruntime Vulkan::Vulkan)
    add_test(NAME test_vkruntime COMMAND test_vkruntime)

    # ── VKBLAS L1/L2 library (reuses VkBLASContext, links vkblas) ────────
    add_library(vkblas_l1l2 STATIC src/vkblas_l1l2/vkblas_l1l2.c)
    target_include_directories(vkblas_l1l2 PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vkblas
        ${CMAKE_CURRENT_SOURCE_DIR}/src/vkblas
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vkblas_l1l2
        ${CMAKE_CURRENT_SOURCE_DIR}/src/vkblas_l1l2
    )
    target_link_libraries(vkblas_l1l2 PUBLIC vkblas Vulkan::Vulkan)
    target_compile_options(vkblas_l1l2 PRIVATE ${VAIT_WARN_FLAGS})

    add_executable(test_vkblas_l1l2 tests/test_vkblas_l1l2.c)
    target_include_directories(test_vkblas_l1l2 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vkblas
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vkblas_l1l2
    )
    target_link_libraries(test_vkblas_l1l2 PRIVATE vkblas_l1l2 vkblas Vulkan::Vulkan)
    add_test(NAME test_vkblas_l1l2 COMMAND test_vkblas_l1l2)

    # ── VKModel library (GGUF loader) ──────────────────────────────────────
    add_library(vkmodel STATIC src/vkmodel/vkmodel.c)
    target_include_directories(vkmodel PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vkmodel
        ${CMAKE_CURRENT_SOURCE_DIR}/src/vkmodel
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vkruntime
        ${CMAKE_CURRENT_SOURCE_DIR}/src/vkruntime
    )
    target_link_libraries(vkmodel PUBLIC vkruntime Vulkan::Vulkan)
    target_compile_options(vkmodel PRIVATE ${VAIT_WARN_FLAGS})

    add_executable(test_vkmodel tests/test_vkmodel.c)
    target_include_directories(test_vkmodel PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vkmodel
    )
    target_link_libraries(test_vkmodel PRIVATE vkmodel vkruntime Vulkan::Vulkan)
    add_test(NAME test_vkmodel COMMAND test_vkmodel)

    # ── VKKV library (cross-model KV transfer) ─────────────────────────────
    add_library(vkkv STATIC src/vkkv/vkkv.c)
    target_include_directories(vkkv PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vkkv
        ${CMAKE_CURRENT_SOURCE_DIR}/src/vkkv
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vkruntime
        ${CMAKE_CURRENT_SOURCE_DIR}/src/vkruntime
    )
    target_link_libraries(vkkv PUBLIC vkruntime Vulkan::Vulkan)
    target_compile_options(vkkv PRIVATE ${VAIT_WARN_FLAGS})

    add_executable(test_vkkv tests/test_vkkv.c)
    target_include_directories(test_vkkv PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vkkv
    )
    target_link_libraries(test_vkkv PRIVATE vkkv vkruntime Vulkan::Vulkan)
    add_test(NAME test_vkkv COMMAND test_vkkv)

    # ── VKStream library (async streaming w/ BAR support) ────────────────────
    add_library(vkstream STATIC src/vkstream/vkstream.c)
    target_include_directories(vkstream PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vkstream
        ${CMAKE_CURRENT_SOURCE_DIR}/src/vkstream
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vkruntime
        ${CMAKE_CURRENT_SOURCE_DIR}/src/vkruntime
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vkmodel
    )
    target_link_libraries(vkstream PUBLIC vkruntime Vulkan::Vulkan)
    target_compile_definitions(vkstream PRIVATE VKSTREAM_EXPORTS)
    target_compile_options(vkstream PRIVATE ${VAIT_WARN_FLAGS})

    add_executable(test_vkstream tests/test_vkstream.c)
    target_include_directories(test_vkstream PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vkstream
    )
    target_link_libraries(test_vkstream PRIVATE vkstream vkruntime Vulkan::Vulkan)
    add_test(NAME test_vkstream COMMAND test_vkstream)

    # HIP-Vulkan bridge test (requires HIP runtime)
    if(VAIT_HIP AND EXISTS "${ROCM_INC_DIR}/hip/hip_runtime.h")
        add_executable(test_hip_bar tests/test_hip_bar.c)
        set_source_files_properties(tests/test_hip_bar.c PROPERTIES LANGUAGE CXX)
        target_include_directories(test_hip_bar PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/include/vkstream
            ${ROCM_INC_DIR}
        )
        target_link_libraries(test_hip_bar PRIVATE vkstream vkruntime Vulkan::Vulkan ${ROCM_LIB_DIR}/amdhip64.lib)
        set_target_properties(test_hip_bar PROPERTIES CXX_STANDARD 20)
        add_test(NAME test_hip_bar COMMAND test_hip_bar)

        # VJITC bridge library validation test
        add_executable(test_hip_bridge_libraries tests/test_hip_bridge_libraries.c)
        set_source_files_properties(tests/test_hip_bridge_libraries.c PROPERTIES LANGUAGE CXX)
        target_include_directories(test_hip_bridge_libraries PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${ROCM_INC_DIR}
        )
        target_link_libraries(test_hip_bridge_libraries PRIVATE
            vkruntime Vulkan::Vulkan
            ${ROCM_LIB_DIR}/amdhip64.lib
        )
        set_target_properties(test_hip_bridge_libraries PROPERTIES CXX_STANDARD 20)
        add_test(NAME test_hip_bridge_libraries COMMAND test_hip_bridge_libraries)
    endif() # HIP runtime available
    # ── VKDist library (distributed compute, loopback Phase 0) ─────────────
    # POSIX-only implementation (pthread.h); skipped under MSVC until a
    # Win32-thread port lands.
    if(NOT MSVC)
    add_library(vkdist STATIC src/vkdist/vkdist.c)
    target_include_directories(vkdist PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vkdist
        ${CMAKE_CURRENT_SOURCE_DIR}/src/vkdist
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vkruntime
        ${CMAKE_CURRENT_SOURCE_DIR}/src/vkruntime
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vkblas
        ${CMAKE_CURRENT_SOURCE_DIR}/src/vkblas
    )
    target_link_libraries(vkdist PUBLIC vkblas vkruntime Vulkan::Vulkan)
    target_compile_options(vkdist PRIVATE ${VAIT_WARN_FLAGS})
    if(WIN32)
        target_link_libraries(vkdist PUBLIC ws2_32)
    endif()

    add_executable(test_vkdist tests/test_vkdist.c)
    target_include_directories(test_vkdist PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vkdist
    )
    target_link_libraries(test_vkdist PRIVATE vkdist vkblas vkruntime Vulkan::Vulkan)
    if(WIN32)
        target_link_libraries(test_vkdist PRIVATE ws2_32)
    endif()
    add_test(NAME test_vkdist COMMAND test_vkdist)

    add_executable(test_vkdist_xpc tests/test_vkdist_xpc.c)
    target_include_directories(test_vkdist_xpc PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vkdist
    )
    target_link_libraries(test_vkdist_xpc PRIVATE vkdist vkblas vkruntime Vulkan::Vulkan)
    if(WIN32)
        target_link_libraries(test_vkdist_xpc PRIVATE ws2_32)
    endif()
    endif() # NOT MSVC (vkdist)
endif()
