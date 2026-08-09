cmake_minimum_required(VERSION 3.20)
project(VAiT VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Prevent in-source builds.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "In-source builds are not allowed. Use a separate build directory.")
endif()

find_package(Vulkan REQUIRED)

if(NOT Vulkan_FOUND)
    message(FATAL_ERROR "Vulkan SDK not found. Set VULKAN_SDK or install the Vulkan SDK.")
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
target_compile_options(vkruntime PRIVATE -Wall -Wextra)

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
target_compile_options(vkmath PRIVATE -Wall -Wextra)

# ── VKBLAS library ──────────────────────────────────────────────────────────
add_library(vkblas STATIC
    src/vkblas/vkblas.c
)
target_include_directories(vkblas PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/vkblas
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vkblas
    ${CMAKE_CURRENT_SOURCE_DIR}/include/vkruntime
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vkruntime
)
target_link_libraries(vkblas PUBLIC Vulkan::Vulkan vkruntime)
target_compile_options(vkblas PRIVATE -Wall -Wextra)

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

    add_executable(test_vkblas tests/test_vkblas.c)
    target_include_directories(test_vkblas PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vkblas
    )
    target_link_libraries(test_vkblas PRIVATE vkblas Vulkan::Vulkan)
    add_test(NAME test_vkblas COMMAND test_vkblas)

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
    target_compile_options(vkquant PRIVATE -Wall -Wextra)

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
    target_compile_options(vkrand PRIVATE -Wall -Wextra)

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
    target_compile_options(vkfft PRIVATE -Wall -Wextra)

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
    target_compile_options(vkblas_l1l2 PRIVATE -Wall -Wextra)

    add_executable(test_vkblas_l1l2 tests/test_vkblas_l1l2.c)
    target_include_directories(test_vkblas_l1l2 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vkblas
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vkblas_l1l2
    )
    target_link_libraries(test_vkblas_l1l2 PRIVATE vkblas_l1l2 vkblas Vulkan::Vulkan)
    add_test(NAME test_vkblas_l1l2 COMMAND test_vkblas_l1l2)
endif()
