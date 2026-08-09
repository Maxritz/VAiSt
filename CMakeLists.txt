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

# ── VKMath library ──────────────────────────────────────────────────────────
add_library(vkmath STATIC
    src/vkmath/vkmath.c
)
target_include_directories(vkmath PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/vkmath
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vkmath
)
target_link_libraries(vkmath PUBLIC Vulkan::Vulkan)
target_compile_options(vkmath PRIVATE -Wall -Wextra)

# ── VKBLAS library ──────────────────────────────────────────────────────────
add_library(vkblas STATIC
    src/vkblas/vkblas.c
)
target_include_directories(vkblas PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/vkblas
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vkblas
)
target_link_libraries(vkblas PUBLIC Vulkan::Vulkan)
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
endif()
