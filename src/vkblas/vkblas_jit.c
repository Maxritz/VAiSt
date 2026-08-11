/**
 * \file vkblas_jit.c
 * \brief Runtime JIT compilation of Vulkan GLSL shaders and HIP kernels
 *        behind the VAIT_JIT feature flag.
 *
 * Two JIT paths are supported when -DVAIT_JIT=ON:
 *
 *   - shaderc (dynamic load): GLSL source compiled to SPIR-V at runtime,
 *     then loaded as VkShaderModule → VkPipeline.
 *   - hipRTC: HIP C source compiled to machine code via hipRTC, loaded
 *     as a HIP module, and dispatched with shared device memory.
 *
 * Both use the zero-copy bridge (VK_EXT_external_memory_host) so HIP kernels
 * share the same VkBuffer allocation as Vulkan dispatches.
 *
 * All JIT functions return VK_ERROR_FEATURE_NOT_PRESENT when VAIT_JIT
 * is not defined.
 */
#define __HIP_PLATFORM_AMD__ 1

#include "vkblas/vkblas.h"
#include "vkblas_internal.h"
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <windows.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef VAIT_JIT

/* ── shaderc types — minimal declarations for dynamic loading ────────── */
#define SHADER_KIND_COMPUTE 2  /* shaderc_compute_shader = 2 (after vertex=0, fragment=1) */
#define SHADER_STATUS_SUCCESS 0

typedef void* shaderc_compiler_t;
typedef void* shaderc_compile_options_t;
typedef void* shaderc_compile_result_t;

typedef shaderc_compiler_t (*PFN_shaderc_compiler_initialize)(void);
typedef void (*PFN_shaderc_compiler_release)(shaderc_compiler_t);
typedef shaderc_compile_result_t (*PFN_shaderc_compile_into_spv)(
    shaderc_compiler_t, const char*, size_t, int,
    const char*, const char*, const shaderc_compile_options_t);
typedef void (*PFN_shaderc_result_release)(shaderc_compile_result_t);
typedef int (*PFN_shaderc_result_get_compilation_status)(shaderc_compile_result_t);
typedef size_t (*PFN_shaderc_result_get_length)(shaderc_compile_result_t);
typedef const char* (*PFN_shaderc_result_get_bytes)(shaderc_compile_result_t);
typedef const char* (*PFN_shaderc_result_get_error_message)(shaderc_compile_result_t);

VkResult vkblas_jit_compile_glsl_to_spirv(
    const char* source, size_t source_len,
    uint8_t** out_spirv, size_t* out_len)
{
    if (!source || !out_spirv || !out_len) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    if (source_len == 0) source_len = strlen(source);

    HMODULE shaderc_lib = LoadLibraryA("shaderc_shared.dll");
    if (!shaderc_lib) {
        shaderc_lib = LoadLibraryA("shaderc.dll");
    }
    if (!shaderc_lib) {
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    PFN_shaderc_compiler_initialize p_init =
        (PFN_shaderc_compiler_initialize)GetProcAddress(shaderc_lib, "shaderc_compiler_initialize");
    PFN_shaderc_compiler_release p_release =
        (PFN_shaderc_compiler_release)GetProcAddress(shaderc_lib, "shaderc_compiler_release");
    PFN_shaderc_compile_into_spv p_compile =
        (PFN_shaderc_compile_into_spv)GetProcAddress(shaderc_lib, "shaderc_compile_into_spv");
    PFN_shaderc_result_release p_result_release =
        (PFN_shaderc_result_release)GetProcAddress(shaderc_lib, "shaderc_result_release");
    PFN_shaderc_result_get_compilation_status p_get_status =
        (PFN_shaderc_result_get_compilation_status)GetProcAddress(shaderc_lib, "shaderc_result_get_compilation_status");
    PFN_shaderc_result_get_length p_get_length =
        (PFN_shaderc_result_get_length)GetProcAddress(shaderc_lib, "shaderc_result_get_length");
    PFN_shaderc_result_get_bytes p_get_bytes =
        (PFN_shaderc_result_get_bytes)GetProcAddress(shaderc_lib, "shaderc_result_get_bytes");
    PFN_shaderc_result_get_error_message p_get_error_msg =
        (PFN_shaderc_result_get_error_message)GetProcAddress(shaderc_lib, "shaderc_result_get_error_message");

    if (!p_init || !p_release || !p_compile || !p_result_release ||
        !p_get_status || !p_get_length || !p_get_bytes || !p_get_error_msg) {
        FreeLibrary(shaderc_lib);
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    shaderc_compiler_t compiler = p_init();
    if (!compiler) {
        FreeLibrary(shaderc_lib);
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    shaderc_compile_result_t result = p_compile(compiler, source, source_len,
        SHADER_KIND_COMPUTE, "jit_compiled.comp", "main", NULL);

    p_release(compiler);

    if (!result) {
        FreeLibrary(shaderc_lib);
        return VK_ERROR_INVALID_SHADER_NV;
    }

    int status = p_get_status(result);
    if (status != SHADER_STATUS_SUCCESS) {
        const char* err = p_get_error_msg(result);
        if (err) fprintf(stderr, "shaderc: error: %s\n", err);
        fprintf(stderr, "shaderc: compile status=%d\n", status);
        p_result_release(result);
        FreeLibrary(shaderc_lib);
        return VK_ERROR_INVALID_SHADER_NV;
    }

    size_t spirv_len = p_get_length(result);
    const char* spirv_bytes = p_get_bytes(result);
    if (!spirv_bytes || spirv_len == 0) {
        p_result_release(result);
        FreeLibrary(shaderc_lib);
        return VK_ERROR_UNKNOWN;
    }

    void* spirv = malloc(spirv_len);
    if (!spirv) {
        p_result_release(result);
        FreeLibrary(shaderc_lib);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }

    memcpy(spirv, spirv_bytes, spirv_len);
    p_result_release(result);
    FreeLibrary(shaderc_lib);

    *out_spirv = (uint8_t*)spirv;
    *out_len = spirv_len;
    return VK_SUCCESS;
}

VkResult vkblas_jit_create_shader_module(VkDevice device,
    const uint8_t* spirv, size_t spirv_len, VkShaderModule* module)
{
    if (!device || !spirv || !module || (spirv_len % 4 != 0)) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spirv_len,
        .pCode = (const uint32_t*)spirv,
    };

    return vkCreateShaderModule(device, &smci, NULL, module);
}

VkResult vkblas_jit_compile_hip(const char* source, size_t source_len,
    void** out_code, size_t* out_len)
{
    if (!source || !out_code || !out_len) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    if (source_len == 0) source_len = strlen(source);

    hiprtcProgram prog;
    hiprtcResult rt = hiprtcCreateProgram(&prog, source, "jit_compiled.cu",
        0, NULL, NULL);
    if (rt != HIPRTC_SUCCESS) {
        return VK_ERROR_UNKNOWN;
    }

    const char* opts[] = {"--offload-arch=gfx1201"};
    rt = hiprtcCompileProgram(prog, 1, opts);

    if (rt != HIPRTC_SUCCESS) {
        size_t log_size = 0;
        hiprtcGetProgramLogSize(prog, &log_size);
        if (log_size > 0) {
            char* log = (char*)malloc(log_size);
            if (log) {
                hiprtcGetProgramLog(prog, log);
                fprintf(stderr, "hipRTC: error: %s\n", log);
                free(log);
            }
        }
        hiprtcDestroyProgram(&prog);
        return VK_ERROR_INVALID_SHADER_NV;
    }

    size_t code_size = 0;
    hiprtcGetCodeSize(prog, &code_size);
    void* code = malloc(code_size);
    if (!code) {
        hiprtcDestroyProgram(&prog);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }

    hiprtcGetCode(prog, (char*)code);
    hiprtcDestroyProgram(&prog);

    *out_code = code;
    *out_len = code_size;
    return VK_SUCCESS;
}

VkResult vkblas_jit_load_hip_module(const void* code, size_t code_len,
    const char* func_name,
    void* module, void* kernel)
{
    if (!code || !func_name || !module || !kernel) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    (void)code_len;

    hipModule_t* mod = (hipModule_t*)module;
    hipFunction_t* fn = (hipFunction_t*)kernel;

    hipError_t err = hipModuleLoadData(mod, code);
    if (err != hipSuccess) {
        fprintf(stderr, "hipRTC: module load failed: %s\n", hipGetErrorString(err));
        return VK_ERROR_UNKNOWN;
    }

    err = hipModuleGetFunction(fn, *mod, func_name);
    if (err != hipSuccess) {
        hipModuleUnload(*mod);
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    return VK_SUCCESS;
}

#else

VkResult vkblas_jit_compile_glsl_to_spirv(
    const char* source, size_t source_len,
    uint8_t** out_spirv, size_t* out_len)
{
    (void)source; (void)source_len; (void)out_spirv; (void)out_len;
    return VK_ERROR_FEATURE_NOT_PRESENT;
}

VkResult vkblas_jit_create_shader_module(VkDevice device,
    const uint8_t* spirv, size_t spirv_len, VkShaderModule* module)
{
    (void)device; (void)spirv; (void)spirv_len; (void)module;
    return VK_ERROR_FEATURE_NOT_PRESENT;
}

VkResult vkblas_jit_compile_hip(const char* source, size_t source_len,
    void** out_code, size_t* out_len)
{
    (void)source; (void)source_len; (void)out_code; (void)out_len;
    return VK_ERROR_FEATURE_NOT_PRESENT;
}

VkResult vkblas_jit_load_hip_module(const void* code, size_t code_len,
    const char* func_name,
    void* module, void* kernel)
{
    (void)code; (void)code_len; (void)func_name; (void)module; (void)kernel;
    return VK_ERROR_FEATURE_NOT_PRESENT;
}

#endif
