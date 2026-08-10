#define __HIP_PLATFORM_AMD__ 1
#include "vkblas/vkblas.h"
#include "vkblas/vkblas_internal.h"
#include <hip/hip_runtime.h>
#include <hipblas/hipblaslt.h>
#include <stdlib.h>
#include <string.h>

/* HIP function pointer table — loaded lazily to avoid static linking */
typedef struct {
    void* hip_lib;
    void* hipblaslt_lib;
    hipblasLtHandle_t lt_handle;
    
    /* HIP functions */
    void* (*hipHostMalloc)(void**, size_t, unsigned int);
    void (*hipHostFree)(void*);
    void (*hipMalloc)(void**, size_t);
    void (*hipFree)(void*);
    hipError_t (*hipMemcpyHtoD)(hipDeviceptr_t, const void*, size_t);
    hipError_t (*hipMemcpyDtoH)(void*, hipDeviceptr_t, size_t);
    void (*hipDeviceSynchronize)(void);
    
    /* hipblasLt functions */
    hipblasStatus_t (*hipblasLtCreate)(hipblasLtHandle_t*);
    hipblasStatus_t (*hipblasLtDestroy)(hipblasLtHandle_t);
    hipblasStatus_t (*hipblasLtMatmul)(hipblasLtHandle_t, hipblasLtMatmulDesc_t, const void*,
        hipblasLtMatrixLayout_t, hipblasLtMatrixLayout_t, const void*,
        hipblasLtMatrixLayout_t, const void*, void*, hipblasLtMatrixLayout_t,
        hipblasLtMatmulDesc_t, int32_t, int32_t, float, hipStream_t);
} vkblas_hip_state_t;

static vkblas_hip_state_t g_hip = {0};

static VkBool32 vkblas_hip_loaded = VK_FALSE;

VkBool32 vkblas_hip_available(void) {
#ifdef USE_HIP
    if (!vkblas_hip_loaded) {
        /* Lazy load — will fail gracefully if HIP not installed */
        return VK_FALSE;
    }
    return g_hip.hip_lib != NULL && g_hip.hipblaslt_lib != NULL;
#else
    return VK_FALSE;
#endif
}

VkResult vkblas_hip_init(vkblas_context_t* ctx) {
#ifdef USE_HIP
    /* Load HIP runtime dynamically */
    g_hip.hip_lib = dlopen("hipblaslt.dll", RTLD_LAZY);
    if (!g_hip.hip_lib) {
        g_hip.hip_lib = dlopen("libamdhip64.so", RTLD_LAZY);
    }
    if (!g_hip.hip_lib) return VK_ERROR_FEATURE_NOT_PRESENT;

    /* Load hipblasLt */
    g_hip.hipblaslt_lib = dlopen("hipblaslt.dll", RTLD_LAZY);
    if (!g_hip.hipblaslt_lib) {
        g_hip.hipblaslt_lib = dlopen("libhipblaslt.so", RTLD_LAZY);
    }

    if (!g_hip.hipblaslt_lib) {
        dlclose(g_hip.hip_lib);
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    /* Resolve function pointers */
    g_hip.hipHostMalloc = (void*)dlsym(g_hip.hip_lib, "hipHostMalloc");
    g_hip.hipHostFree = (void*)dlsym(g_hip.hip_lib, "hipHostFree");
    g_hip.hipMalloc = (void*)dlsym(g_hip.hip_lib, "hipMalloc");
    g_hip.hipFree = (void*)dlsym(g_hip.hip_lib, "hipFree");
    g_hip.hipMemcpyHtoD = (void*)dlsym(g_hip.hip_lib, "hipMemcpyHtoD");
    g_hip.hipMemcpyDtoH = (void*)dlsym(g_hip.hip_lib, "hipMemcpyDtoH");
    g_hip.hipDeviceSynchronize = (void*)dlsym(g_hip.hip_lib, "hipDeviceSynchronize");

    g_hip.hipblasLtCreate = (void*)dlsym(g_hip.hipblaslt_lib, "hipblasLtCreate");
    g_hip.hipblasLtDestroy = (void*)dlsym(g_hip.hipblaslt_lib, "hipblasLtDestroy");
    g_hip.hipblasLtMatmul = (void*)dlsym(g_hip.hipblaslt_lib, "hipblasLtMatmul");

    if (!g_hip.hipblasLtCreate || !g_hip.hipblasLtMatmul) {
        dlclose(g_hip.hip_lib);
        dlclose(g_hip.hipblaslt_lib);
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    /* Initialize hipblasLt handle */
    hipblasStatus_t status = g_hip.hipblasLtCreate(&g_hip.lt_handle);
    if (status != HIPBLAS_STATUS_SUCCESS) {
        dlclose(g_hip.hip_lib);
        dlclose(g_hip.hipblaslt_lib);
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    vkblas_hip_loaded = VK_TRUE;
#endif
    return VK_SUCCESS;
}

void vkblas_hip_cleanup(void) {
#ifdef USE_HIP
    if (g_hip.lt_handle) {
        g_hip.hipblasLtDestroy(g_hip.lt_handle);
    }
    if (g_hip.hip_lib) dlclose(g_hip.hip_lib);
    if (g_hip.hipblaslt_lib) dlclose(g_hip.hipblaslt_lib);
    memset(&g_hip, 0, sizeof(g_hip));
    vkblas_hip_loaded = VK_FALSE;
#endif
}

/*
 * Bridge: VKBLAS cooperative matrix → HIP hipblasLt matmul
 * Handles the driver 26.7.1 crash by bypassing Vulkan coop matrix entirely
 */
VkResult vkblas_hip_gemm_f16(
    vkblas_context_t* ctx,
    uint32_t m, uint32_t n, uint32_t k,
    const void* a, const void* b, void* c,
    float alpha_f32, float beta_f32
) {
#ifndef USE_HIP
    return VK_ERROR_FEATURE_NOT_PRESENT;
#endif
    
    if (!vkblas_hip_available()) {
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    /*
     * HIP bridge workflow:
     * 1. a/b/c are VkBuffers (device-local memory)
     * 2. Get device addresses from Vulkan buffers
     * 3. Create HIP device pointers from those addresses
     * 4. Call hipblasLtMatmul with FP16 inputs, FP32 accumulator
     * 5. Result written back to same device memory
     * 6. Signal completion via Vulkan fence (no extra copy)
     */

    /* Step 1-3: Map Vulkan buffer device addresses to HIP device pointers */
    VkDeviceAddress a_addr = vkblas_get_buffer_device_address(ctx, a);
    VkDeviceAddress b_addr = vkblas_get_buffer_device_address(ctx, b);
    VkDeviceAddress c_addr = vkblas_get_buffer_device_address(ctx, c);

    hipDeviceptr_t hip_a = (hipDeviceptr_t)(uintptr_t)a_addr;
    hipDeviceptr_t hip_b = (hipDeviceptr_t)(uintptr_t)b_addr;
    hipDeviceptr_t hip_c = (hipDeviceptr_t)(uintptr_t)c_addr;

    /* Step 4: Configure hipblasLt operation */
    hipblasLtMatmulDesc_t operationDesc;
    g_hip.hipblasLtMatmulDescCreate(&operationDesc, HIPBLAS_COMPUTE_32F, HIPBLAS_R_32F);
    g_hip.hipblasLtMatmulDescSetAttribute(operationDesc,
        HIPBLASLT_MATMUL_DESC_TRANSA, &(hipblasOperation_t){HIPBLAS_OP_T});
    g_hip.hipblasLtMatmulDescSetAttribute(operationDesc,
        HIPBLASLT_MATMUL_DESC_TRANSB, &(hipblasOperation_t){HIPBLAS_OP_N});

    /* Matrix layouts: row-major FP16 */
    hipblasLtMatrixLayout_t matA, matB, matC;
    int32_t ld_a = k, ld_b = k, ld_c = n;  /* leading dimensions */
    g_hip.hipblasLtMatrixLayoutCreate(&matA, HIPBLAS_R_16F, k, m, ld_a);
    g_hip.hipblasLtMatrixLayoutCreate(&matB, HIPBLAS_R_16F, n, k, ld_b);
    g_hip.hipblasLtMatrixLayoutCreate(&matC, HIPBLAS_R_16F, n, m, ld_c);

    /* Algorithm selection */
    hipblasLtMatmulAlgo_t algo;
    g_hip.hipblasLtMatmulAlgoCapGet(g_hip.lt_handle, HIPBLASLT_MATMUL_CAP_ROCM_VERSION,
        sizeof(algo), &algo, NULL);
    g_hip.hipblasLtMatmulAlgoInit(g_hip.lt_handle, 0, 0, HIPBLASLT_MATMUL_ALGO_CONFIG_ID_0, &algo);

    /* Scale factors */
    float scale_a = 1.0f, scale_b = 1.0f, scale_c = alpha_f32;
    void* alpha = &scale_a;
    void* beta = &beta_f32;

    /* Execution — uses same CUDA/HIP stream as Vulkan queue */
    hipStream_t stream = 0;  /* default stream = same as Vulkan */
    g_hip.hipblasLtMatmul(g_hip.lt_handle, operationDesc,
        alpha, hip_b, matB, hip_a, matA, beta,
        hip_c, matC, hip_c, matC, &algo, 0, 0, 0, stream);

    /* Cleanup */
    g_hip.hipblasLtMatmulAlgoClear(&algo);
    g_hip.hipblasLtMatrixLayoutDestroy(matA);
    g_hip.hipblasLtMatrixLayoutDestroy(matB);
    g_hip.hipblasLtMatrixLayoutDestroy(matC);
    g_hip.hipblasLtMatmulDescDestroy(operationDesc);

    /* Synchronize — this is the bridge boundary */
    g_hip.hipDeviceSynchronize();

    return VK_SUCCESS;
}
