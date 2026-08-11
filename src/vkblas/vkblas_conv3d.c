/**
 * \file vkblas_conv3d.c
 * \brief 3D convolution via MIOpen VJITC bridge.
 *
 * Uses dynamic dispatch (LoadLibrary + GetProcAddress) to access MIOpen
 * functions at runtime, avoiding header incompatibilities. This means
 * MIOpen.dll is loaded at runtime — the test should be skipped gracefully
 * if the DLL is not present.
 *
 * Buffer arguments are HIP device pointers shared with Vulkan VkBuffers
 * via VK_EXT_external_memory_host (see vkstream.h).
 */
#define __HIP_PLATFORM_AMD__ 1

#include "vkblas/vkblas.h"
#include "vkblas_internal.h"
#include <hip/hip_runtime.h>
#include <windows.h>
#include <string.h>

/* ── Opaque MIOpen types (we only pass pointers to them) ──────────────── */
typedef void miopenHandle_t;
typedef void* miopenTensorDescriptor_t;
typedef void* miopenConvolutionDescriptor_t;
typedef int miopenDataType_t;
typedef int miopenConvolutionMode_t;
typedef int miopenConvFwdAlgorithm_t;
typedef int miopenStatus_t;

#define miopenStatusSuccess 0
#define miopenFloat         1
#define miopenConvolution   0

typedef struct miopenConvAlgoPerf_t {
    miopenConvFwdAlgorithm_t fwd_algo;
    int bwd_data_algo;
    int bwd_weights_algo;
    void* reserved;
    size_t workspace;
    double time;
} miopenConvAlgoPerf_t;

/* ── MIOpen function pointer types (match exported C names) ──────────── */
typedef miopenStatus_t (*PFN_miopenCreate)(miopenHandle_t** handle);
typedef miopenStatus_t (*PFN_miopenDestroy)(miopenHandle_t* handle);
typedef miopenStatus_t (*PFN_miopenCreateTensorDescriptor)(miopenTensorDescriptor_t* desc);
typedef miopenStatus_t (*PFN_miopenDestroyTensorDescriptor)(miopenTensorDescriptor_t desc);
typedef miopenStatus_t (*PFN_miopenSetTensorDescriptor)(miopenTensorDescriptor_t desc,
    miopenDataType_t dataType, int nbDims, const int* dimsA, const int* stridesA);
typedef miopenStatus_t (*PFN_miopenCreateConvolutionDescriptor)(miopenConvolutionDescriptor_t* desc);
typedef miopenStatus_t (*PFN_miopenDestroyConvolutionDescriptor)(miopenConvolutionDescriptor_t desc);
typedef miopenStatus_t (*PFN_miopenInitConvolutionNdDescriptor)(miopenConvolutionDescriptor_t convDesc,
    int spatialDim, const int* padA, const int* strideA, const int* dilationA, miopenConvolutionMode_t mode);
typedef miopenStatus_t (*PFN_miopenConvolutionForward)(miopenHandle_t* handle,
    const void* alpha, const miopenTensorDescriptor_t xDesc, const void* x,
    const miopenTensorDescriptor_t wDesc, const void* w,
    const miopenConvolutionDescriptor_t convDesc, miopenConvFwdAlgorithm_t algo,
    const void* beta, const miopenTensorDescriptor_t yDesc, void* y,
    void* workSpace, size_t workSpaceSize);
typedef miopenStatus_t (*PFN_miopenConvolutionForwardGetWorkSpaceSize)(miopenHandle_t* handle,
    const miopenTensorDescriptor_t wDesc, const miopenTensorDescriptor_t xDesc,
    const miopenConvolutionDescriptor_t convDesc, const miopenTensorDescriptor_t yDesc, size_t* workSpaceSize);
typedef miopenStatus_t (*PFN_miopenFindConvolutionForwardAlgorithm)(miopenHandle_t* handle,
    const miopenTensorDescriptor_t xDesc, const void* x,
    const miopenTensorDescriptor_t wDesc, const void* w,
    const miopenConvolutionDescriptor_t convDesc,
    const miopenTensorDescriptor_t yDesc, void* y,
    const int requestAlgoCount, int* returnedAlgoCount,
    miopenConvAlgoPerf_t* perfResults,
    void* workSpace, size_t workSpaceSize, int exhaustiveSearch);

VkResult vkblas_conv3d_f32(
    VkBLASContext* ctx,
    uint32_t n, uint32_t c, uint32_t di, uint32_t hi, uint32_t wi,
    uint32_t k, uint32_t dd, uint32_t dh, uint32_t dw,
    uint32_t kd, uint32_t kh, uint32_t kw,
    uint32_t pad_d, uint32_t pad_h, uint32_t pad_w,
    uint32_t stride_d, uint32_t stride_h, uint32_t stride_w,
    uint32_t dil_d, uint32_t dil_h, uint32_t dil_w,
    float alpha, void* x, void* w, float beta, void* y,
    VkCommandBuffer cmd)
{
    if (!ctx || !x || !w || !y) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    (void)cmd;

    HMODULE miopen_lib = LoadLibraryA("miopen.dll");
    if (!miopen_lib) {
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    PFN_miopenCreate p_miopenCreate = (PFN_miopenCreate)GetProcAddress(miopen_lib, "miopenCreate");
    PFN_miopenDestroy p_miopenDestroy = (PFN_miopenDestroy)GetProcAddress(miopen_lib, "miopenDestroy");
    PFN_miopenCreateTensorDescriptor p_createTensor = (PFN_miopenCreateTensorDescriptor)GetProcAddress(miopen_lib, "miopenCreateTensorDescriptor");
    PFN_miopenDestroyTensorDescriptor p_destroyTensor = (PFN_miopenDestroyTensorDescriptor)GetProcAddress(miopen_lib, "miopenDestroyTensorDescriptor");
    PFN_miopenSetTensorDescriptor p_setTensor = (PFN_miopenSetTensorDescriptor)GetProcAddress(miopen_lib, "miopenSetTensorDescriptor");
    PFN_miopenCreateConvolutionDescriptor p_createConv = (PFN_miopenCreateConvolutionDescriptor)GetProcAddress(miopen_lib, "miopenCreateConvolutionDescriptor");
    PFN_miopenDestroyConvolutionDescriptor p_destroyConv = (PFN_miopenDestroyConvolutionDescriptor)GetProcAddress(miopen_lib, "miopenDestroyConvolutionDescriptor");
    PFN_miopenInitConvolutionNdDescriptor p_initConvNd = (PFN_miopenInitConvolutionNdDescriptor)GetProcAddress(miopen_lib, "miopenInitConvolutionNdDescriptor");
    PFN_miopenConvolutionForward p_convFwd = (PFN_miopenConvolutionForward)GetProcAddress(miopen_lib, "miopenConvolutionForward");
    PFN_miopenConvolutionForwardGetWorkSpaceSize p_getWorkSpaceSize = (PFN_miopenConvolutionForwardGetWorkSpaceSize)GetProcAddress(miopen_lib, "miopenConvolutionForwardGetWorkSpaceSize");
    PFN_miopenFindConvolutionForwardAlgorithm p_findAlgo = (PFN_miopenFindConvolutionForwardAlgorithm)GetProcAddress(miopen_lib, "miopenFindConvolutionForwardAlgorithm");

    if (!p_miopenCreate || !p_miopenDestroy || !p_createTensor || !p_destroyTensor ||
        !p_setTensor || !p_createConv || !p_destroyConv || !p_initConvNd ||
        !p_convFwd || !p_getWorkSpaceSize || !p_findAlgo) {
        FreeLibrary(miopen_lib);
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    /* Create MIOpen handle */
    miopenHandle_t* handle;
    if (p_miopenCreate(&handle) != miopenStatusSuccess) {
        FreeLibrary(miopen_lib);
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    /* Input tensor descriptor (NCDHW) */
    miopenTensorDescriptor_t x_desc;
    p_createTensor(&x_desc);
    int in_lengths[5] = {(int)n, (int)c, (int)di, (int)hi, (int)wi};
    int in_strides[5] = {(int)(c * di * hi * wi), (int)(di * hi * wi),
                         (int)(hi * wi), (int)wi, 1};
    p_setTensor(x_desc, miopenFloat, 5, in_lengths, in_strides);

    /* Weight tensor descriptor (KCDHW) */
    miopenTensorDescriptor_t w_desc;
    p_createTensor(&w_desc);
    int w_lengths[5] = {(int)k, (int)c, (int)kd, (int)kh, (int)kw};
    int w_strides[5] = {(int)(c * kd * kh * kw), (int)(kd * kh * kw),
                        (int)(kh * kw), (int)kw, 1};
    p_setTensor(w_desc, miopenFloat, 5, w_lengths, w_strides);

    /* Output tensor descriptor (NKDHW) */
    miopenTensorDescriptor_t y_desc;
    p_createTensor(&y_desc);
    int out_lengths[5] = {(int)n, (int)k, (int)dd, (int)dh, (int)dw};
    int out_strides[5] = {(int)(k * dd * dh * dw), (int)(dd * dh * dw),
                          (int)(dh * dw), (int)dw, 1};
    p_setTensor(y_desc, miopenFloat, 5, out_lengths, out_strides);

    /* Convolution descriptor (3D) */
    miopenConvolutionDescriptor_t conv_desc;
    p_createConv(&conv_desc);
    int pads[3]    = {(int)pad_d, (int)pad_h, (int)pad_w};
    int strides[3] = {(int)stride_d, (int)stride_h, (int)stride_w};
    int dilations[3]= {(int)dil_d, (int)dil_h, (int)dil_w};
    p_initConvNd(conv_desc, 3, pads, strides, dilations, miopenConvolution);

    /* Find algorithm */
    miopenConvAlgoPerf_t algo_perf;
    int algo_count = 1;
    p_findAlgo(handle, x_desc, x, w_desc, w, conv_desc, y_desc, y,
        1, &algo_count, &algo_perf, NULL, 0, 0);

    miopenConvFwdAlgorithm_t algo = algo_perf.fwd_algo;

    /* Get workspace size */
    size_t workspace_size = 0;
    p_getWorkSpaceSize(handle, w_desc, x_desc, conv_desc, y_desc, &workspace_size);

    void* workspace = NULL;
    if (workspace_size > 0) {
        hipMalloc(&workspace, workspace_size);
    }

    /* Execute forward convolution */
    miopenStatus_t status = p_convFwd(handle,
        &alpha, x_desc, x, w_desc, w, conv_desc, algo,
        &beta, y_desc, y, workspace, workspace_size);

    /* Cleanup */
    if (workspace) hipFree(workspace);
    p_destroyTensor(x_desc);
    p_destroyTensor(w_desc);
    p_destroyTensor(y_desc);
    p_destroyConv(conv_desc);
    p_miopenDestroy(handle);
    FreeLibrary(miopen_lib);

    return (status == miopenStatusSuccess) ? VK_SUCCESS : VK_ERROR_UNKNOWN;
}
