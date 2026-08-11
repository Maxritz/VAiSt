/**
 * \file vkblas_conv12d.c
 * \brief 1D/2D convolution via MIOpen VJITC bridge (mirrors conv3d pattern).
 */
#define __HIP_PLATFORM_AMD__ 1

#include "vkblas/vkblas.h"
#include "vkblas_internal.h"
#include <hip/hip_runtime.h>
#include <windows.h>
#include <string.h>

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

typedef miopenStatus_t (*PFN_miopenCreate)(miopenHandle_t** handle);
typedef miopenStatus_t (*PFN_miopenDestroy)(miopenHandle_t* handle);
typedef miopenStatus_t (*PFN_miopenCreateTensorDescriptor)(miopenTensorDescriptor_t* desc);
typedef miopenStatus_t (*PFN_miopenDestroyTensorDescriptor)(miopenTensorDescriptor_t desc);
typedef miopenStatus_t (*PFN_miopenSetTensorDescriptor)(miopenTensorDescriptor_t desc,
    miopenDataType_t dataType, int nbDims, const int* dimsA, const int* stridesA);
typedef miopenStatus_t (*PFN_miopenSet4dTensorDescriptor)(miopenTensorDescriptor_t desc,
    miopenDataType_t dataType, int n, int c, int h, int w);
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

static VkResult load_miopen_symbols(HMODULE lib,
    PFN_miopenCreate* p_create, PFN_miopenDestroy* p_destroy,
    PFN_miopenCreateTensorDescriptor* p_createTensor, PFN_miopenDestroyTensorDescriptor* p_destroyTensor,
    PFN_miopenSetTensorDescriptor* p_setTensor, PFN_miopenSet4dTensorDescriptor* p_set4dTensor,
    PFN_miopenCreateConvolutionDescriptor* p_createConv, PFN_miopenDestroyConvolutionDescriptor* p_destroyConv,
    PFN_miopenInitConvolutionNdDescriptor* p_initConvNd,
    PFN_miopenConvolutionForward* p_convFwd,
    PFN_miopenConvolutionForwardGetWorkSpaceSize* p_getWorkSpaceSize,
    PFN_miopenFindConvolutionForwardAlgorithm* p_findAlgo)
{
    *p_create       = (PFN_miopenCreate)GetProcAddress(lib, "miopenCreate");
    *p_destroy      = (PFN_miopenDestroy)GetProcAddress(lib, "miopenDestroy");
    *p_createTensor = (PFN_miopenCreateTensorDescriptor)GetProcAddress(lib, "miopenCreateTensorDescriptor");
    *p_destroyTensor= (PFN_miopenDestroyTensorDescriptor)GetProcAddress(lib, "miopenDestroyTensorDescriptor");
    *p_setTensor    = (PFN_miopenSetTensorDescriptor)GetProcAddress(lib, "miopenSetTensorDescriptor");
    *p_set4dTensor  = (PFN_miopenSet4dTensorDescriptor)GetProcAddress(lib, "miopenSet4dTensorDescriptor");
    *p_createConv   = (PFN_miopenCreateConvolutionDescriptor)GetProcAddress(lib, "miopenCreateConvolutionDescriptor");
    *p_destroyConv  = (PFN_miopenDestroyConvolutionDescriptor)GetProcAddress(lib, "miopenDestroyConvolutionDescriptor");
    *p_initConvNd   = (PFN_miopenInitConvolutionNdDescriptor)GetProcAddress(lib, "miopenInitConvolutionNdDescriptor");
    *p_convFwd      = (PFN_miopenConvolutionForward)GetProcAddress(lib, "miopenConvolutionForward");
    *p_getWorkSpaceSize = (PFN_miopenConvolutionForwardGetWorkSpaceSize)GetProcAddress(lib, "miopenConvolutionForwardGetWorkSpaceSize");
    *p_findAlgo     = (PFN_miopenFindConvolutionForwardAlgorithm)GetProcAddress(lib, "miopenFindConvolutionForwardAlgorithm");

    return (*p_create && *p_destroy && *p_createTensor && *p_destroyTensor &&
            *p_setTensor && *p_set4dTensor && *p_createConv && *p_destroyConv && *p_initConvNd &&
            *p_convFwd && *p_getWorkSpaceSize && *p_findAlgo)
        ? VK_SUCCESS : VK_ERROR_FEATURE_NOT_PRESENT;
}

VkResult vkblas_conv1d_f32(
    VkBLASContext* ctx,
    uint32_t n, uint32_t c, uint32_t li,
    uint32_t k, uint32_t lo,
    uint32_t kl,
    uint32_t pad_l, uint32_t stride_l, uint32_t dil_l,
    float alpha, void* x, void* w, float beta, void* y,
    VkCommandBuffer cmd)
{
    if (!ctx || !x || !w || !y) return VK_ERROR_INITIALIZATION_FAILED;
    (void)cmd;

    HMODULE lib = LoadLibraryA("miopen.dll");
    if (!lib) return VK_ERROR_FEATURE_NOT_PRESENT;

    PFN_miopenCreate p_create; PFN_miopenDestroy p_destroy;
    PFN_miopenCreateTensorDescriptor p_createTensor; PFN_miopenDestroyTensorDescriptor p_destroyTensor;
    PFN_miopenSetTensorDescriptor p_setTensor; PFN_miopenSet4dTensorDescriptor p_set4dTensor;
    PFN_miopenCreateConvolutionDescriptor p_createConv; PFN_miopenDestroyConvolutionDescriptor p_destroyConv;
    PFN_miopenInitConvolutionNdDescriptor p_initConvNd;
    PFN_miopenConvolutionForward p_convFwd;
    PFN_miopenConvolutionForwardGetWorkSpaceSize p_getWorkSpaceSize;
    PFN_miopenFindConvolutionForwardAlgorithm p_findAlgo;

    if (load_miopen_symbols(lib, &p_create, &p_destroy, &p_createTensor, &p_destroyTensor,
        &p_setTensor, &p_set4dTensor, &p_createConv, &p_destroyConv, &p_initConvNd, &p_convFwd,
        &p_getWorkSpaceSize, &p_findAlgo) != VK_SUCCESS) {
        FreeLibrary(lib); return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    miopenHandle_t* handle;
    if (p_create(&handle) != miopenStatusSuccess) { FreeLibrary(lib); return VK_ERROR_INITIALIZATION_FAILED; }

    /* conv1d via 4D NCHW with H=1: x=(n,c,1,li), w=(k,c,1,kl), y=(n,k,1,lo) */
    miopenTensorDescriptor_t x_desc, w_desc, y_desc;
    p_createTensor(&x_desc); p_createTensor(&w_desc); p_createTensor(&y_desc);
    p_set4dTensor(x_desc, miopenFloat, (int)n, (int)c, 1, (int)li);
    p_set4dTensor(w_desc, miopenFloat, (int)k, (int)c, 1, (int)kl);
    p_set4dTensor(y_desc, miopenFloat, (int)n, (int)k, 1, (int)lo);

    miopenConvolutionDescriptor_t conv_desc;
    p_createConv(&conv_desc);
    int pads[2]    = {0, (int)pad_l};
    int strides[2] = {1, (int)stride_l};
    int dilations[2]= {1, (int)dil_l};
    p_initConvNd(conv_desc, 2, pads, strides, dilations, miopenConvolution);

    miopenConvAlgoPerf_t algo_perf; int algo_count = 1;
    p_findAlgo(handle, x_desc, x, w_desc, w, conv_desc, y_desc, y,
        1, &algo_count, &algo_perf, NULL, 0, 0);

    size_t wsz = 0;
    p_getWorkSpaceSize(handle, w_desc, x_desc, conv_desc, y_desc, &wsz);
    void* ws = NULL; if (wsz > 0) hipMalloc(&ws, wsz);

    miopenStatus_t status = p_convFwd(handle, &alpha, x_desc, x, w_desc, w,
        conv_desc, algo_perf.fwd_algo, &beta, y_desc, y, ws, wsz);

    if (ws) hipFree(ws);
    p_destroyTensor(x_desc); p_destroyTensor(w_desc); p_destroyTensor(y_desc);
    p_destroyConv(conv_desc);
    p_destroy(handle); FreeLibrary(lib);

    return (status == miopenStatusSuccess) ? VK_SUCCESS : VK_ERROR_UNKNOWN;
}

VkResult vkblas_conv2d_f32(
    VkBLASContext* ctx,
    uint32_t n, uint32_t c, uint32_t hi, uint32_t wi,
    uint32_t k, uint32_t dh, uint32_t dw,
    uint32_t kh, uint32_t kw,
    uint32_t pad_h, uint32_t pad_w,
    uint32_t stride_h, uint32_t stride_w,
    uint32_t dil_h, uint32_t dil_w,
    float alpha, void* x, void* w, float beta, void* y,
    VkCommandBuffer cmd)
{
    if (!ctx || !x || !w || !y) return VK_ERROR_INITIALIZATION_FAILED;
    (void)cmd;

    HMODULE lib = LoadLibraryA("miopen.dll");
    if (!lib) return VK_ERROR_FEATURE_NOT_PRESENT;

    PFN_miopenCreate p_create; PFN_miopenDestroy p_destroy;
    PFN_miopenCreateTensorDescriptor p_createTensor; PFN_miopenDestroyTensorDescriptor p_destroyTensor;
    PFN_miopenSetTensorDescriptor p_setTensor; PFN_miopenSet4dTensorDescriptor p_set4dTensor;
    PFN_miopenCreateConvolutionDescriptor p_createConv; PFN_miopenDestroyConvolutionDescriptor p_destroyConv;
    PFN_miopenInitConvolutionNdDescriptor p_initConvNd;
    PFN_miopenConvolutionForward p_convFwd;
    PFN_miopenConvolutionForwardGetWorkSpaceSize p_getWorkSpaceSize;
    PFN_miopenFindConvolutionForwardAlgorithm p_findAlgo;

    if (load_miopen_symbols(lib, &p_create, &p_destroy, &p_createTensor, &p_destroyTensor,
        &p_setTensor, &p_set4dTensor, &p_createConv, &p_destroyConv, &p_initConvNd, &p_convFwd,
        &p_getWorkSpaceSize, &p_findAlgo) != VK_SUCCESS) {
        FreeLibrary(lib); return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    miopenHandle_t* handle;
    if (p_create(&handle) != miopenStatusSuccess) { FreeLibrary(lib); return VK_ERROR_INITIALIZATION_FAILED; }

    /* Input (NCHW), Weight (KCHW), Output (NKDHW) — 4D tensors */
    miopenTensorDescriptor_t x_desc, w_desc, y_desc;
    p_createTensor(&x_desc); p_createTensor(&w_desc); p_createTensor(&y_desc);
    p_set4dTensor(x_desc, miopenFloat, (int)n, (int)c, (int)hi, (int)wi);
    p_set4dTensor(w_desc, miopenFloat, (int)k, (int)c, (int)kh, (int)kw);
    p_set4dTensor(y_desc, miopenFloat, (int)n, (int)k, (int)dh, (int)dw);

    miopenConvolutionDescriptor_t conv_desc;
    p_createConv(&conv_desc);
    int pads[2]    = {(int)pad_h, (int)pad_w};
    int strides[2] = {(int)stride_h, (int)stride_w};
    int dilations[2]= {(int)dil_h, (int)dil_w};
    p_initConvNd(conv_desc, 2, pads, strides, dilations, miopenConvolution);

    miopenConvAlgoPerf_t algo_perf; int algo_count = 1;
    p_findAlgo(handle, x_desc, x, w_desc, w, conv_desc, y_desc, y,
        1, &algo_count, &algo_perf, NULL, 0, 0);

    size_t wsz = 0;
    p_getWorkSpaceSize(handle, w_desc, x_desc, conv_desc, y_desc, &wsz);
    void* ws = NULL; if (wsz > 0) hipMalloc(&ws, wsz);

    miopenStatus_t status = p_convFwd(handle, &alpha, x_desc, x, w_desc, w,
        conv_desc, algo_perf.fwd_algo, &beta, y_desc, y, ws, wsz);

    if (ws) hipFree(ws);
    p_destroyTensor(x_desc); p_destroyTensor(w_desc); p_destroyTensor(y_desc);
    p_destroyConv(conv_desc);
    p_destroy(handle); FreeLibrary(lib);

    return (status == miopenStatusSuccess) ? VK_SUCCESS : VK_ERROR_UNKNOWN;
}
