/**
 * \file vkblas_conv3d.c
 * \brief Native Vulkan 3D convolution (register-blocked direct, RB=2).
 *
 * The kernel is the benchmark winner on RDNA2/RDNA4: 512-thread workgroups,
 * one workgroup per (n, k), each thread computes 2 spatial outputs sharing
 * input taps. f16/bf16/f64 variants can be added by embedding their rb2
 * shaders and exposing new dtype codes.
 */
#include <string.h>

#include "vkblas/vkblas.h"
#include "vkblas_internal.h"

VkResult vkblas_conv3d_f32(
    VkBLASContext* ctx,
    uint32_t n, uint32_t c, uint32_t di, uint32_t hi, uint32_t wi,
    uint32_t k, uint32_t dd, uint32_t dh, uint32_t dw,
    uint32_t kd, uint32_t kh, uint32_t kw,
    uint32_t pad_d, uint32_t pad_h, uint32_t pad_w,
    uint32_t stride_d, uint32_t stride_h, uint32_t stride_w,
    uint32_t dil_d, uint32_t dil_h, uint32_t dil_w,
    float alpha, VkBuffer x, VkBuffer w, float beta, VkBuffer y,
    VkCommandBuffer cmd)
{
    return vkblas_conv_common(ctx, cmd, VKBLAS_DTYPE_CONV_RB2_F32,
                              n, c, di, hi, wi, k, dd, dh, dw,
                              kd, kh, kw, pad_d, pad_h, pad_w,
                              stride_d, stride_h, stride_w,
                              dil_d, dil_h, dil_w,
                              alpha, x, w, beta, y);
}
