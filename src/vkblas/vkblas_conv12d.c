/**
 * \file vkblas_conv12d.c
 * \brief Native Vulkan 1D/2D convolution (register-blocked direct, RB=2).
 *
 * conv1d and conv2d are dispatched through the same 3D rb2 kernel: conv2d
 * uses di=1, kd=1; conv1d uses di=hi=1, kd=kh=1.
 */
#include <string.h>

#include "vkblas/vkblas.h"
#include "vkblas_internal.h"

VkResult vkblas_conv2d_f32(
    VkBLASContext* ctx,
    uint32_t n, uint32_t c, uint32_t hi, uint32_t wi,
    uint32_t k, uint32_t dh, uint32_t dw,
    uint32_t kh, uint32_t kw,
    uint32_t pad_h, uint32_t pad_w,
    uint32_t stride_h, uint32_t stride_w,
    uint32_t dil_h, uint32_t dil_w,
    float alpha, VkBuffer x, VkBuffer w, float beta, VkBuffer y,
    VkCommandBuffer cmd)
{
    return vkblas_conv_common(ctx, cmd, VKBLAS_DTYPE_CONV_RB2_F32,
                              n, c, 1u, hi, wi, k, 1u, dh, dw,
                              1u, kh, kw, 0u, pad_h, pad_w,
                              1u, stride_h, stride_w,
                              1u, dil_h, dil_w,
                              alpha, x, w, beta, y);
}

VkResult vkblas_conv1d_f32(
    VkBLASContext* ctx,
    uint32_t n, uint32_t c, uint32_t li,
    uint32_t k, uint32_t lo,
    uint32_t kl,
    uint32_t pad_l, uint32_t stride_l, uint32_t dil_l,
    float alpha, VkBuffer x, VkBuffer w, float beta, VkBuffer y,
    VkCommandBuffer cmd)
{
    return vkblas_conv_common(ctx, cmd, VKBLAS_DTYPE_CONV_RB2_F32,
                              n, c, 1u, 1u, li, k, 1u, 1u, lo,
                              1u, 1u, kl, 0u, 0u, pad_l,
                              1u, 1u, stride_l,
                              1u, 1u, dil_l,
                              alpha, x, w, beta, y);
}
