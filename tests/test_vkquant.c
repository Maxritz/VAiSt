/**
 * \file test_vkquant.c
 * \brief Public-API test harness for the VKQuant library.
 *
 * Bootstraps a minimal Vulkan instance + physical/logical device, creates a
 * VkQuantContext, and validates every dequantization / forward-quantization
 * op against CPU references implementing the exact ggml block byte formats
 * (ggml-common.h + ggml-quants.c, ported bit-for-bit).
 *
 * Dequant tests: synthesize deterministic block bytes, run the GPU dequant,
 * compare against the CPU reference (1e-6 tolerance, bit-exact in practice).
 * Quant tests: quantize known f32 blocks on the GPU, read back, dequantize
 * through our own dequant, assert per-element error < 1e-1 (prints max err).
 *
 * Exit status: 0 when all checks pass. Returns 1 on any real failure.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../include/vkquant/vkquant.h"
#include "vkquant_tables.h"

/* ===========================================================================
 * Test parameters
 * ========================================================================== */

#define TEST_NUM_BLOCKS    8u    /**< Blocks per legacy 32-elem quant format. */
#define TEST_Q8_BLOCK_SIZE 36u   /**< Bytes per Q8_0 block (4 + 32 int8).    */
#define TEST_Q4_BLOCK_SIZE 20u   /**< Bytes per Q4_0 block (4 + 16 nibbles). */
#define TEST_ELEMS_PER_BLOCK 32u /**< f32 elements per legacy block.         */
#define TEST_Q8_SCALE       0.5f /**< Q8_0 scale d.                          */
#define TEST_Q4_SCALE       1.5f /**< Q4_0 scale d.                          */
#define TEST_STAGING_SIZE  ((VkDeviceSize)(4u << 20))  /**< 4 MiB host buffer. */

#define TEST_Q4K_BLOCK_SIZE 144u  /**< Bytes per Q4_K block (ggml).          */
#define TEST_Q6K_BLOCK_SIZE 210u  /**< Bytes per Q6_K block (ggml).          */
#define TEST_IQ4XS_BLOCK_SIZE 136u/**< Bytes per IQ4_XS block (ggml).        */
#define TEST_K_BLOCKS       2u    /**< Super-blocks per K-format test.       */
#define TEST_K_ELEMS        (TEST_K_BLOCKS * 256u) /**< 512 f32 each.        */

#define TEST_QUANT_BLOCKS   4u    /**< 32-element blocks for legacy quant.   */
#define TEST_QUANT_ELEMS    (TEST_QUANT_BLOCKS * 32u) /**< 128 f32 each.     */
#define TEST_KQUANT_BLOCKS  2u    /**< 256-element blocks for K-quant.       */
#define TEST_KQUANT_ELEMS   (TEST_KQUANT_BLOCKS * 256u) /**< 512 f32 each.   */

#define TEST_F32_TOLERANCE 1e-6f  /**< dequant comparison tolerance.         */
#define TEST_QUANT_TOLERANCE 0.1f /**< quant round-trip error bound.         */

/* ===========================================================================
 * Harness state
 * ========================================================================== */

typedef struct {
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue queue;
    VkCommandPool cmd_pool;
    VkCommandBuffer cmd;
    VkDeviceMemory mem;
    VkBuffer staging;
    void *mapped;
    VkDeviceSize align;
    VkDeviceSize cursor;
    VkFence fence;
    VkQuantContext *quant_ctx;
} harness_t;

typedef struct {
    VkBuffer in;
    VkBuffer out;
    VkDeviceSize off_in;
    VkDeviceSize off_out;
    VkDeviceSize off_readback;
    VkDeviceSize off_expected;
} op_t;

typedef struct {
    VkBuffer in;
    VkBuffer qbytes;
    VkBuffer out;
    VkDeviceSize off_in;
    VkDeviceSize off_q;
    VkDeviceSize off_out;
    VkDeviceSize off_readback;
} rtop_t;

/* ===========================================================================
 * f16 helpers (IEEE half precision)
 * ========================================================================== */

static uint16_t rd16(const uint8_t *p)
{
    return (uint16_t)(p[0] | (p[1] << 8));
}

static float f16_to_f32(uint16_t h)
{
    uint32_t s = ((uint32_t)(h & 0x8000u)) << 16;
    uint32_t e = (h >> 10) & 0x1Fu;
    uint32_t m = h & 0x3FFu;
    uint32_t f;
    if (e == 0) {
        if (m == 0) {
            f = s;
        } else {
            e = 1;
            while (!(m & 0x400u)) { m <<= 1; e--; }
            m &= 0x3FFu;
            f = s | ((e + 112u) << 23) | (m << 13);
        }
    } else if (e == 0x1F) {
        f = s | 0x7F800000u | (m << 13);
    } else {
        f = s | ((e + 112u) << 23) | (m << 13);
    }
    float r;
    memcpy(&r, &f, 4);
    return r;
}

static uint16_t f32_to_f16(float f)
{
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp   = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    uint32_t expbits = (x >> 23) & 0xFFu;

    if (expbits == 0xFFu) { /* inf / nan */
        return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0));
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t shift = 14u - (uint32_t)exp;
        uint16_t half = (uint16_t)(mant >> shift);
        if (mant & (1u << (shift - 1u))) half++;
        return (uint16_t)(sign | half);
    }
    uint16_t half = (uint16_t)(((uint32_t)exp << 10) | (mant >> 13));
    if (mant & 0x1000u) half++;
    return (uint16_t)(sign | half);
}

static void write_f16(uint8_t *dst, float v)
{
    uint16_t h = f32_to_f16(v);
    dst[0] = (uint8_t)(h & 0xFF);
    dst[1] = (uint8_t)(h >> 8);
}

/* ===========================================================================
 * Deterministic pseudo-random source for dequant test blocks
 * ========================================================================== */

static uint32_t rng_state = 0xC0FFEE11u;

static uint32_t rng_next(void)
{
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}

/* ===========================================================================
 * CPU references (bit-exact ggml dequant, ggml-common.h / ggml-quants.c)
 * ========================================================================== */

static const int8_t iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113
};

static void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m)
{
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

static int grid_u64(const uint64_t *g, int idx, int j)
{
    return (int)((g[idx] >> (8 * j)) & 0xFF);
}

static int grid_u32(const uint32_t *g, int idx, int j)
{
    return (int)((g[idx] >> (8 * j)) & 0xFF);
}

static void ref_dequant_q4k(const uint8_t *bytes, float *out, int nb)
{
    for (int i = 0; i < nb; i++) {
        const uint8_t *b = bytes + (size_t)i * 144;
        const uint8_t *scales = b + 4;
        const uint8_t *qs     = b + 16;
        float d  = f16_to_f32(rd16(b));
        float dm = f16_to_f32(rd16(b + 2));
        int is = 0;
        uint8_t sc, mn;
        for (int j = 0; j < 256; j += 64) {
            get_scale_min_k4(is + 0, scales, &sc, &mn);
            float d1 = d * sc; float m1 = dm * mn;
            get_scale_min_k4(is + 1, scales, &sc, &mn);
            float d2 = d * sc; float m2 = dm * mn;
            const uint8_t *q = qs + (j / 64) * 32;
            for (int l = 0; l < 32; ++l) out[i * 256 + j + l]     = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) out[i * 256 + j + l + 32] = d2 * (q[l] >> 4) - m2;
            is += 2;
        }
    }
}

static void ref_dequant_q6k(const uint8_t *bytes, float *out, int nb)
{
    for (int i = 0; i < nb; i++) {
        const uint8_t *b = bytes + (size_t)i * 210;
        const uint8_t *ql = b + 0;
        const uint8_t *qh = b + 128;
        const int8_t  *sc = (const int8_t *)(b + 192);
        float d = f16_to_f32(rd16(b + 208));
        float *y = out + (size_t)i * 256;
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l + 0]  = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y += 128; ql += 64; qh += 32; sc += 8;
        }
    }
}

static void ref_dequant_iq4xs(const uint8_t *bytes, float *out, int nb)
{
    for (int i = 0; i < nb; i++) {
        const uint8_t *b = bytes + (size_t)i * 136;
        const uint8_t *qs = b + 8;
        float d = f16_to_f32(rd16(b));
        uint16_t scales_h = rd16(b + 2);
        const uint8_t *scales_l = b + 4;
        for (int ib = 0; ib < 8; ++ib) {
            int ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) |
                     (((scales_h >> (2 * ib)) & 3) << 4);
            float dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                out[i * 256 + ib * 32 + j]      = dl * iq4nl[qs[ib * 16 + j] & 0xF];
                out[i * 256 + ib * 32 + j + 16] = dl * iq4nl[qs[ib * 16 + j] >> 4];
            }
        }
    }
}

/* Q4_1: d f16 @0, m f16 @2, qs[16] @4.  out = d*nib + m */
static void ref_dequant_q4_1(const uint8_t *bytes, float *out, int nb)
{
    for (int i = 0; i < nb; i++) {
        const uint8_t *b = bytes + (size_t)i * 20;
        float d = f16_to_f32(rd16(b));
        float m = f16_to_f32(rd16(b + 2));
        for (int j = 0; j < 16; ++j) {
            out[i * 32 + j]      = d * (b[4 + j] & 0xF) + m;
            out[i * 32 + j + 16] = d * (b[4 + j] >> 4) + m;
        }
    }
}

/* Q5_0: d f16 @0, qh[4] @2, qs[16] @6.  out = d*((nib|xh)-16) */
static void ref_dequant_q5_0(const uint8_t *bytes, float *out, int nb)
{
    for (int i = 0; i < nb; i++) {
        const uint8_t *b = bytes + (size_t)i * 22;
        float d = f16_to_f32(rd16(b));
        uint32_t qh = (uint32_t)(b[2] | (b[3] << 8) | (b[4] << 16) | ((uint32_t)b[5] << 24));
        for (int j = 0; j < 16; ++j) {
            uint8_t xh_0 = (uint8_t)(((qh >> (j + 0)) << 4) & 0x10);
            uint8_t xh_1 = (uint8_t)(((qh >> (j + 12))     ) & 0x10);
            int32_t x0 = ((b[6 + j] & 0x0F) | xh_0) - 16;
            int32_t x1 = ((b[6 + j] >>   4) | xh_1) - 16;
            out[i * 32 + j]      = x0 * d;
            out[i * 32 + j + 16] = x1 * d;
        }
    }
}

/* Q5_1: d f16 @0, m f16 @2, qh[4] @4, qs[16] @8.  out = d*(nib|xh)+m */
static void ref_dequant_q5_1(const uint8_t *bytes, float *out, int nb)
{
    for (int i = 0; i < nb; i++) {
        const uint8_t *b = bytes + (size_t)i * 24;
        float d = f16_to_f32(rd16(b));
        float m = f16_to_f32(rd16(b + 2));
        uint32_t qh = (uint32_t)(b[4] | (b[5] << 8) | (b[6] << 16) | ((uint32_t)b[7] << 24));
        for (int j = 0; j < 16; ++j) {
            uint8_t xh_0 = (uint8_t)(((qh >> (j + 0)) << 4) & 0x10);
            uint8_t xh_1 = (uint8_t)(((qh >> (j + 12))     ) & 0x10);
            int x0 = (b[8 + j] & 0x0F) | xh_0;
            int x1 = (b[8 + j] >>   4) | xh_1;
            out[i * 32 + j]      = x0 * d + m;
            out[i * 32 + j + 16] = x1 * d + m;
        }
    }
}

/* Q8_1: d f16 @0, s f16 @2, qs[32] @4.  out = d*qs */
static void ref_dequant_q8_1(const uint8_t *bytes, float *out, int nb)
{
    for (int i = 0; i < nb; i++) {
        const uint8_t *b = bytes + (size_t)i * 36;
        float d = f16_to_f32(rd16(b));
        for (int j = 0; j < 32; ++j) out[i * 32 + j] = d * (int8_t)b[4 + j];
    }
}

/* Q2_K: scales[16] @0, qs[64] @16, d @80, dmin @82 */
static void ref_dequant_q2k(const uint8_t *bytes, float *out, int nb)
{
    for (int i = 0; i < nb; i++) {
        const uint8_t *b = bytes + (size_t)i * 84;
        const uint8_t *scales = b;
        const uint8_t *q      = b + 16;
        float d   = f16_to_f32(rd16(b + 80));
        float min = f16_to_f32(rd16(b + 82));
        int is = 0;
        float *y = out + (size_t)i * 256;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t sc = scales[is++];
                float dl = d * (sc & 0xF); float ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * (int8_t)((q[l] >> shift) & 3) - ml;
                sc = scales[is++];
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * (int8_t)((q[l + 16] >> shift) & 3) - ml;
                shift += 2;
            }
            q += 32;
        }
    }
}

/* Q3_K: hmask[32] @0, qs[64] @32, scales[12] @96, d @108 */
static void ref_dequant_q3k(const uint8_t *bytes, float *out, int nb)
{
    for (int i = 0; i < nb; i++) {
        const uint8_t *b = bytes + (size_t)i * 110;
        const uint8_t *hm = b;
        const uint8_t *q  = b + 32;
        const uint8_t *scales = b + 96;
        float d = f16_to_f32(rd16(b + 108));
        int8_t s[16];
        for (int k = 0; k < 4; ++k) {
            s[k]      = (int8_t)((scales[k] & 0xF) | ((scales[8 + k] & 0x3) << 4));
            s[4 + k]  = (int8_t)((scales[4 + k] & 0xF) | (((scales[8 + k] >> 2) & 0x3) << 4));
            s[8 + k]  = (int8_t)((scales[k] >> 4) | (((scales[8 + k] >> 4) & 0x3) << 4));
            s[12 + k] = (int8_t)((scales[4 + k] >> 4) | (((scales[8 + k] >> 6) & 0x3) << 4));
        }
        int is = 0;
        uint8_t m = 1;
        float *y = out + (size_t)i * 256;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                float dl = d * (s[is++] - 32);
                for (int l = 0; l < 16; ++l)
                    *y++ = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
                dl = d * (s[is++] - 32);
                for (int l = 0; l < 16; ++l)
                    *y++ = dl * ((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
}

/* Q5_K: d @0, dmin @2, scales[12] @4, qh[32] @16, qs[128] @48 */
static void ref_dequant_q5k(const uint8_t *bytes, float *out, int nb)
{
    for (int i = 0; i < nb; i++) {
        const uint8_t *b = bytes + (size_t)i * 176;
        const uint8_t *scales = b + 4;
        const uint8_t *ql = b + 48;
        const uint8_t *qh = b + 16;
        float d   = f16_to_f32(rd16(b));
        float min = f16_to_f32(rd16(b + 2));
        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        float *y = out + (size_t)i * 256;
        for (int j = 0; j < 256; j += 64) {
            get_scale_min_k4(is + 0, scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32; is += 2;
            u1 <<= 2; u2 <<= 2;
        }
    }
}

/* IQ4_NL: d f16 @0, qs[16] @2, 32 elems/block */
static void ref_dequant_iq4_nl(const uint8_t *bytes, float *out, int nb)
{
    for (int i = 0; i < nb; i++) {
        const uint8_t *b = bytes + (size_t)i * 18;
        float d = f16_to_f32(rd16(b));
        for (int j = 0; j < 16; ++j) {
            out[i * 32 + j]      = d * iq4nl[b[2 + j] & 0xF];
            out[i * 32 + j + 16] = d * iq4nl[b[2 + j] >> 4];
        }
    }
}

/* IQ1_S: d f16 @0, qs[32] @2, qh[16 u16] @34 */
static void ref_dequant_iq1_s(const uint8_t *bytes, float *out, int nb)
{
    for (int i = 0; i < nb; i++) {
        const uint8_t *b = bytes + (size_t)i * 50;
        float d = f16_to_f32(rd16(b));
        const uint8_t *qs = b + 2;
        const uint16_t *qh = (const uint16_t *)(b + 34);
        float *y = out + (size_t)i * 256;
        for (int ib = 0; ib < 8; ++ib) {
            uint16_t h = qh[ib];
            float dl = d * (2 * ((h >> 12) & 7) + 1);
            float delta = (h & 0x8000) ? -0.125f : 0.125f;
            for (int il = 0; il < 4; ++il) {
                const int8_t *grid = (const int8_t *)&vkquant_iq1s_grid[qs[4 * ib + il] | (((h >> 3 * il) & 7) << 8)];
                for (int j = 0; j < 8; ++j) y[j] = dl * (grid[j] + delta);
                y += 8;
            }
        }
    }
}

/* IQ1_M: qs[32] @0, qh[16] @32, scales[8] @48 */
static void ref_dequant_iq1_m(const uint8_t *bytes, float *out, int nb)
{
    for (int i = 0; i < nb; i++) {
        const uint8_t *b = bytes + (size_t)i * 56;
        const uint16_t *sc = (const uint16_t *)(b + 48);
        uint16_t scale_u16 = (uint16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
        float d = f16_to_f32(scale_u16);
        const uint8_t *qs = b;
        const uint8_t *qh = b + 32;
        float *y = out + (size_t)i * 256;
        float delta[4];
        uint16_t idx[4];
        for (int ib = 0; ib < 8; ++ib) {
            float dl1 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1);
            float dl2 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1);
            idx[0] = (uint16_t)(qs[0] | ((qh[0] << 8) & 0x700));
            idx[1] = (uint16_t)(qs[1] | ((qh[0] << 4) & 0x700));
            idx[2] = (uint16_t)(qs[2] | ((qh[1] << 8) & 0x700));
            idx[3] = (uint16_t)(qs[3] | ((qh[1] << 4) & 0x700));
            delta[0] = qh[0] & 0x08 ? -0.125f : 0.125f;
            delta[1] = qh[0] & 0x80 ? -0.125f : 0.125f;
            delta[2] = qh[1] & 0x08 ? -0.125f : 0.125f;
            delta[3] = qh[1] & 0x80 ? -0.125f : 0.125f;
            for (int l = 0; l < 2; ++l) {
                const int8_t *grid = (const int8_t *)&vkquant_iq1s_grid[idx[l]];
                for (int j = 0; j < 8; ++j) y[j] = dl1 * (grid[j] + delta[l]);
                y += 8;
            }
            for (int l = 2; l < 4; ++l) {
                const int8_t *grid = (const int8_t *)&vkquant_iq1s_grid[idx[l]];
                for (int j = 0; j < 8; ++j) y[j] = dl2 * (grid[j] + delta[l]);
                y += 8;
            }
            qs += 4; qh += 2;
        }
    }
}

/* IQ2_XS: d f16 @0, qs[32 u16] @2, scales[8] @66 */
static void ref_dequant_iq2_xs(const uint8_t *bytes, float *out, int nb)
{
    for (int i = 0; i < nb; i++) {
        const uint8_t *b = bytes + (size_t)i * 74;
        float d = f16_to_f32(rd16(b));
        const uint16_t *qs = (const uint16_t *)(b + 2);
        const uint8_t *scales = b + 66;
        float *y = out + (size_t)i * 256;
        for (int ib = 0; ib < 8; ++ib) {
            float db[2];
            db[0] = d * (0.5f + (scales[ib] & 0xf)) * 0.25f;
            db[1] = d * (0.5f + (scales[ib] >> 4)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                uint16_t q2 = qs[4 * ib + l];
                const uint8_t *grid = (const uint8_t *)&vkquant_iq2xs_grid[q2 & 511];
                uint8_t signs = vkquant_ksigns_iq2xs[q2 >> 9];
                float dl = db[l / 2];
                for (int j = 0; j < 8; ++j)
                    y[j] = dl * grid[j] * (signs & vkquant_kmask_iq2xs[j] ? -1.f : 1.f);
                y += 8;
            }
        }
    }
}

/* IQ2_S: d f16 @0, qs[64] @2, qh[8] @66, scales[8] @74 */
static void ref_dequant_iq2_s(const uint8_t *bytes, float *out, int nb)
{
    for (int i = 0; i < nb; i++) {
        const uint8_t *b = bytes + (size_t)i * 82;
        float d = f16_to_f32(rd16(b));
        const uint8_t *qs = b + 2;
        const uint8_t *qh = b + 66;
        const uint8_t *scales = b + 74;
        const uint8_t *signs = qs + 32;
        float *y = out + (size_t)i * 256;
        for (int ib = 0; ib < 8; ++ib) {
            float db[2];
            db[0] = d * (0.5f + (scales[ib] & 0xf)) * 0.25f;
            db[1] = d * (0.5f + (scales[ib] >> 4)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                float dl = db[l / 2];
                const uint8_t *grid = (const uint8_t *)&vkquant_iq2s_grid[qs[4 * ib + l] | ((qh[ib] << (8 - 2 * l)) & 0x300)];
                for (int j = 0; j < 8; ++j)
                    y[j] = dl * grid[j] * (signs[4 * ib + l] & vkquant_kmask_iq2xs[j] ? -1.f : 1.f);
                y += 8;
            }
        }
    }
}

/* IQ2_XXS: d f16 @0, qs[32 u16] @2 */
static void ref_dequant_iq2_xxs(const uint8_t *bytes, float *out, int nb)
{
    for (int i = 0; i < nb; i++) {
        const uint8_t *b = bytes + (size_t)i * 66;
        float d = f16_to_f32(rd16(b));
        const uint8_t *qs = b + 2;
        float *y = out + (size_t)i * 256;
        for (int ib = 0; ib < 8; ++ib) {
            uint32_t aux32[2];
            memcpy(aux32, qs + 8 * ib, 8);
            const uint8_t *aux8 = (const uint8_t *)aux32;
            float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t *grid = (const uint8_t *)&vkquant_iq2xxs_grid[aux8[l]];
                uint8_t signs = vkquant_ksigns_iq2xs[(aux32[1] >> 7 * l) & 127];
                for (int j = 0; j < 8; ++j)
                    y[j] = db * grid[j] * (signs & vkquant_kmask_iq2xs[j] ? -1.f : 1.f);
                y += 8;
            }
        }
    }
}

/* IQ3_XXS: d f16 @0, qs[96] @2 */
static void ref_dequant_iq3_xxs(const uint8_t *bytes, float *out, int nb)
{
    for (int i = 0; i < nb; i++) {
        const uint8_t *b = bytes + (size_t)i * 98;
        float d = f16_to_f32(rd16(b));
        const uint8_t *qs = b + 2;
        const uint8_t *ss = qs + 64;
        float *y = out + (size_t)i * 256;
        for (int ib = 0; ib < 8; ++ib) {
            uint32_t aux32;
            memcpy(&aux32, ss + 4 * ib, 4);
            float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
            for (int l = 0; l < 4; ++l) {
                uint8_t signs = vkquant_ksigns_iq2xs[(aux32 >> 7 * l) & 127];
                const uint8_t *grid1 = (const uint8_t *)&vkquant_iq3xxs_grid[qs[8 * ib + 2 * l]];
                const uint8_t *grid2 = (const uint8_t *)&vkquant_iq3xxs_grid[qs[8 * ib + 2 * l + 1]];
                for (int j = 0; j < 4; ++j) {
                    y[j + 0] = db * grid1[j] * (signs & vkquant_kmask_iq2xs[j + 0] ? -1.f : 1.f);
                    y[j + 4] = db * grid2[j] * (signs & vkquant_kmask_iq2xs[j + 4] ? -1.f : 1.f);
                }
                y += 8;
            }
        }
    }
}

/* IQ3_S: d f16 @0, qs[64] @2, qh[8] @66, signs[32] @74, scales[4] @106 */
static void ref_dequant_iq3_s(const uint8_t *bytes, float *out, int nb)
{
    for (int i = 0; i < nb; i++) {
        const uint8_t *b = bytes + (size_t)i * 110;
        float d = f16_to_f32(rd16(b));
        const uint8_t *qs = b + 2;
        const uint8_t *qh = b + 66;
        const uint8_t *signs = b + 74;
        const uint8_t *scales = b + 106;
        float *y = out + (size_t)i * 256;
        for (int ib32 = 0; ib32 < 8; ib32 += 2) {
            const float db1 = d * (1 + 2 * (scales[ib32 / 2] & 0xf));
            const float db2 = d * (1 + 2 * (scales[ib32 / 2] >> 4));
            for (int l = 0; l < 4; ++l) {
                const uint8_t *grid1 = (const uint8_t *)&vkquant_iq3s_grid[qs[2 * l + 0] | ((qh[0] << (8 - 2 * l)) & 256)];
                const uint8_t *grid2 = (const uint8_t *)&vkquant_iq3s_grid[qs[2 * l + 1] | ((qh[0] << (7 - 2 * l)) & 256)];
                for (int j = 0; j < 4; ++j) {
                    y[j + 0] = db1 * grid1[j] * (signs[l] & vkquant_kmask_iq2xs[j + 0] ? -1.f : 1.f);
                    y[j + 4] = db1 * grid2[j] * (signs[l] & vkquant_kmask_iq2xs[j + 4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 8; signs += 4;
            for (int l = 0; l < 4; ++l) {
                const uint8_t *grid1 = (const uint8_t *)&vkquant_iq3s_grid[qs[2 * l + 0] | ((qh[1] << (8 - 2 * l)) & 256)];
                const uint8_t *grid2 = (const uint8_t *)&vkquant_iq3s_grid[qs[2 * l + 1] | ((qh[1] << (7 - 2 * l)) & 256)];
                for (int j = 0; j < 4; ++j) {
                    y[j + 0] = db2 * grid1[j] * (signs[l] & vkquant_kmask_iq2xs[j + 0] ? -1.f : 1.f);
                    y[j + 4] = db2 * grid2[j] * (signs[l] & vkquant_kmask_iq2xs[j + 4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qh += 2; qs += 8; signs += 4;
        }
    }
}

/* TQ1_0: qs[48] @0, qh[4] @48, d f16 @52 */
static void ref_dequant_tq1_0(const uint8_t *bytes, float *out, int nb)
{
    const uint8_t pow3[6] = { 1, 3, 9, 27, 81, 243 };
    for (int i = 0; i < nb; i++) {
        const uint8_t *b = bytes + (size_t)i * 54;
        float d = f16_to_f32(rd16(b + 52));
        float *y = out + (size_t)i * 256;
        for (size_t j = 0; j < 48 - 48 % 32; j += 32)
            for (size_t n = 0; n < 5; ++n)
                for (size_t m = 0; m < 32; ++m) {
                    uint8_t q = (uint8_t)(b[j + m] * pow3[n]);
                    int16_t xi = (int16_t)(((uint16_t)q * 3) >> 8);
                    *y++ = (float)(xi - 1) * d;
                }
        for (size_t j = 48 - 48 % 32; j < 48; j += 16)
            for (size_t n = 0; n < 5; ++n)
                for (size_t m = 0; m < 16; ++m) {
                    uint8_t q = (uint8_t)(b[j + m] * pow3[n]);
                    int16_t xi = (int16_t)(((uint16_t)q * 3) >> 8);
                    *y++ = (float)(xi - 1) * d;
                }
        for (size_t n = 0; n < 4; ++n)
            for (size_t j = 0; j < 4; ++j) {
                uint8_t q = (uint8_t)(b[48 + j] * pow3[n]);
                int16_t xi = (int16_t)(((uint16_t)q * 3) >> 8);
                *y++ = (float)(xi - 1) * d;
            }
    }
}

/* TQ2_0: qs[64] @0, d f16 @64 */
static void ref_dequant_tq2_0(const uint8_t *bytes, float *out, int nb)
{
    for (int i = 0; i < nb; i++) {
        const uint8_t *b = bytes + (size_t)i * 66;
        float d = f16_to_f32(rd16(b + 64));
        float *y = out + (size_t)i * 256;
        for (size_t j = 0; j < 64; j += 32)
            for (size_t l = 0; l < 4; ++l)
                for (size_t m = 0; m < 32; ++m) {
                    int8_t q = (int8_t)((b[j + m] >> (l * 2)) & 3);
                    *y++ = (float)(q - 1) * d;
                }
    }
}

/* ===========================================================================
 * Block fillers (deterministic dequant test inputs)
 * ========================================================================== */

static void fill_f16_fields(uint8_t *dst, size_t block_bytes,
                            const uint16_t *f16offs, const float *f16vals, int nf16)
{
    for (size_t i = 0; i < block_bytes; ++i) dst[i] = (uint8_t)rng_next();
    for (int f = 0; f < nf16; ++f) write_f16(dst + f16offs[f], f16vals[f]);
}

#define FILL_F16_1(dst, sz, o0, v0) \
    do { uint16_t o[1] = { (uint16_t)(o0) }; float v[1] = { (v0) }; \
         fill_f16_fields((dst), (sz), o, v, 1); } while (0)
#define FILL_F16_2(dst, sz, o0, v0, o1, v1) \
    do { uint16_t o[2] = { (uint16_t)(o0), (uint16_t)(o1) }; float v[2] = { (v0), (v1) }; \
         fill_f16_fields((dst), (sz), o, v, 2); } while (0)

static void fill_q4_1(uint8_t *dst, size_t sz)   { FILL_F16_2(dst, sz, 0, 0.5f, 2, -0.25f); }
static void fill_q5_0(uint8_t *dst, size_t sz)   { FILL_F16_1(dst, sz, 0, 0.5f); }
static void fill_q5_1(uint8_t *dst, size_t sz)   { FILL_F16_2(dst, sz, 0, 0.5f, 2, -0.25f); }
static void fill_q8_1(uint8_t *dst, size_t sz)   { FILL_F16_2(dst, sz, 0, 0.5f, 2, 2.0f); }
static void fill_q2k(uint8_t *dst, size_t sz)    { FILL_F16_2(dst, sz, 80, 0.25f, 82, -0.1f); }
static void fill_q3k(uint8_t *dst, size_t sz)    { FILL_F16_1(dst, sz, 108, 0.05f); }
static void fill_q5k(uint8_t *dst, size_t sz)    { FILL_F16_2(dst, sz, 0, 0.25f, 2, -0.1f); }
static void fill_iq4_nl(uint8_t *dst, size_t sz) { FILL_F16_1(dst, sz, 0, 1.0f); }
static void fill_iq1_s(uint8_t *dst, size_t sz)  { FILL_F16_1(dst, sz, 0, 0.05f); }
static void fill_iq2_xs(uint8_t *dst, size_t sz) { FILL_F16_1(dst, sz, 0, 0.05f); }
static void fill_iq2_s(uint8_t *dst, size_t sz)  { FILL_F16_1(dst, sz, 0, 0.05f); }
static void fill_iq2_xxs(uint8_t *dst, size_t sz){ FILL_F16_1(dst, sz, 0, 0.05f); }
static void fill_iq3_s(uint8_t *dst, size_t sz)  { FILL_F16_1(dst, sz, 0, 0.05f); }
static void fill_iq3_xxs(uint8_t *dst, size_t sz){ FILL_F16_1(dst, sz, 0, 0.05f); }
static void fill_tq1_0(uint8_t *dst, size_t sz)  { FILL_F16_1(dst, sz, 52, 0.25f); }
static void fill_tq2_0(uint8_t *dst, size_t sz)  { FILL_F16_1(dst, sz, 64, 0.25f); }

/* IQ1_M: no f16 header; force the 4 assembled scale words to a finite d. */
static void fill_iq1_m(uint8_t *dst, size_t sz)
{
    for (size_t i = 0; i < sz; ++i) dst[i] = (uint8_t)rng_next();
    /* sc words (u16 LE @48..55) chosen so the assembled scale.u16 = 0x3800 (0.5) */
    const uint8_t fixed[8] = { 0x00, 0x30, 0x00, 0x80, 0x00, 0x30, 0x00, 0x80 };
    memcpy(dst + 48, fixed, 8);
}

/* ===========================================================================
 * Deterministic f32 source for the quant round-trip tests.
 * ========================================================================== */

static float gen_quant_src(uint32_t i)
{
    return 0.50f * sinf((float)(i * 7u) * 0.17f) + 0.03f * (float)(i % 7);
}

/* Gentler, low-frequency source for the K-quant round-trips: within any
 * 16-element group the range stays small enough that even the 2-bit Q2_K
 * levels keep the round-trip error under the 1e-1 bound. */
static float gen_quant_src_k(uint32_t i)
{
    return 0.20f * sinf((float)i * 0.05f) + 0.08f * sinf((float)i * 0.013f);
}

/* ===========================================================================
 * Bootstrap helpers
 * ========================================================================== */

static VkResult create_instance(const char *app_name, VkInstance *out_instance)
{
    VkApplicationInfo app_info;
    memset(&app_info, 0, sizeof(app_info));
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = app_name;
    app_info.applicationVersion = 1;
    app_info.pEngineName = "vait";
    app_info.engineVersion = 1;
    app_info.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo create_info;
    memset(&create_info, 0, sizeof(create_info));
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;
    create_info.enabledLayerCount = 0;
    create_info.ppEnabledLayerNames = NULL;
    create_info.enabledExtensionCount = 0;
    create_info.ppEnabledExtensionNames = NULL;
    return vkCreateInstance(&create_info, NULL, out_instance);
}

static VkResult find_physical_device(VkInstance instance,
                                     VkPhysicalDevice *out_physical_device)
{
    uint32_t count = 0;
    VkResult r = vkEnumeratePhysicalDevices(instance, &count, NULL);
    if (r != VK_SUCCESS || count == 0) return VK_ERROR_INITIALIZATION_FAILED;

    VkPhysicalDevice *devices =
        (VkPhysicalDevice *)malloc(count * sizeof(VkPhysicalDevice));
    if (!devices) return VK_ERROR_OUT_OF_HOST_MEMORY;

    r = vkEnumeratePhysicalDevices(instance, &count, devices);
    if (r == VK_SUCCESS) *out_physical_device = devices[0];
    free(devices);
    return r;
}

static VkBool32 query_shader_int64(VkPhysicalDevice physical_device)
{
    VkPhysicalDeviceFeatures2 features2;
    memset(&features2, 0, sizeof(features2));
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    vkGetPhysicalDeviceFeatures2(physical_device, &features2);
    return features2.features.shaderInt64;
}

static VkBool32 query_shader_int8(VkPhysicalDevice physical_device)
{
    VkPhysicalDeviceVulkan12Features vulkan12;
    memset(&vulkan12, 0, sizeof(vulkan12));
    vulkan12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;

    VkPhysicalDeviceFeatures2 features2;
    memset(&features2, 0, sizeof(features2));
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &vulkan12;
    vkGetPhysicalDeviceFeatures2(physical_device, &features2);
    return vulkan12.shaderInt8;
}

static VkBool32 queue_family_supports_compute(VkPhysicalDevice physical_device,
                                              uint32_t family)
{
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, NULL);
    if (count == 0) return VK_FALSE;

    VkQueueFamilyProperties *props =
        (VkQueueFamilyProperties *)malloc(count * sizeof(*props));
    if (!props) return VK_FALSE;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, props);

    VkBool32 supported = VK_FALSE;
    if (family < count) {
        supported =
            (props[family].queueFlags & VK_QUEUE_COMPUTE_BIT) ? VK_TRUE : VK_FALSE;
    }
    free(props);
    return supported;
}

static VkResult create_device(VkPhysicalDevice physical_device,
                              VkDevice *out_device)
{
    float priority = 1.0f;

    VkDeviceQueueCreateInfo queue_info;
    memset(&queue_info, 0, sizeof(queue_info));
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = 0;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;

    VkPhysicalDeviceFeatures features;
    memset(&features, 0, sizeof(features));
    features.shaderInt64 = VK_TRUE;

    VkPhysicalDeviceVulkan12Features vulkan12;
    memset(&vulkan12, 0, sizeof(vulkan12));
    vulkan12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vulkan12.shaderInt8 = VK_TRUE;

    VkDeviceCreateInfo create_info;
    memset(&create_info, 0, sizeof(create_info));
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.pNext = &vulkan12;
    create_info.queueCreateInfoCount = 1;
    create_info.pQueueCreateInfos = &queue_info;
    create_info.enabledExtensionCount = 0;
    create_info.ppEnabledExtensionNames = NULL;
    create_info.pEnabledFeatures = &features;
    return vkCreateDevice(physical_device, &create_info, NULL, out_device);
}

static VkResult create_command_pool_and_buffer(VkDevice device,
                                               VkCommandPool *out_pool,
                                               VkCommandBuffer *out_cmd)
{
    VkCommandPoolCreateInfo pool_info;
    memset(&pool_info, 0, sizeof(pool_info));
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = 0;

    VkResult r = vkCreateCommandPool(device, &pool_info, NULL, out_pool);
    if (r != VK_SUCCESS) return r;

    VkCommandBufferAllocateInfo alloc_info;
    memset(&alloc_info, 0, sizeof(alloc_info));
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = *out_pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;
    r = vkAllocateCommandBuffers(device, &alloc_info, out_cmd);
    if (r != VK_SUCCESS) {
        vkDestroyCommandPool(device, *out_pool, NULL);
        *out_pool = VK_NULL_HANDLE;
    }
    return r;
}

static VkResult create_sub_buffer(VkDevice device, VkDeviceMemory mem,
                                  VkDeviceSize offset, VkDeviceSize size,
                                  VkBuffer *out_buffer)
{
    VkBufferCreateInfo buffer_info;
    memset(&buffer_info, 0, sizeof(buffer_info));
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                      | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                      | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult r = vkCreateBuffer(device, &buffer_info, NULL, out_buffer);
    if (r != VK_SUCCESS) return r;
    return vkBindBufferMemory(device, *out_buffer, mem, offset);
}

static VkResult allocate_staging_memory(VkPhysicalDevice physical_device,
                                        VkDevice device, VkDeviceSize size,
                                        VkDeviceMemory *out_memory,
                                        VkBuffer *out_staging,
                                        VkDeviceSize *out_align)
{
    VkBufferCreateInfo probe_info;
    memset(&probe_info, 0, sizeof(probe_info));
    probe_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    probe_info.size = size;
    probe_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                     | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                     | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    probe_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer probe = VK_NULL_HANDLE;
    VkResult r = vkCreateBuffer(device, &probe_info, NULL, &probe);
    if (r != VK_SUCCESS) return r;

    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(device, probe, &requirements);
    vkDestroyBuffer(device, probe, NULL);

    VkPhysicalDeviceMemoryProperties memory_properties;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);

    uint32_t memory_index = UINT32_MAX;
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++) {
        if ((requirements.memoryTypeBits & (1u << i)) == 0u) continue;
        VkMemoryPropertyFlags flags = memory_properties.memoryTypes[i].propertyFlags;
        if ((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0u &&
            (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0u) {
            memory_index = i;
            break;
        }
    }
    if (memory_index == UINT32_MAX) return VK_ERROR_FEATURE_NOT_PRESENT;

    VkMemoryAllocateInfo alloc_info;
    memset(&alloc_info, 0, sizeof(alloc_info));
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = size;
    alloc_info.memoryTypeIndex = memory_index;

    r = vkAllocateMemory(device, &alloc_info, NULL, out_memory);
    if (r != VK_SUCCESS) return r;

    r = create_sub_buffer(device, *out_memory, 0, size, out_staging);
    if (r != VK_SUCCESS) {
        vkFreeMemory(device, *out_memory, NULL);
        *out_memory = VK_NULL_HANDLE;
        return r;
    }

    *out_align = requirements.alignment;
    return VK_SUCCESS;
}

static VkDeviceSize align_up(VkDeviceSize value, VkDeviceSize align)
{
    return (value + align - 1) & ~(align - 1);
}

static VkDeviceSize take_region(VkDeviceSize *cursor, VkDeviceSize align,
                                VkDeviceSize size)
{
    VkDeviceSize offset = align_up(*cursor, align);
    *cursor = offset + size;
    return offset;
}

static VkResult setup_quant_op(VkDevice device, VkDeviceMemory mem,
                               VkDeviceSize *cursor, VkDeviceSize align,
                               VkDeviceSize in_bytes, uint32_t out_count,
                               op_t *op)
{
    VkDeviceSize out_bytes = (VkDeviceSize)out_count * sizeof(float);

    op->off_in        = take_region(cursor, align, in_bytes);
    op->off_out       = take_region(cursor, align, out_bytes);
    op->off_readback  = take_region(cursor, align, out_bytes);
    op->off_expected  = take_region(cursor, align, out_bytes);

    VkResult r = create_sub_buffer(device, mem, op->off_in, in_bytes, &op->in);
    if (r != VK_SUCCESS) return r;
    return create_sub_buffer(device, mem, op->off_out, out_bytes, &op->out);
}

static VkResult setup_roundtrip_op(VkDevice device, VkDeviceMemory mem,
                                   VkDeviceSize *cursor, VkDeviceSize align,
                                   uint32_t num_blocks, uint32_t elems_per_block,
                                   uint32_t block_bytes, rtop_t *op)
{
    VkDeviceSize in_bytes = (VkDeviceSize)num_blocks * elems_per_block * sizeof(float);
    VkDeviceSize q_bytes  = (VkDeviceSize)num_blocks * block_bytes;
    VkDeviceSize out_bytes = (VkDeviceSize)num_blocks * elems_per_block * sizeof(float);

    op->off_in       = take_region(cursor, align, in_bytes);
    op->off_q        = take_region(cursor, align, q_bytes);
    op->off_out      = take_region(cursor, align, out_bytes);
    op->off_readback = take_region(cursor, align, out_bytes);

    VkResult r = create_sub_buffer(device, mem, op->off_in, in_bytes, &op->in);
    if (r != VK_SUCCESS) return r;
    r = create_sub_buffer(device, mem, op->off_q, q_bytes, &op->qbytes);
    if (r != VK_SUCCESS) return r;
    return create_sub_buffer(device, mem, op->off_out, out_bytes, &op->out);
}

/* ===========================================================================
 * Command recording helpers
 * ========================================================================== */

static void record_compute_to_transfer_barrier(VkCommandBuffer cmd)
{
    VkMemoryBarrier barrier;
    memset(&barrier, 0, sizeof(barrier));
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 1, &barrier, 0, NULL, 0, NULL);
}

static void record_compute_to_compute_barrier(VkCommandBuffer cmd)
{
    VkMemoryBarrier barrier;
    memset(&barrier, 0, sizeof(barrier));
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &barrier, 0, NULL, 0, NULL);
}

static void record_copy_readback(VkCommandBuffer cmd, VkBuffer src,
                                 VkBuffer dst, VkDeviceSize dst_offset,
                                 VkDeviceSize size)
{
    VkBufferCopy region;
    memset(&region, 0, sizeof(region));
    region.srcOffset = 0;
    region.dstOffset = dst_offset;
    region.size = size;
    vkCmdCopyBuffer(cmd, src, dst, 1, &region);
}

static int record_dispatch(VkResult result, const char *name)
{
    if (result == VK_SUCCESS) {
        printf("  %-20s : recorded\n", name);
        return 1;
    }
    printf("  %-20s : FAIL (record, VkResult=%d)\n", name, (int)result);
    return 0;
}

static int check_output(const char *name, const void *mapped,
                        VkDeviceSize off_readback, const float *expected,
                        uint32_t count, float tolerance)
{
    const float *got = (const float *)((const char *)mapped + off_readback);
    int pass = 1;
    uint32_t mismatches = 0;
    float max_diff = 0.0f;

    for (uint32_t i = 0; i < count; i++) {
        float diff = fabsf(got[i] - expected[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > tolerance) {
            if (mismatches < 6) {
                printf("    mismatch[%u]: got %.6f expected %.6f (diff %.3e)\n",
                       i, got[i], expected[i], diff);
            }
            mismatches++;
            pass = 0;
        }
    }
    printf("  %-20s : %s (max diff %.3e)\n", name, pass ? "PASS" : "FAIL", max_diff);
    return pass;
}

static int check_roundtrip(const char *name, const void *mapped,
                           VkDeviceSize off_readback, uint32_t count,
                           float tolerance, float (*src_fn)(uint32_t))
{
    const float *got = (const float *)((const char *)mapped + off_readback);
    int pass = 1;
    float max_err = 0.0f;
    uint32_t mismatches = 0;

    for (uint32_t i = 0; i < count; i++) {
        float expected = src_fn(i);
        float diff = fabsf(got[i] - expected);
        if (diff > max_err) max_err = diff;
        if (diff > tolerance) {
            if (mismatches < 6) {
                printf("    mismatch[%u]: got %.6f expected %.6f (diff %.3e)\n",
                       i, got[i], expected, diff);
            }
            mismatches++;
            pass = 0;
        }
    }
    printf("  %-20s : %s (max abs err %.5f)\n", name, pass ? "PASS" : "FAIL", max_err);
    return pass;
}

/* ===========================================================================
 * New-format case tables
 * ========================================================================== */

typedef struct {
    const char *name;
    uint32_t block_bytes;
    uint32_t elems_per_block;
    uint32_t num_blocks;
    VkResult (*dequant_fn)(VkQuantContext *, VkCommandBuffer, uint32_t, VkBuffer, VkBuffer);
    void (*ref_fn)(const uint8_t *, float *, int);
    void (*fill_fn)(uint8_t *, size_t);
    op_t op;
} dqcase_t;

typedef struct {
    const char *name;
    uint32_t block_bytes;
    uint32_t elems_per_block;
    uint32_t num_blocks;
    float (*src_fn)(uint32_t);
    VkResult (*quant_fn)(VkQuantContext *, VkCommandBuffer, uint32_t, VkBuffer, VkBuffer);
    VkResult (*dequant_fn)(VkQuantContext *, VkCommandBuffer, uint32_t, VkBuffer, VkBuffer);
    rtop_t op;
} qcase_t;

static dqcase_t s_dq_cases[] = {
    { "dequant_q4_1_f32",   20u,  32u,  TEST_NUM_BLOCKS, vkquant_dequant_q4_1_f32,    ref_dequant_q4_1,    fill_q4_1,    {0} },
    { "dequant_q5_0_f32",   22u,  32u,  TEST_NUM_BLOCKS, vkquant_dequant_q5_0_f32,    ref_dequant_q5_0,    fill_q5_0,    {0} },
    { "dequant_q5_1_f32",   24u,  32u,  TEST_NUM_BLOCKS, vkquant_dequant_q5_1_f32,    ref_dequant_q5_1,    fill_q5_1,    {0} },
    { "dequant_q8_1_f32",   36u,  32u,  TEST_NUM_BLOCKS, vkquant_dequant_q8_1_f32,    ref_dequant_q8_1,    fill_q8_1,    {0} },
    { "dequant_iq4_nl_f32", 18u,  32u,  TEST_NUM_BLOCKS, vkquant_dequant_iq4_nl_f32,  ref_dequant_iq4_nl,  fill_iq4_nl,  {0} },
    { "dequant_q2k_f32",    84u,  256u, TEST_K_BLOCKS,   vkquant_dequant_q2k_f32,     ref_dequant_q2k,     fill_q2k,     {0} },
    { "dequant_q3k_f32",    110u, 256u, TEST_K_BLOCKS,   vkquant_dequant_q3k_f32,     ref_dequant_q3k,     fill_q3k,     {0} },
    { "dequant_q5k_f32",    176u, 256u, TEST_K_BLOCKS,   vkquant_dequant_q5k_f32,     ref_dequant_q5k,     fill_q5k,     {0} },
    { "dequant_iq1_s_f32",  50u,  256u, TEST_K_BLOCKS,   vkquant_dequant_iq1_s_f32,   ref_dequant_iq1_s,   fill_iq1_s,   {0} },
    { "dequant_iq1_m_f32",  56u,  256u, TEST_K_BLOCKS,   vkquant_dequant_iq1_m_f32,   ref_dequant_iq1_m,   fill_iq1_m,   {0} },
    { "dequant_iq2_xs_f32", 74u,  256u, TEST_K_BLOCKS,   vkquant_dequant_iq2_xs_f32,  ref_dequant_iq2_xs,  fill_iq2_xs,  {0} },
    { "dequant_iq2_s_f32",  82u,  256u, TEST_K_BLOCKS,   vkquant_dequant_iq2_s_f32,   ref_dequant_iq2_s,   fill_iq2_s,   {0} },
    { "dequant_iq2_xxs_f32",66u,  256u, TEST_K_BLOCKS,   vkquant_dequant_iq2_xxs_f32, ref_dequant_iq2_xxs, fill_iq2_xxs, {0} },
    { "dequant_iq3_s_f32",  110u, 256u, TEST_K_BLOCKS,   vkquant_dequant_iq3_s_f32,   ref_dequant_iq3_s,   fill_iq3_s,   {0} },
    { "dequant_iq3_xxs_f32",98u,  256u, TEST_K_BLOCKS,   vkquant_dequant_iq3_xxs_f32, ref_dequant_iq3_xxs, fill_iq3_xxs, {0} },
    { "dequant_tq1_0_f32",  54u,  256u, TEST_K_BLOCKS,   vkquant_dequant_tq1_0_f32,   ref_dequant_tq1_0,   fill_tq1_0,   {0} },
    { "dequant_tq2_0_f32",  66u,  256u, TEST_K_BLOCKS,   vkquant_dequant_tq2_0_f32,   ref_dequant_tq2_0,   fill_tq2_0,   {0} },
};
#define DQ_CASE_COUNT (sizeof(s_dq_cases) / sizeof(s_dq_cases[0]))

static qcase_t s_q_cases[] = {
    { "quant+q4_1_rt", 20u,  32u,  TEST_QUANT_BLOCKS,   gen_quant_src,   vkquant_quantize_q4_1_f32, vkquant_dequant_q4_1_f32, {0} },
    { "quant+q5_0_rt", 22u,  32u,  TEST_QUANT_BLOCKS,   gen_quant_src,   vkquant_quantize_q5_0_f32, vkquant_dequant_q5_0_f32, {0} },
    { "quant+q5_1_rt", 24u,  32u,  TEST_QUANT_BLOCKS,   gen_quant_src,   vkquant_quantize_q5_1_f32, vkquant_dequant_q5_1_f32, {0} },
    { "quant+q8_1_rt", 36u,  32u,  TEST_QUANT_BLOCKS,   gen_quant_src,   vkquant_quantize_q8_1_f32, vkquant_dequant_q8_1_f32, {0} },
    { "quant+q2k_rt",  84u,  256u, TEST_KQUANT_BLOCKS,  gen_quant_src_k, vkquant_quantize_q2k_f32,  vkquant_dequant_q2k_f32,  {0} },
    { "quant+q3k_rt",  110u, 256u, TEST_KQUANT_BLOCKS,  gen_quant_src_k, vkquant_quantize_q3k_f32,  vkquant_dequant_q3k_f32,  {0} },
    { "quant+q4k_rt",  144u, 256u, TEST_KQUANT_BLOCKS,  gen_quant_src_k, vkquant_quantize_q4k_f32,  vkquant_dequant_q4k_f32,  {0} },
    { "quant+q5k_rt",  176u, 256u, TEST_KQUANT_BLOCKS,  gen_quant_src_k, vkquant_quantize_q5k_f32,  vkquant_dequant_q5k_f32,  {0} },
    { "quant+q6k_rt",  210u, 256u, TEST_KQUANT_BLOCKS,  gen_quant_src_k, vkquant_quantize_q6k_f32,  vkquant_dequant_q6k_f32,  {0} },
};
#define Q_CASE_COUNT (sizeof(s_q_cases) / sizeof(s_q_cases[0]))

/* ===========================================================================
 * main
 * ========================================================================== */

int main(void)
{
    harness_t h;
    memset(&h, 0, sizeof(h));
    h.mem = VK_NULL_HANDLE;
    h.staging = VK_NULL_HANDLE;
    h.fence = VK_NULL_HANDLE;

    op_t op_q8, op_q4, op_q4k, op_q6k, op_iq4xs;
    rtop_t op_q8rt, op_q4rt;
    memset(&op_q8, 0, sizeof(op_q8));
    memset(&op_q4, 0, sizeof(op_q4));
    memset(&op_q4k, 0, sizeof(op_q4k));
    memset(&op_q6k, 0, sizeof(op_q6k));
    memset(&op_iq4xs, 0, sizeof(op_iq4xs));
    memset(&op_q8rt, 0, sizeof(op_q8rt));
    memset(&op_q4rt, 0, sizeof(op_q4rt));

    const uint32_t total = TEST_NUM_BLOCKS * TEST_ELEMS_PER_BLOCK;

    int overall_pass = 1;
    VkResult r;

    /* ── 1. Instance ────────────────────────────────────────────────────── */
    r = create_instance("test_vkquant", &h.instance);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkquant: vkCreateInstance failed (%d)\n", (int)r);
        return 1;
    }

    /* ── 2. Physical device ─────────────────────────────────────────────── */
    r = find_physical_device(h.instance, &h.physical_device);
    if (r != VK_SUCCESS) {
        printf("test_vkquant: SKIP (no physical device found)\n");
        vkDestroyInstance(h.instance, NULL);
        return 0;
    }

    /* ── 3. shaderInt8 gate ─────────────────────────────────────────────── */
    if (query_shader_int8(h.physical_device) == VK_FALSE) {
        printf("test_vkquant: SKIP (shaderInt8 not supported)\n");
        vkDestroyInstance(h.instance, NULL);
        return 0;
    }
    (void)query_shader_int64(h.physical_device);

    /* ── 4. Queue family gate ───────────────────────────────────────────── */
    if (queue_family_supports_compute(h.physical_device, 0) == VK_FALSE) {
        printf("test_vkquant: SKIP (queue family 0 lacks compute)\n");
        vkDestroyInstance(h.instance, NULL);
        return 0;
    }

    /* ── 5. Logical device ──────────────────────────────────────────────── */
    r = create_device(h.physical_device, &h.device);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkquant: vkCreateDevice failed (%d)\n", (int)r);
        goto cleanup;
    }
    vkGetDeviceQueue(h.device, 0, 0, &h.queue);

    /* ── 6. Command pool + one command buffer ───────────────────────────── */
    r = create_command_pool_and_buffer(h.device, &h.cmd_pool, &h.cmd);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkquant: command pool/buffer failed (%d)\n", (int)r);
        goto cleanup;
    }

    /* ── 7. Staging memory ──────────────────────────────────────────────── */
    r = allocate_staging_memory(h.physical_device, h.device, TEST_STAGING_SIZE,
                                &h.mem, &h.staging, &h.align);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkquant: staging allocation failed (%d)\n", (int)r);
        goto cleanup;
    }
    r = vkMapMemory(h.device, h.mem, 0, VK_WHOLE_SIZE, 0, &h.mapped);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkquant: vkMapMemory failed (%d)\n", (int)r);
        goto cleanup;
    }

    /* ── 8. Context ─────────────────────────────────────────────────────── */
    r = vkquant_create_context(h.physical_device, h.device, &h.quant_ctx);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkquant: vkquant_create_context failed (%d)\n", (int)r);
        goto cleanup;
    }
    printf("test_vkquant: device ready (arch=%s, tier=%u, staging=%u)\n",
           vkquant_get_arch_name(h.quant_ctx), vkquant_get_arch_index(h.quant_ctx),
           (unsigned)TEST_STAGING_SIZE);

    /* ── 9. Region layout for each op ───────────────────────────────────── */
    r = setup_quant_op(h.device, h.mem, &h.cursor, h.align,
                       (VkDeviceSize)TEST_NUM_BLOCKS * TEST_Q8_BLOCK_SIZE,
                       total, &op_q8);
    if (r != VK_SUCCESS) goto cleanup;
    r = setup_quant_op(h.device, h.mem, &h.cursor, h.align,
                       (VkDeviceSize)TEST_NUM_BLOCKS * TEST_Q4_BLOCK_SIZE,
                       total, &op_q4);
    if (r != VK_SUCCESS) goto cleanup;

    r = setup_quant_op(h.device, h.mem, &h.cursor, h.align,
                       (VkDeviceSize)TEST_K_BLOCKS * TEST_Q4K_BLOCK_SIZE,
                       TEST_K_ELEMS, &op_q4k);
    if (r != VK_SUCCESS) goto cleanup;
    r = setup_quant_op(h.device, h.mem, &h.cursor, h.align,
                       (VkDeviceSize)TEST_K_BLOCKS * TEST_Q6K_BLOCK_SIZE,
                       TEST_K_ELEMS, &op_q6k);
    if (r != VK_SUCCESS) goto cleanup;
    r = setup_quant_op(h.device, h.mem, &h.cursor, h.align,
                       (VkDeviceSize)TEST_K_BLOCKS * TEST_IQ4XS_BLOCK_SIZE,
                       TEST_K_ELEMS, &op_iq4xs);
    if (r != VK_SUCCESS) goto cleanup;

    r = setup_roundtrip_op(h.device, h.mem, &h.cursor, h.align,
                           TEST_QUANT_BLOCKS, 32u, TEST_Q8_BLOCK_SIZE, &op_q8rt);
    if (r != VK_SUCCESS) goto cleanup;
    r = setup_roundtrip_op(h.device, h.mem, &h.cursor, h.align,
                           TEST_QUANT_BLOCKS, 32u, TEST_Q4_BLOCK_SIZE, &op_q4rt);
    if (r != VK_SUCCESS) goto cleanup;

    for (uint32_t i = 0; i < DQ_CASE_COUNT; i++) {
        r = setup_quant_op(h.device, h.mem, &h.cursor, h.align,
                           (VkDeviceSize)s_dq_cases[i].num_blocks * s_dq_cases[i].block_bytes,
                           s_dq_cases[i].num_blocks * s_dq_cases[i].elems_per_block,
                           &s_dq_cases[i].op);
        if (r != VK_SUCCESS) goto cleanup;
    }
    for (uint32_t i = 0; i < Q_CASE_COUNT; i++) {
        r = setup_roundtrip_op(h.device, h.mem, &h.cursor, h.align,
                               s_q_cases[i].num_blocks, s_q_cases[i].elems_per_block,
                               s_q_cases[i].block_bytes, &s_q_cases[i].op);
        if (r != VK_SUCCESS) goto cleanup;
    }

    /* ── 10. Fill raw quantized bytes with known values ─────────────────── */
    unsigned char *q8_bytes = (unsigned char *)h.mapped + op_q8.off_in;
    unsigned char *q4_bytes = (unsigned char *)h.mapped + op_q4.off_in;

    for (uint32_t b = 0; b < TEST_NUM_BLOCKS; b++) {
        uint32_t scale_off = b * TEST_Q8_BLOCK_SIZE;
        float scale = TEST_Q8_SCALE;
        memcpy(q8_bytes + scale_off, &scale, 4);
        for (uint32_t i = 0; i < TEST_ELEMS_PER_BLOCK; i++) {
            int8_t q = (int8_t)((i % 7) - 3);
            q8_bytes[scale_off + 4 + i] = (unsigned char)q;
        }
    }

    for (uint32_t b = 0; b < TEST_NUM_BLOCKS; b++) {
        uint32_t scale_off = b * TEST_Q4_BLOCK_SIZE;
        float scale = TEST_Q4_SCALE;
        memcpy(q4_bytes + scale_off, &scale, 4);
        for (uint32_t j = 0; j < 16; j++) {
            uint32_t e0 = 2u * j;
            uint32_t e1 = 2u * j + 1u;
            unsigned char nib0 = (unsigned char)(e0 % 16);
            unsigned char nib1 = (unsigned char)(e1 % 16);
            q4_bytes[scale_off + 4 + j] = (unsigned char)((nib1 << 4) | nib0);
        }
    }

    unsigned char *q4k_bytes = (unsigned char *)h.mapped + op_q4k.off_in;
    unsigned char *q6k_bytes = (unsigned char *)h.mapped + op_q6k.off_in;
    unsigned char *iq4xs_bytes = (unsigned char *)h.mapped + op_iq4xs.off_in;

    {
        const float q4k_d[2] = { 0.50f, 0.75f };
        const float q4k_dm[2] = { 0.25f, -0.125f };
        for (int b = 0; b < (int)TEST_K_BLOCKS; b++) {
            uint8_t sc[8], mn[8], nib[256];
            for (int j = 0; j < 8; ++j) {
                sc[j] = (uint8_t)((j * 7 + b * 3) & 0x3F);
                mn[j] = (uint8_t)((j * 5 + b * 11 + 3) & 0x3F);
            }
            for (int i = 0; i < 256; ++i) nib[i] = (uint8_t)((i * 7 + b * 13) & 0xF);
            uint16_t dh = f32_to_f16(q4k_d[b]), mh = f32_to_f16(q4k_dm[b]);
            unsigned char *dst = q4k_bytes + (size_t)b * TEST_Q4K_BLOCK_SIZE;
            dst[0] = (unsigned char)(dh & 0xFF); dst[1] = (unsigned char)(dh >> 8);
            dst[2] = (unsigned char)(mh & 0xFF); dst[3] = (unsigned char)(mh >> 8);
            for (int j = 0; j < 4; ++j) {
                dst[4 + j]     = (unsigned char)((sc[j] & 0x3F) | ((sc[j + 4] >> 4) << 6));
                dst[4 + j + 4] = (unsigned char)((mn[j] & 0x3F) | ((mn[j + 4] >> 4) << 6));
                dst[4 + j + 8] = (unsigned char)((sc[j + 4] & 0xF) | ((mn[j + 4] & 0xF) << 4));
            }
            for (int j = 0; j < 256; j += 64)
                for (int l = 0; l < 32; ++l)
                    dst[16 + (j / 64) * 32 + l] =
                        (unsigned char)(nib[j + l] | (nib[j + l + 32] << 4));
        }
    }
    {
        const float q6k_d[2] = { 0.50f, 0.125f };
        const int8_t sc_b[2][16] = {
            { 1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16 },
            { 16, -15, 14, -13, 12, -11, 10, -9, 8, -7, 6, -5, 4, -3, 2, -1 },
        };
        for (int b = 0; b < (int)TEST_K_BLOCKS; b++) {
            uint8_t level[256];
            for (int i = 0; i < 256; ++i) level[i] = (uint8_t)((i * 5 + b * 17) & 0x3F);
            unsigned char *dst = q6k_bytes + (size_t)b * TEST_Q6K_BLOCK_SIZE;
            memset(dst + 0, 0, 192);
            for (int i = 0; i < 16; ++i) dst[192 + i] = (unsigned char)sc_b[b][i];
            uint16_t dh = f32_to_f16(q6k_d[b]);
            dst[208] = (unsigned char)(dh & 0xFF); dst[209] = (unsigned char)(dh >> 8);
            for (int i = 0; i < 256; ++i) {
                int chunk = i / 128, rem = i % 128, sub = rem / 32, l = rem % 32;
                uint8_t ql4 = level[i] & 0xF;
                uint8_t qh2 = (level[i] >> 4) & 3;
                unsigned char *ql = dst + chunk * 64;
                unsigned char *qh = dst + 128 + chunk * 32;
                if (sub == 0)      { ql[l] |= (unsigned char)ql4;              qh[l] |= (unsigned char)(qh2 << 0); }
                else if (sub == 1) { ql[l + 32] |= (unsigned char)ql4;         qh[l] |= (unsigned char)(qh2 << 2); }
                else if (sub == 2) { ql[l] |= (unsigned char)(ql4 << 4);       qh[l] |= (unsigned char)(qh2 << 4); }
                else               { ql[l + 32] |= (unsigned char)(ql4 << 4);  qh[l] |= (unsigned char)(qh2 << 6); }
            }
        }
    }
    {
        const float iq4_d[2] = { 1.0f, 0.25f };
        const uint8_t ls_b[2][8] = {
            { 16, 33, 48, 63, 0, 8, 40, 55 },
            { 63, 0, 31, 32, 24, 17, 50, 7 },
        };
        for (int b = 0; b < (int)TEST_K_BLOCKS; b++) {
            uint8_t nib[256];
            for (int i = 0; i < 256; ++i) nib[i] = (uint8_t)((i * 3 + b * 5) & 0xF);
            unsigned char *dst = iq4xs_bytes + (size_t)b * TEST_IQ4XS_BLOCK_SIZE;
            uint16_t dh = f32_to_f16(iq4_d[b]);
            dst[0] = (unsigned char)(dh & 0xFF); dst[1] = (unsigned char)(dh >> 8);
            memset(dst + 2, 0, 6);
            uint16_t sh = 0;
            for (int ib = 0; ib < 8; ++ib) sh |= (uint16_t)((ls_b[b][ib] >> 4) & 3) << (2 * ib);
            dst[2] = (unsigned char)(sh & 0xFF); dst[3] = (unsigned char)(sh >> 8);
            for (int ib = 0; ib < 8; ++ib) {
                uint8_t low = ls_b[b][ib] & 0xF;
                if (ib & 1) dst[4 + ib / 2] |= (unsigned char)(low << 4);
                else        dst[4 + ib / 2] |= low;
            }
            for (int ib = 0; ib < 8; ++ib)
                for (int j = 0; j < 16; ++j)
                    dst[8 + ib * 16 + j] = (unsigned char)((nib[ib * 32 + j + 16] << 4) | nib[ib * 32 + j]);
        }
    }

    /* Fill the new-format dequant inputs deterministically. */
    for (uint32_t i = 0; i < DQ_CASE_COUNT; i++) {
        unsigned char *dst = (unsigned char *)h.mapped + s_dq_cases[i].op.off_in;
        for (uint32_t b = 0; b < s_dq_cases[i].num_blocks; b++)
            s_dq_cases[i].fill_fn(dst + (size_t)b * s_dq_cases[i].block_bytes,
                                  s_dq_cases[i].block_bytes);
    }

    /* Fill f32 quant sources. */
    float *q8_src = (float *)((char *)h.mapped + op_q8rt.off_in);
    float *q4_src = (float *)((char *)h.mapped + op_q4rt.off_in);
    for (uint32_t i = 0; i < TEST_QUANT_ELEMS; i++) {
        q8_src[i] = gen_quant_src(i);
        q4_src[i] = gen_quant_src(i);
    }
    for (uint32_t i = 0; i < Q_CASE_COUNT; i++) {
        float *src = (float *)((char *)h.mapped + s_q_cases[i].op.off_in);
        for (uint32_t e = 0; e < s_q_cases[i].num_blocks * s_q_cases[i].elems_per_block; e++)
            src[e] = s_q_cases[i].src_fn(e);
    }

    /* ── 11. CPU reference dequant (exact block formats) ────────────────── */
    float *exp_q8 = (float *)((char *)h.mapped + op_q8.off_expected);
    float *exp_q4 = (float *)((char *)h.mapped + op_q4.off_expected);

    for (uint32_t idx = 0; idx < total; idx++) {
        uint32_t lane = idx % TEST_ELEMS_PER_BLOCK;
        float q8_v = (float)(int)((lane % 7) - 3);
        exp_q8[idx] = TEST_Q8_SCALE * q8_v;
        float q4_v = (float)(int)((lane % 16) - 8);
        exp_q4[idx] = TEST_Q4_SCALE * q4_v;
    }

    float *exp_q4k  = (float *)((char *)h.mapped + op_q4k.off_expected);
    float *exp_q6k  = (float *)((char *)h.mapped + op_q6k.off_expected);
    float *exp_iq4  = (float *)((char *)h.mapped + op_iq4xs.off_expected);
    ref_dequant_q4k(q4k_bytes, exp_q4k, TEST_K_BLOCKS);
    ref_dequant_q6k(q6k_bytes, exp_q6k, TEST_K_BLOCKS);
    ref_dequant_iq4xs(iq4xs_bytes, exp_iq4, TEST_K_BLOCKS);

    for (uint32_t i = 0; i < DQ_CASE_COUNT; i++) {
        float *exp = (float *)((char *)h.mapped + s_dq_cases[i].op.off_expected);
        const unsigned char *src = (const unsigned char *)h.mapped + s_dq_cases[i].op.off_in;
        s_dq_cases[i].ref_fn(src, exp, (int)s_dq_cases[i].num_blocks);
    }

    /* ── 12. Record all dispatches into one command buffer ──────────────── */
    VkCommandBufferBeginInfo begin_info;
    memset(&begin_info, 0, sizeof(begin_info));
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    r = vkBeginCommandBuffer(h.cmd, &begin_info);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkquant: vkBeginCommandBuffer failed (%d)\n", (int)r);
        overall_pass = 0;
        goto cleanup;
    }

    overall_pass &= record_dispatch(
        vkquant_dequant_q8_0_f32(h.quant_ctx, h.cmd, TEST_NUM_BLOCKS,
                                 op_q8.in, op_q8.out),
        "dequant_q8_0_f32");
    overall_pass &= record_dispatch(
        vkquant_dequant_q4_0_f32(h.quant_ctx, h.cmd, TEST_NUM_BLOCKS,
                                 op_q4.in, op_q4.out),
        "dequant_q4_0_f32");
    overall_pass &= record_dispatch(
        vkquant_dequant_q4k_f32(h.quant_ctx, h.cmd, TEST_K_BLOCKS,
                                op_q4k.in, op_q4k.out),
        "dequant_q4k_f32");
    overall_pass &= record_dispatch(
        vkquant_dequant_q6k_f32(h.quant_ctx, h.cmd, TEST_K_BLOCKS,
                                op_q6k.in, op_q6k.out),
        "dequant_q6k_f32");
    overall_pass &= record_dispatch(
        vkquant_dequant_iq4xs_f32(h.quant_ctx, h.cmd, TEST_K_BLOCKS,
                                  op_iq4xs.in, op_iq4xs.out),
        "dequant_iq4xs_f32");

    for (uint32_t i = 0; i < DQ_CASE_COUNT; i++) {
        overall_pass &= record_dispatch(
            s_dq_cases[i].dequant_fn(h.quant_ctx, h.cmd, s_dq_cases[i].num_blocks,
                                     s_dq_cases[i].op.in, s_dq_cases[i].op.out),
            s_dq_cases[i].name);
    }

    overall_pass &= record_dispatch(
        vkquant_quantize_q8_0_f32(h.quant_ctx, h.cmd, TEST_QUANT_BLOCKS,
                                  op_q8rt.in, op_q8rt.qbytes),
        "quantize_q8_0_f32");
    overall_pass &= record_dispatch(
        vkquant_quantize_q4_0_f32(h.quant_ctx, h.cmd, TEST_QUANT_BLOCKS,
                                  op_q4rt.in, op_q4rt.qbytes),
        "quantize_q4_0_f32");

    /* New forward-quant formats. */
    for (uint32_t i = 0; i < Q_CASE_COUNT; i++) {
        overall_pass &= record_dispatch(
            s_q_cases[i].quant_fn(h.quant_ctx, h.cmd, s_q_cases[i].num_blocks,
                                  s_q_cases[i].op.in, s_q_cases[i].op.qbytes),
            s_q_cases[i].name);
    }

    /* Quantized bytes -> round-trip dequant. */
    record_compute_to_compute_barrier(h.cmd);

    overall_pass &= record_dispatch(
        vkquant_dequant_q8_0_f32(h.quant_ctx, h.cmd, TEST_QUANT_BLOCKS,
                                 op_q8rt.qbytes, op_q8rt.out),
        "dequant(q8_0 rt)");
    overall_pass &= record_dispatch(
        vkquant_dequant_q4_0_f32(h.quant_ctx, h.cmd, TEST_QUANT_BLOCKS,
                                 op_q4rt.qbytes, op_q4rt.out),
        "dequant(q4_0 rt)");

    for (uint32_t i = 0; i < Q_CASE_COUNT; i++) {
        overall_pass &= record_dispatch(
            s_q_cases[i].dequant_fn(h.quant_ctx, h.cmd, s_q_cases[i].num_blocks,
                                    s_q_cases[i].op.qbytes, s_q_cases[i].op.out),
            "dequant rt");
    }

    /* Make the shader writes visible to the transfer readback copies. */
    record_compute_to_transfer_barrier(h.cmd);

    record_copy_readback(h.cmd, op_q8.out, h.staging,
                         op_q8.off_readback, total * sizeof(float));
    record_copy_readback(h.cmd, op_q4.out, h.staging,
                         op_q4.off_readback, total * sizeof(float));
    record_copy_readback(h.cmd, op_q4k.out, h.staging,
                         op_q4k.off_readback, TEST_K_ELEMS * sizeof(float));
    record_copy_readback(h.cmd, op_q6k.out, h.staging,
                         op_q6k.off_readback, TEST_K_ELEMS * sizeof(float));
    record_copy_readback(h.cmd, op_iq4xs.out, h.staging,
                         op_iq4xs.off_readback, TEST_K_ELEMS * sizeof(float));
    record_copy_readback(h.cmd, op_q8rt.out, h.staging,
                         op_q8rt.off_readback, TEST_QUANT_ELEMS * sizeof(float));
    record_copy_readback(h.cmd, op_q4rt.out, h.staging,
                         op_q4rt.off_readback, TEST_QUANT_ELEMS * sizeof(float));

    for (uint32_t i = 0; i < DQ_CASE_COUNT; i++) {
        record_copy_readback(h.cmd, s_dq_cases[i].op.out, h.staging,
                             s_dq_cases[i].op.off_readback,
                             (VkDeviceSize)s_dq_cases[i].num_blocks *
                             s_dq_cases[i].elems_per_block * sizeof(float));
    }
    for (uint32_t i = 0; i < Q_CASE_COUNT; i++) {
        record_copy_readback(h.cmd, s_q_cases[i].op.out, h.staging,
                             s_q_cases[i].op.off_readback,
                             (VkDeviceSize)s_q_cases[i].num_blocks *
                             s_q_cases[i].elems_per_block * sizeof(float));
    }

    r = vkEndCommandBuffer(h.cmd);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkquant: vkEndCommandBuffer failed (%d)\n", (int)r);
        overall_pass = 0;
        goto cleanup;
    }

    /* ── 13. One submit, one fence, device idle ─────────────────────────── */
    VkFenceCreateInfo fence_info;
    memset(&fence_info, 0, sizeof(fence_info));
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    r = vkCreateFence(h.device, &fence_info, NULL, &h.fence);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkquant: vkCreateFence failed (%d)\n", (int)r);
        overall_pass = 0;
        goto cleanup;
    }

    VkSubmitInfo submit_info;
    memset(&submit_info, 0, sizeof(submit_info));
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &h.cmd;

    r = vkQueueSubmit(h.queue, 1, &submit_info, h.fence);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkquant: vkQueueSubmit failed (%d)\n", (int)r);
        overall_pass = 0;
        goto cleanup;
    }
    vkWaitForFences(h.device, 1, &h.fence, VK_TRUE, UINT64_MAX);
    vkQueueWaitIdle(h.queue);

    /* ── 14. Compare GPU results against the CPU references ─────────────── */
    overall_pass &= check_output("dequant_q8_0_f32", h.mapped, op_q8.off_readback,
                                 exp_q8, total, TEST_F32_TOLERANCE);
    overall_pass &= check_output("dequant_q4_0_f32", h.mapped, op_q4.off_readback,
                                 exp_q4, total, TEST_F32_TOLERANCE);
    overall_pass &= check_output("dequant_q4k_f32", h.mapped, op_q4k.off_readback,
                                 exp_q4k, TEST_K_ELEMS, TEST_F32_TOLERANCE);
    overall_pass &= check_output("dequant_q6k_f32", h.mapped, op_q6k.off_readback,
                                 exp_q6k, TEST_K_ELEMS, TEST_F32_TOLERANCE);
    overall_pass &= check_output("dequant_iq4xs_f32", h.mapped, op_iq4xs.off_readback,
                                 exp_iq4, TEST_K_ELEMS, TEST_F32_TOLERANCE);
    overall_pass &= check_roundtrip("quant+q8_0_rt", h.mapped, op_q8rt.off_readback,
                                    TEST_QUANT_ELEMS, TEST_QUANT_TOLERANCE, gen_quant_src);
    overall_pass &= check_roundtrip("quant+q4_0_rt", h.mapped, op_q4rt.off_readback,
                                    TEST_QUANT_ELEMS, TEST_QUANT_TOLERANCE, gen_quant_src);

    for (uint32_t i = 0; i < DQ_CASE_COUNT; i++) {
        overall_pass &= check_output(s_dq_cases[i].name, h.mapped,
                                     s_dq_cases[i].op.off_readback,
                                     (const float *)((const char *)h.mapped + s_dq_cases[i].op.off_expected),
                                     s_dq_cases[i].num_blocks * s_dq_cases[i].elems_per_block,
                                     TEST_F32_TOLERANCE);
    }
    for (uint32_t i = 0; i < Q_CASE_COUNT; i++) {
        overall_pass &= check_roundtrip(s_q_cases[i].name, h.mapped,
                                        s_q_cases[i].op.off_readback,
                                        s_q_cases[i].num_blocks * s_q_cases[i].elems_per_block,
                                        TEST_QUANT_TOLERANCE, s_q_cases[i].src_fn);
    }

cleanup:
    if (h.quant_ctx) vkquant_destroy_context(h.quant_ctx);
    if (op_q8.in)   vkDestroyBuffer(h.device, op_q8.in, NULL);
    if (op_q8.out)  vkDestroyBuffer(h.device, op_q8.out, NULL);
    if (op_q4.in)   vkDestroyBuffer(h.device, op_q4.in, NULL);
    if (op_q4.out)  vkDestroyBuffer(h.device, op_q4.out, NULL);
    if (op_q4k.in)  vkDestroyBuffer(h.device, op_q4k.in, NULL);
    if (op_q4k.out) vkDestroyBuffer(h.device, op_q4k.out, NULL);
    if (op_q6k.in)  vkDestroyBuffer(h.device, op_q6k.in, NULL);
    if (op_q6k.out) vkDestroyBuffer(h.device, op_q6k.out, NULL);
    if (op_iq4xs.in)  vkDestroyBuffer(h.device, op_iq4xs.in, NULL);
    if (op_iq4xs.out) vkDestroyBuffer(h.device, op_iq4xs.out, NULL);
    if (op_q8rt.in)    vkDestroyBuffer(h.device, op_q8rt.in, NULL);
    if (op_q8rt.qbytes) vkDestroyBuffer(h.device, op_q8rt.qbytes, NULL);
    if (op_q8rt.out)   vkDestroyBuffer(h.device, op_q8rt.out, NULL);
    if (op_q4rt.in)    vkDestroyBuffer(h.device, op_q4rt.in, NULL);
    if (op_q4rt.qbytes) vkDestroyBuffer(h.device, op_q4rt.qbytes, NULL);
    if (op_q4rt.out)   vkDestroyBuffer(h.device, op_q4rt.out, NULL);
    for (uint32_t i = 0; i < DQ_CASE_COUNT; i++) {
        if (s_dq_cases[i].op.in)  vkDestroyBuffer(h.device, s_dq_cases[i].op.in, NULL);
        if (s_dq_cases[i].op.out) vkDestroyBuffer(h.device, s_dq_cases[i].op.out, NULL);
    }
    for (uint32_t i = 0; i < Q_CASE_COUNT; i++) {
        if (s_q_cases[i].op.in)       vkDestroyBuffer(h.device, s_q_cases[i].op.in, NULL);
        if (s_q_cases[i].op.qbytes)   vkDestroyBuffer(h.device, s_q_cases[i].op.qbytes, NULL);
        if (s_q_cases[i].op.out)      vkDestroyBuffer(h.device, s_q_cases[i].op.out, NULL);
    }
    if (h.staging)  vkDestroyBuffer(h.device, h.staging, NULL);
    if (h.mapped)   vkUnmapMemory(h.device, h.mem);
    if (h.mem != VK_NULL_HANDLE) vkFreeMemory(h.device, h.mem, NULL);
    if (h.fence != VK_NULL_HANDLE) vkDestroyFence(h.device, h.fence, NULL);
    if (h.cmd != VK_NULL_HANDLE)
        vkFreeCommandBuffers(h.device, h.cmd_pool, 1, &h.cmd);
    if (h.cmd_pool != VK_NULL_HANDLE)
        vkDestroyCommandPool(h.device, h.cmd_pool, NULL);
    if (h.device != VK_NULL_HANDLE) vkDestroyDevice(h.device, NULL);
    if (h.instance != VK_NULL_HANDLE) vkDestroyInstance(h.instance, NULL);

    printf("test_vkquant: %s\n", overall_pass ? "PASS" : "FAIL");
    return overall_pass ? 0 : 1;
}
