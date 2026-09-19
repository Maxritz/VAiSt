#ifndef VAIST_QUANT_H
#define VAIST_QUANT_H
#include "vaist_core.h"
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* GGUF-compatible quantization types.
 * Values 0-4 preserve ABI from the original vaist_quant.h (q8_0/q4_0/q4_1 +
 * MatMul-free ternary/binary). 5-17 extend coverage to the full K-quant /
 * i-Quant spectrum from llama.cpp (b2899), which FreeToken's csrc/gguf
 * dequantize.cuh + vecdotq.cuh implement via CUDA.  These are dequant-only
 * (no dp4a / warp shuffle) -> clean C99 / Vulkan compute-shader ports. */
typedef enum VaistQuantType {
    VAIST_Q8_0=0,   /* ggml q8_0:  8-bit, per-block scale (QK8_0=32) */
    VAIST_Q4_0=1,   /* ggml q4_0:  4-bit symmetric (QK4_0=32) */
    VAIST_Q4_1=2,   /* ggml q4_1:  4-bit asymmetric (QK4_1=32) */
    VAIST_TERNARY=3,       /* int8 signs {-1,0,+1} + per-row scale (MatMul-free) */
    VAIST_BINARY=4,        /* 1-bit packed weights + per-row scale (MatMul-free) */
    /* ---- K-quants (QK_K=256, except where noted) ---- */
    VAIST_Q2_K=5,
    VAIST_Q3_K=6,
    VAIST_Q4_K=7,
    VAIST_Q5_0=8,
    VAIST_Q5_1=9,
    VAIST_Q5_K=10,
    VAIST_Q6_K=11,
    /* ---- i-Quants ---- */
    VAIST_IQ2_XXS=12,
    VAIST_IQ2_XS=13,
    VAIST_IQ2_S=14,
    VAIST_IQ3_XXS=15,
    VAIST_IQ3_S=16,
    VAIST_IQ3_XS=17,   /* = iq3_s in some llama.cpp branches */
    VAIST_IQ1_S=18,
    VAIST_IQ1_M=19,
    VAIST_IQ4_NL=20,  /* QK4_NL=32, q4_0-style nonlinearity-aware */
    VAIST_IQ4_XS=21,
     VAIST_Q8_1=22,    /* q8_1: 8-bit with delta+sum (activation layout in FreeToken mmq) */
     VAIST_NVFP4=23,   /* NVIDIA FP4 block: 36 B/block of 64 (E2M1 ×32 + E4M3 scale ×1) */
     VAIST_TQ2_0=24,   /* GGUF Ternary T2_0: 66 B/block of 256 (2-bit packed + FP16 scale) */
     VAIST_MXFP4=25,   /* AMD MXFP4: 17 B/block of 32 (4-bit + E8M0 scale) */
     VAIST_Q_COUNT=26,
} VaistQuantType;

#define VAIST_QK4_0 32
#define VAIST_QK_K  256   /* block size for K-quants / i-Quants (llama.cpp default) */

/* GGUF byte-exact block layouts (pack(1)).
 * Exposed so callers (Vulkan compute shaders, tests) can construct blocks
 * directly.  These match llama.cpp's block_* structs from ggml-common.h. */
#pragma pack(push,1)
typedef struct { uint16_t d; uint8_t qs[VAIST_QK4_0/2]; } vaist_block_q4_0;
typedef struct { uint32_t dm; uint8_t qs[VAIST_QK4_0/2]; } vaist_block_q4_1;
typedef struct { uint16_t d; uint8_t qh[4]; uint8_t qs[VAIST_QK4_0/2]; } vaist_block_q5_0;
typedef struct { uint32_t dm; uint8_t qh[4]; uint8_t qs[VAIST_QK4_0/2]; } vaist_block_q5_1;
typedef struct { uint16_t d; int8_t qs[VAIST_QK4_0]; } vaist_block_q8_0;
typedef struct { uint32_t ds; int8_t qs[VAIST_QK4_0]; } vaist_block_q8_1;
typedef struct { uint8_t scales[VAIST_QK_K/16]; uint8_t qs[VAIST_QK_K/4]; uint32_t dm; } vaist_block_q2_K;
typedef struct { uint8_t hmask[VAIST_QK_K/8]; uint8_t qs[VAIST_QK_K/4]; uint8_t scales[12]; uint16_t d; } vaist_block_q3_K;
typedef struct { uint32_t dm; uint8_t scales[3*VAIST_QK_K/64]; uint8_t qs[VAIST_QK_K/2]; } vaist_block_q4_K;
typedef struct { uint32_t dm; uint8_t scales[12]; uint8_t qh[VAIST_QK_K/8]; uint8_t qs[VAIST_QK_K/2]; } vaist_block_q5_K;
typedef struct { uint8_t ql[VAIST_QK_K/2]; uint8_t qh[VAIST_QK_K/4]; int8_t scales[VAIST_QK_K/16]; uint16_t d; } vaist_block_q6_K;
typedef struct { uint16_t d; uint8_t qs[VAIST_QK_K/4]; uint8_t scales[VAIST_QK_K/32]; } vaist_block_iq2_xxs;
typedef struct { uint16_t d; uint8_t qs[VAIST_QK_K/4]; } vaist_block_iq2_xs;
typedef struct { uint16_t d; uint8_t qs[VAIST_QK_K/4]; uint8_t qh[VAIST_QK_K/32]; uint8_t scales[VAIST_QK_K/32]; } vaist_block_iq2_s;
typedef struct { uint16_t d; uint8_t qs[3*(VAIST_QK_K/8)]; } vaist_block_iq3_xxs;
typedef struct { uint16_t d; uint8_t qs[VAIST_QK_K/4]; uint8_t qh[VAIST_QK_K/32]; uint8_t signs[VAIST_QK_K/8]; uint8_t scales[4]; } vaist_block_iq3_s;
typedef struct { uint8_t qs[VAIST_QK_K/8]; uint8_t qh[VAIST_QK_K/16]; uint8_t scales[VAIST_QK_K/32]; } vaist_block_iq1_m;
typedef struct { uint16_t d; uint8_t qs[VAIST_QK_K/8]; uint16_t qh[VAIST_QK_K/32]; } vaist_block_iq1_s;
typedef struct { uint16_t d; uint8_t qs[VAIST_QK4_0/2]; } vaist_block_iq4_nl;
typedef struct { uint16_t d; uint32_t qs; uint8_t scales_l[VAIST_QK_K/64]; uint8_t qs_extra[VAIST_QK_K/8 - 5]; } vaist_block_iq4_xs;
/* NVIDIA NVFP4: 64 elements in 36 bytes — 32 FP4 nibbles (16 bytes) + 1 E4M3 scale (4 bytes) + padding */
typedef struct { uint8_t qs[32]; uint8_t _pad[16]; uint8_t scale_bytes[4]; } vaist_block_nvfp4;
/* GGUF TQ2_0: 256 elements in 66 bytes — 64 bytes 2-bit packed (256×2bits) + F16 scale */
typedef struct { uint8_t qs[64]; uint16_t d; } vaist_block_tq2_0;
/* AMD MXFP4: 32 elements in 17 bytes — 16 FP4 nibbles + 1 E8M0 scale byte */
typedef struct { uint8_t qs[16]; uint8_t scale; } vaist_block_mxfp4;
#pragma pack(pop)

VAIST_API VaistStatus vaist_quantize_f32(VaistQuantType q,const float*src,size_t n,void*dst,size_t cap,size_t*used);
VAIST_API VaistStatus vaist_dequantize_f32(VaistQuantType q,const void*src,size_t bytes,float*dst,size_t n);
VAIST_API size_t vaist_quant_block_bytes(VaistQuantType q);
VAIST_API size_t vaist_quant_block_size(VaistQuantType q);  /* #elements per block (32 or 256) */
/* New: encode a weight matrix into ternary/binary *without* a full
 * dequant round-trip — emits the sign array + scale used by the MatMul-free
 * matvec shaders directly. */
VAIST_API VaistStatus vaist_pack_ternary(const float*src,size_t n,int8_t*signs,float*scale); /* out: n signs, 1 scale */
VAIST_API VaistStatus vaist_pack_binary(const float*src,size_t n,uint8_t*bits,float*scale); /* out: ceil(n/8) bits, 1 scale */

/* --- FP8 support (E4M3 / E5M2) --- */
/**
 * \brief Encode float32 to FP8 E4M3.
 * \note USAGE: Scalar utility for weight conversion and KV cache quantization.
 *       Does NOT integrate with model forward pass — caller must use the result
 *       to populate GPU buffers and dispatch appropriate dequant kernels.
 */
VAIST_API int vaist_fp8_e4m3_encode(float f32);
VAIST_API float vaist_fp8_e4m3_decode(int v);
VAIST_API int vaist_fp8_e5m2_encode(float f32);
VAIST_API float vaist_fp8_e5m2_decode(int v);
/* dt: 8=VAIST_F8_E4M3, 9=VAIST_F8_E5M2 (matches VaistDType values) */
VAIST_API VaistStatus vaist_quantize_fp8(const float*src,size_t n,void*dst,int dt);
#ifdef __cplusplus
}
#endif
#endif
