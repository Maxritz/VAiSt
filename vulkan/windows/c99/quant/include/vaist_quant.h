#ifndef VAIST_QUANT_H
#define VAIST_QUANT_H
#include "vaist_core.h"
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Added: VAIST_TERNARY/VAIST_BINARY feed the MatMul-free compute path
 * (2406.02528 ternary; 2608.03142 / 2608.01528 bitnet/xnor). */
typedef enum VaistQuantType {
    VAIST_Q8_0=0,
    VAIST_Q4_0=1,
    VAIST_Q4_1=2,
    VAIST_TERNARY=3,      /* int8 signs {-1,0,+1} + per-row scale */
    VAIST_BINARY=4,      /* 1-bit packed weights + per-row scale */
} VaistQuantType;

VAIST_API VaistStatus vaist_quantize_f32(VaistQuantType q,const float*src,size_t n,void*dst,size_t cap,size_t*used);
VAIST_API VaistStatus vaist_dequantize_f32(VaistQuantType q,const void*src,size_t bytes,float*dst,size_t n);
VAIST_API size_t vaist_quant_block_bytes(VaistQuantType q);
/* New: encode a weight matrix into ternary/binary *without* a full
 * dequant round-trip — emits the sign array + scale used by the MatMul-free
 * matvec shaders directly. */
VAIST_API VaistStatus vaist_pack_ternary(const float*src,size_t n,int8_t*signs,float*scale); /* out: n signs, 1 scale */
VAIST_API VaistStatus vaist_pack_binary(const float*src,size_t n,uint8_t*bits,float*scale); /* out: ceil(n/8) bits, 1 scale */
#ifdef __cplusplus
}
#endif
#endif
