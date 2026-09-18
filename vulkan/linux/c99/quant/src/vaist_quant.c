#include "vaist_quant.h"
#include "vaist_quant_tables.h"
#include <stdint.h>
#include <string.h>
#include <math.h>

#define QK4_0 32
#define QK4_1 32
#define QK5_0 32
#define QK8_0 32
#define QK8_1 32
#define QK_K  256
#define QK4_NL 32

#define IQ1S_DELTA 0.125f
#define IQ1M_DELTA 0.125f

#pragma pack(push,1)
typedef struct { uint16_t d; uint8_t qs[QK4_0/2]; } block_q4_0;
typedef struct { uint32_t dm; uint8_t qs[QK4_1/2]; } block_q4_1;
typedef struct { uint16_t d; uint8_t qh[4]; uint8_t qs[QK5_0/2]; } block_q5_0;
typedef struct { uint32_t dm; uint8_t qh[4]; uint8_t qs[QK5_0/2]; } block_q5_1;
typedef struct { uint16_t d; int8_t qs[QK8_0]; } block_q8_0;
typedef struct { uint32_t ds; int8_t qs[QK8_0]; } block_q8_1;
typedef struct { uint8_t scales[QK_K/16]; uint8_t qs[QK_K/4]; uint32_t dm; } block_q2_K;
typedef struct { uint8_t hmask[QK_K/8]; uint8_t qs[QK_K/4]; uint8_t scales[12]; uint16_t d; } block_q3_K;
typedef struct { uint32_t dm; uint8_t scales[3*QK_K/64]; uint8_t qs[QK_K/2]; } block_q4_K;
typedef struct { uint32_t dm; uint8_t scales[12]; uint8_t qh[QK_K/8]; uint8_t qs[QK_K/2]; } block_q5_K;
typedef struct { uint8_t ql[QK_K/2]; uint8_t qh[QK_K/4]; int8_t scales[QK_K/16]; uint16_t d; } block_q6_K;
typedef struct { uint16_t d; uint16_t qs[QK_K/8]; } block_iq2_xxs;
typedef struct { uint16_t d; uint16_t qs[QK_K/8]; uint8_t scales[QK_K/32]; } block_iq2_xs;
typedef struct { uint16_t d; uint8_t qs[QK_K/4]; uint8_t qh[QK_K/32]; uint8_t scales[QK_K/32]; } block_iq2_s;
typedef struct { uint16_t d; uint8_t qs[3*(QK_K/8)]; } block_iq3_xxs;
typedef struct { uint16_t d; uint8_t qs[QK_K/4]; uint8_t qh[QK_K/32]; uint8_t signs[QK_K/8]; uint8_t scales[4]; } block_iq3_s;
typedef struct { uint8_t qs[QK_K/8]; uint8_t qh[QK_K/16]; uint8_t scales[QK_K/32]; } block_iq1_m;
typedef struct { uint16_t d; uint8_t qs[QK_K/8]; uint16_t qh[QK_K/32]; } block_iq1_s;
typedef struct { uint16_t d; uint8_t qs[QK4_NL/2]; } block_iq4_nl;
typedef struct { uint16_t d; uint32_t scales_h; uint8_t scales_l[QK_K/64]; uint8_t qs[QK_K/2]; } block_iq4_xs;
#pragma pack(pop)

/* ---- portable fp16 <-> fp32 (no intrinsics; VAiSt-safe) ---- */
static float f16_to_f32(uint16_t h){
    uint32_t sign=((uint32_t)h>>15)&1u;
    uint32_t exp=(h>>10)&0x1fu;
    uint32_t mant=h&0x3ffu;
    uint32_t f;
    if(exp==0u){
        if(mant==0u){ f=(sign<<31); }
        else {
            uint32_t m=mant; int e=1;
            while((m&0x400u)==0u){ m<<=1; e--; }
            m&=~0x400u;
            f=(sign<<31)|(((uint32_t)(127-15-e))<<23)|m;
        }
    } else if(exp==0x1fu){
        f=(sign<<31)|0x7f800000u|(mant?0x400000u:0u);
    } else {
        f=(sign<<31)|(((uint32_t)exp+(127u-15u))<<23)|(mant<<13);
    }
    float out; memcpy(&out,&f,4); return out;
}
static uint16_t f32_to_f16(float v){
    /* IEEE 754 half-precision conversion (VAiSt-safe, no intrinsics).
     * f32 unbiased exp e32 = (f>>23)&0xff - 127.
     * fp16 normal: e32 in [-14..15] -> biased (e32+15) in [1..30].
     * fp16 denorm: e32 == -15 (i.e. f32 exp field == 112). */
    uint32_t f; memcpy(&f,&v,4);
    uint32_t sign=(f>>16)&0x8000u;
    int e32=(int)((f>>23)&0xffu)-127;
    uint32_t mant=f&0x7fffffu;
    if(e32>=16){ return (uint16_t)(sign|0x7c00u); }            /* overflow -> Inf */
    if(e32< -15){ return (uint16_t)sign; }                       /* underflow -> 0 */
    if(e32== -15){
        /* denormal fp16: the f32 mantissa (with implicit bit) shifts to 10-bit */
        uint32_t m14=(mant|0x800000u)>>(23-14);                  /* keep 14 bits + implicit */
        uint32_t m=m14>>4;                                       /* 10-bit mantissa */
        uint32_t g=(m14>>3)&1u, d=(m14>>2)&1u;                   /* guard/round of dropped bits */
        m+=(g&&(d||(m14&1u)))?1u:0u;
        if(m>=0x400u){ return (uint16_t)(sign|0x7c00u); }       /* carry overflow -> Inf */
        return (uint16_t)(sign|m);
    }
    /* normal fp16 */
    uint32_t mh=mant>>13u;
    uint32_t gr=(mant>>12)&1u, gn=(mant>>11)&1u, gd=(mant>>10)&1u;
    mh+=((gr&&(gn||gd)))?1u:0u;
    if(mh>=0x400u){ mh=0; e32++; if(e32>=16) return (uint16_t)(sign|0x7c00u); }
    return (uint16_t)(sign|((uint32_t)(e32+15))<<10|mh);
}

/* half2 decode: lo = first half (delta), hi = second half (min/sum) */
static float f16_lo(uint32_t h2){ return f16_to_f32((uint16_t)(h2&0xffffu)); }
static float f16_hi(uint32_t h2){ return f16_to_f32((uint16_t)((h2>>16)&0xffffu)); }

static size_t blocks32(size_t n){ return (n+31u)/32u; }
static size_t blocks256(size_t n){ return (n+255u)/256u; }

/* get_scale_min_k4: decode 6-bit scale+min nibble pair j from q4_K/q5_K scales[] */
static void get_scale_min_k4(int j,const uint8_t* q,uint8_t* scale,uint8_t* min){
    if(j<4){ *scale=q[j]&63; *min=q[j+4]&63; }
    else   { *scale=(q[j+4]&0xF)|((q[j-4]>>6)<<4); *min=(q[j+4]>>4)|((q[j-0]>>6)<<4); }
}

/* ---- block size ---- */
VAIST_API size_t vaist_quant_block_bytes(VaistQuantType q){
    switch(q){
    case VAIST_Q8_0:  return sizeof(block_q8_0);
    case VAIST_Q4_0:  return sizeof(block_q4_0);
    case VAIST_Q4_1:  return sizeof(block_q4_1);
    case VAIST_Q8_1:  return sizeof(block_q8_1);
    case VAIST_Q5_0:  return sizeof(block_q5_0);
    case VAIST_Q5_1:  return sizeof(block_q5_1);
    case VAIST_Q2_K:  return sizeof(block_q2_K);
    case VAIST_Q3_K:  return sizeof(block_q3_K);
    case VAIST_Q4_K:  return sizeof(block_q4_K);
    case VAIST_Q5_K:  return sizeof(block_q5_K);
    case VAIST_Q6_K:  return sizeof(block_q6_K);
    case VAIST_IQ2_XXS: return sizeof(block_iq2_xxs);
    case VAIST_IQ2_XS:  return sizeof(block_iq2_xs);
    case VAIST_IQ2_S:   return sizeof(block_iq2_s);
    case VAIST_IQ3_XXS: return sizeof(block_iq3_xxs);
    case VAIST_IQ3_S:   return sizeof(block_iq3_s);
    case VAIST_IQ3_XS:  return sizeof(block_iq3_s);
    case VAIST_IQ1_S:   return sizeof(block_iq1_s);
    case VAIST_IQ1_M:   return sizeof(block_iq1_m);
    case VAIST_IQ4_NL:  return sizeof(block_iq4_nl);
    case VAIST_IQ4_XS:  return sizeof(block_iq4_xs);
    case VAIST_TERNARY: case VAIST_BINARY: return 0;
    default: return 0;
    }
}

/* ---- block count for N values ---- */
static size_t blocks_for(VaistQuantType q,size_t n){
    switch(q){
    case VAIST_Q8_0: case VAIST_Q4_0: case VAIST_Q4_1:
    case VAIST_Q5_0: case VAIST_Q5_1: case VAIST_Q8_1: return blocks32(n);
    case VAIST_Q2_K: case VAIST_Q3_K: case VAIST_Q4_K: case VAIST_Q5_K: case VAIST_Q6_K:
    case VAIST_IQ2_XXS: case VAIST_IQ2_XS: case VAIST_IQ2_S: case VAIST_IQ3_XXS:
    case VAIST_IQ3_S: case VAIST_IQ3_XS: case VAIST_IQ1_S: case VAIST_IQ1_M:
    case VAIST_IQ4_XS: return blocks256(n);
    case VAIST_IQ4_NL: return blocks32(n); /* QK4_NL=32 */
    default: return 0;
    }
}

VAIST_API size_t vaist_quant_block_size(VaistQuantType q){
    switch(q){
    case VAIST_Q8_0: case VAIST_Q4_0: case VAIST_Q4_1:
    case VAIST_Q5_0: case VAIST_Q5_1: case VAIST_Q8_1: return 32;
    case VAIST_Q2_K: case VAIST_Q3_K: case VAIST_Q4_K: case VAIST_Q5_K: case VAIST_Q6_K:
    case VAIST_IQ2_XXS: case VAIST_IQ2_XS: case VAIST_IQ2_S: case VAIST_IQ3_XXS:
    case VAIST_IQ3_S: case VAIST_IQ3_XS: case VAIST_IQ1_S: case VAIST_IQ1_M:
    case VAIST_IQ4_XS: return 256;
    case VAIST_IQ4_NL: return 32; /* QK4_NL=32 */
    default: return 0;
    }
}

VAIST_API VaistStatus vaist_quantize_f32(VaistQuantType q,const float*s,size_t n,void*d,size_t cap,size_t*u){
    /* Only the base 3 (+ q8_1) are quantize-able here; K-quants/i-Quants use
     * upstream ggml quantizer tools.  TERRY/BINARY handled by pack_*(). */
    size_t nb,i,j,bs; float a,sc; unsigned char*o=(unsigned char*)d;
    if(!s||!d||!n||!u) return VAIST_INVALID_ARGUMENT;
    if(q!=VAIST_Q8_0 && q!=VAIST_Q4_0 && q!=VAIST_Q4_1 && q!=VAIST_Q8_1)
        return VAIST_INVALID_ARGUMENT;
    nb=blocks32(n); bs=vaist_quant_block_bytes(q);
    if(cap<nb*bs) return VAIST_OUT_OF_MEMORY;
    for(i=0;i<nb;i++){
        size_t len=(n-i*32u<32u)?(n-i*32u):32u;
        a=0;
        for(j=0;j<len;j++){ float x=fabsf(s[i*32u+j]); if(x>a)a=x; }
        sc=(q==VAIST_Q8_0||(q==VAIST_Q8_1))?(a/127.0f):(a/7.0f);
        if(sc==0)sc=1.0f;
        if(q==VAIST_Q8_0){
            block_q8_0*b=(block_q8_0*)(o+i*bs); b->d=f32_to_f16(sc);
            for(j=0;j<32;j++) b->qs[j]=(j<len)?(int8_t)lrintf(s[i*32u+j]/sc):0;
        } else if(q==VAIST_Q8_1){
            block_q8_1*b=(block_q8_1*)(o+i*bs);
            float sum=0; for(j=0;j<len;j++) sum+=s[i*32u+j];
            b->ds=((uint32_t)f32_to_f16(sc))|((uint32_t)f32_to_f16(sum))<<16;
            for(j=0;j<32;j++) b->qs[j]=(j<len)?(int8_t)lrintf(s[i*32u+j]/sc):0;
        } else if(q==VAIST_Q4_0){
            block_q4_0*b=(block_q4_0*)(o+i*bs); b->d=f32_to_f16(sc);
            for(j=0;j<16;j++){
                int lo=(j*2<len)?(int)(lrintf(s[i*32u+j*2]/sc))+8:0;
                int hi=(j*2+1<len)?(int)(lrintf(s[i*32u+j*2+1]/sc))+8:0;
                b->qs[j]=(uint8_t)((lo&0xf)|((hi&0xf)<<4));
            }
        } else if(q==VAIST_Q4_1){
            block_q4_1*b=(block_q4_1*)(o+i*bs);
            float sum=0; for(j=0;j<len;j++) sum+=s[i*32u+j];
            b->dm=((uint32_t)f32_to_f16(sc))|((uint32_t)f32_to_f16(sum))<<16;
            for(j=0;j<16;j++){
                int lo=(j*2<len)?(int)(lrintf(s[i*32u+j*2]/sc))+8:0;
                int hi=(j*2+1<len)?(int)(lrintf(s[i*32u+j*2+1]/sc))+8:0;
                b->qs[j]=(uint8_t)((lo&0xf)|((hi&0xf)<<4));
            }
        }
    }
    *u=nb*bs;
    return VAIST_OK;
}

/* ---- per-quant dequantize (scalar, block-parallel).  Each call dequantizes
 * one 32- or 256-element block.  Caller allocates >= n floats (n may be the
 * full vector length); trailing partial blocks are zero-filled to block size.
 * Mirrors the CUDA dequantize_block_* thread-mapping by iterating per-lane. ---- */
#define OUT(o,idx,val) do{ if((size_t)(idx)<n ) o[(idx)]=(val); }while(0)

VAIST_API VaistStatus vaist_dequantize_f32(VaistQuantType q,const void*src,size_t bytes,float*o,size_t n){
    if(!src||!o||!n) return VAIST_INVALID_ARGUMENT;
    size_t bs=vaist_quant_block_bytes(q);
    size_t nb=blocks_for(q,n);
    if(bytes<nb*bs) return VAIST_INVALID_ARGUMENT;
    const unsigned char*d=(const unsigned char*)src;
    size_t i;
    for(i=0;i<nb;i++){
        size_t base=i*bs;
        size_t nfull=(n-i*256u)<256u?(size_t)(n-i*256u):256u; /* clamped below per type */
        switch(q){
        case VAIST_Q8_0: {
            const block_q8_0*b=(const void*)(d+base);
            float sc=f16_to_f32(b->d);
            for(size_t j=0;j<32;j++) OUT(o,i*32+j, sc*(float)b->qs[j]);
            break;
        }
        case VAIST_Q4_0: {
            const block_q4_0*b=(const void*)(d+base);
            float sc=f16_to_f32(b->d);
            for(size_t j=0;j<16;j++){
                int8_t lo=(b->qs[j]&0xF)-8, hi=(b->qs[j]>>4)-8;
                OUT(o,i*32+j*2+0, sc*(float)lo); OUT(o,i*32+j*2+1, sc*(float)hi);
            }
            break;
        }
        case VAIST_Q4_1: {
            const block_q4_1*b=(const void*)(d+base);
            float d4=f16_lo(b->dm), m4=f16_hi(b->dm);
            for(size_t j=0;j<16;j++){
                float lo=d4*(float)(b->qs[j]&0xF)+m4;
                float hi=d4*(float)(b->qs[j]>>4)+m4;
                OUT(o,i*32+j*2+0, lo); OUT(o,i*32+j*2+1, hi);
            }
            break;
        }
        case VAIST_Q5_0: {
            const block_q5_0*b=(const void*)(d+base);
            float sc=f16_to_f32(b->d);
            uint32_t qh; memcpy(&qh,b->qh,4);
            for(size_t j=0;j<16;j++){
                int xh0=(qh>>j)&0x10, xh1=(qh>>(j+12))&0x10;
                int8_t lo=(int8_t)(((b->qs[j]&0xF)|xh0)-16);
                int8_t hi=(int8_t)(((b->qs[j]>>4)|xh1)-16);
                OUT(o,i*32+j*2+0, sc*(float)lo); OUT(o,i*32+j*2+1, sc*(float)hi);
            }
            break;
        }
        case VAIST_Q5_1: {
            const block_q5_1*b=(const void*)(d+base);
            float d5=f16_lo(b->dm), m5=f16_hi(b->dm);
            uint32_t qh; memcpy(&qh,b->qh,4);
            for(size_t j=0;j<16;j++){
                int xh0=(qh>>j)&0x10, xh1=(qh>>(j+12))&0x10;
                float lo=d5*(float)((b->qs[j]&0xF)|xh0)+m5;
                float hi=d5*(float)((b->qs[j]>>4)|xh1)+m5;
                OUT(o,i*32+j*2+0, lo); OUT(o,i*32+j*2+1, hi);
            }
            break;
        }
        case VAIST_Q8_1: {
            const block_q8_1*b=(const void*)(d+base);
            float sc=f16_lo(b->ds);
            for(size_t j=0;j<32;j++) OUT(o,i*32+j, sc*(float)b->qs[j]);
            break;
        }
        case VAIST_Q2_K: {
            /* 256 values grouped 4-per-scale; 64 groups, 2-bit q in qs[64],
             * 6-bit scales+mins nibbles in scales[16], dm = half2(dall,dmin) */
            const block_q2_K*b=(const void*)(d+base);
            float dall=f16_lo(b->dm), dmin=f16_hi(b->dm);
            for(size_t g=0;g<64;g++){
                uint8_t sc=(b->scales[g]&0xF);
                uint8_t mn=((b->scales[g]>>4)&0xF);
                float d1=dall*(float)sc, m1=dmin*(float)mn;
                uint8_t qv=b->qs[g]; /* 2 bits per value, packed 4 vals in... */
                /* q2_K packs 4 values (2 bits each) into... actually 256 vals / 4 = 64 bytes */
                size_t off=i*256+g*4;
                OUT(o,off+0, d1*(float)((qv>>0)&3)-m1);
                OUT(o,off+1, d1*(float)((qv>>2)&3)-m1);
                OUT(o,off+2, d1*(float)((qv>>4)&3)-m1);
                OUT(o,off+3, d1*(float)((qv>>6)&3)-m1);
            }
            break;
        }
        case VAIST_Q3_K: {
            /* 256 values, 2-bit q in qs[64], hmask bit-per-element, scale deltas in scales[12].
             * Each thread (0..63) writes 4 consecutive elements. */
            const block_q3_K*b=(const void*)(d+base);
            float d3=f16_to_f32(b->d);
            for(size_t tid=0;tid<64;tid++){
                /* replicate CUDA: r=tid/4, is0=r%2, l0=16*is0+4*(tid%4), n=tid/8, j=tid%4 */
                size_t r=tid/4; size_t is0=r%2; size_t l0=16*is0+4*(tid%4);
                size_t ng=tid/8; size_t j=tid-4*ng; size_t is=8*ng+2*j+is0;
                size_t shift=2*j;
                uint8_t us;
                if(is<4)       us=(b->scales[is]&0xF)|(((b->scales[is+8]>>0)&3)<<4);
                else if(is<8)  us=(b->scales[is]&0xF)|(((b->scales[is+4]>>2)&3)<<4);
                else if(is<12) us=(b->scales[is-8]>>4)|(((b->scales[is]>>4)&3)<<4);
                else           us=(b->scales[is-8]>>4)|(((b->scales[is-4]>>6)&3)<<4);
                int us_s=(int)us-32;
                float dl=d3*(float)us_s;
                const uint8_t*qt=b->qs+32*ng;
                uint8_t hm=b->hmask[tid/8] & (1u<<(tid%8));
                size_t off=i*256+128*ng+32*j;
                for(size_t l=0;l<4;l++){
                    OUT(o,off+l0+l, dl*(float)((int8_t)((qt[(l0+l)>>1]>>(4*((l0+l)%2)))&3) - (hm?0:4)));
                }
            }
            break;
        }
        case VAIST_Q4_K: {
            /* 256 values, 4-bit q in qs[128], 2 scale groups per il (32 il's).
             * Each thread (0..31): il=tid/8, ir=tid%8, writes 4+4 values. */
            const block_q4_K*b=(const void*)(d+base);
            float dall=f16_lo(b->dm), dmin=f16_hi(b->dm);
            for(size_t tid=0;tid<32;tid++){
                size_t il=tid/8; size_t ir=tid%8; size_t is=2*il; size_t nn=4;
                uint8_t sc,m; get_scale_min_k4(is+0,b->scales,&sc,&m);
                float d1=dall*(float)sc, m1=dmin*(float)m;
                get_scale_min_k4(is+1,b->scales,&sc,&m);
                float d2=dall*(float)sc, m2=dmin*(float)m;
                const uint8_t*q=b->qs+32*il+nn*ir;
                for(size_t l=0;l<nn;l++){
                    OUT(o,i*256+64*il+nn*ir+l,        d1*(float)(q[l]&0xF)-m1);
                    OUT(o,i*256+64*il+nn*ir+l+32,     d2*(float)(q[l]>>4)-m2);
                }
            }
            break;
        }
        case VAIST_Q5_K: {
            /* 256 values, 4-bit q in qs[128] + 1-bit qh[32], 2 scale groups per il.
             * Each thread (0..63): il=tid/16, ir=tid%16, writes 2+2 values. */
            const block_q5_K*b=(const void*)(d+base);
            float dall=f16_lo(b->dm), dmin=f16_hi(b->dm);
            for(size_t tid=0;tid<64;tid++){
                size_t il=tid/16; size_t ir=tid%16; size_t is=2*il;
                uint8_t sc,m; get_scale_min_k4(is+0,b->scales,&sc,&m);
                float d1=dall*(float)sc, m1=dmin*(float)m;
                get_scale_min_k4(is+1,b->scales,&sc,&m);
                float d2=dall*(float)sc, m2=dmin*(float)m;
                const uint8_t*ql=b->qs+32*il+2*ir;
                const uint8_t*qh=b->qh+2*ir;
                uint8_t hm=1u<<(2*il);
                OUT(o,i*256+64*il+2*ir+0,  d1*(float)((ql[0]&0xF)+(qh[0]&hm?16:0))-m1);
                OUT(o,i*256+64*il+2*ir+1,  d1*(float)((ql[1]&0xF)+(qh[1]&hm?16:0))-m1);
                hm<<=1;
                OUT(o,i*256+64*il+2*ir+32, d2*(float)((ql[0]>>4)+(qh[0]&hm?16:0))-m2);
                OUT(o,i*256+64*il+2*ir+33, d2*(float)((ql[1]>>4)+(qh[1]&hm?16:0))-m2);
            }
            break;
        }
        case VAIST_Q6_K: {
            /* 256 values, 4-bit ql[128] + 2-bit qh[64], 16 int8 scales.
             * ip=0..1, il=0..31: 4 output rows per (ip,il). */
            const block_q6_K*b=(const void*)(d+base);
            float d=f16_to_f32(b->d);
            for(size_t ip=0;ip<2;ip++)for(size_t il=0;il<32;il++){
                size_t is=8*ip+il/16;
                const uint8_t*ql=b->ql+64*ip+il;
                uint8_t qh=b->qh[32*ip+il];
                const int8_t*sc=b->scales+is;
                size_t off=i*256+128*ip+il;
                OUT(o,off+0,  d*(float)(sc[0]*((int8_t)((ql[0]&0xF)|(((qh>>0)&3)<<4))-32)));
                OUT(o,off+32, d*(float)(sc[2]*((int8_t)((ql[32]&0xF)|(((qh>>2)&3)<<4))-32)));
                OUT(o,off+64, d*(float)(sc[4]*((int8_t)((ql[0]>>4)|(((qh>>4)&3)<<4))-32)));
                OUT(o,off+96, d*(float)(sc[6]*((int8_t)((ql[32]>>4)|(((qh>>6)&3)<<4))-32)));
            }
            break;
        }
        case VAIST_IQ4_NL: {
            /* QK4_NL=32, 8 sub-blocks of 32 per 256-value output block */
            for(size_t bl=0; bl<8; bl++){
                const block_iq4_nl*bb=(const block_iq4_nl*)(d+base+bl*sizeof(block_iq4_nl));
                float d_iq=f16_to_f32(bb->d);
                size_t boff=i*256+bl*32;
                for(size_t ib=0;ib<8;ib++){
                    const uint8_t* q4=bb->qs+4*ib;
                    size_t off=boff+32*ib+4*ib; /* matches CUDA yy+...+4*il where il=ib */
                    for(size_t j=0;j<4;j++){
                        OUT(o,off+j,     d_iq*(float)kvalues_iq4nl[q4[j]&0xf]);
                        OUT(o,off+16+j,  d_iq*(float)kvalues_iq4nl[q4[j]>>4]);
                    }
                }
            }
            break;
        }
        case VAIST_IQ2_XXS: {
            const block_iq2_xxs*b=(const void*)(d+base);
            for(size_t tid=0;tid<32;tid++){
                size_t il=tid/8; size_t ib=tid%8; size_t off=i*256+32*ib+8*il;
                const uint16_t* q2=b->qs+4*ib;
                const uint8_t* aux8=(const uint8_t*)q2;
                uint8_t gidx=aux8[il];                       /* byte il of q2[0..1] */
                const uint8_t* grid=(const uint8_t*)(&iq2xxs_grid[gidx]);
                uint32_t aux32=((uint32_t)q2[2])|(((uint32_t)q2[3])<<16);
                float scale=f16_to_f32(b->d)*(0.5f+(float)(aux32>>28)*0.25f);
                uint8_t signs=ksigns_iq2xs[(aux32>>(7*il))&127];
                for(size_t j=0;j<8;j++)
                    OUT(o,off+j, scale*(float)grid[j]*(signs&kmask_iq2xs[j]?-1.f:1.f));
            }
            break;
        }
        case VAIST_IQ2_XS: {
            const block_iq2_xs*b=(const void*)(d+base);
            for(size_t tid=0;tid<32;tid++){
                size_t il=tid/8; size_t ib=tid%8; size_t off=i*256+32*ib+8*il;
                const uint16_t* q2=b->qs+4*ib;
                uint16_t idx=(uint16_t)(q2[il]&511);
                const uint8_t* grid=(const uint8_t*)(&iq2xs_grid[idx]);
                float scale=f16_to_f32(b->d)*(0.5f+(float)((b->scales[ib]>>(4*(il/2)))&0xf))*0.25f;
                uint8_t signs=ksigns_iq2xs[q2[il]>>9];
                for(size_t j=0;j<8;j++)
                    OUT(o,off+j, scale*(float)grid[j]*(signs&kmask_iq2xs[j]?-1.f:1.f));
            }
            break;
        }
        case VAIST_IQ3_XXS: {
            const block_iq3_xxs*b=(const void*)(d+base);
            for(size_t tid=0;tid<32;tid++){
                size_t il=tid/8; size_t ib=tid%8; size_t off=i*256+32*ib+8*il;
                const uint8_t* q3=b->qs+8*ib;
                const uint16_t* gas=(const uint16_t*)(b->qs+QK_K/4)+2*ib;
                uint32_t aux32=((uint32_t)gas[0])|(((uint32_t)gas[1])<<16);
                const uint32_t* iq3g=(const uint32_t*)iq3xxs_grid; /* read as uint32 to index by byte */
                uint32_t g1v=iq3g[q3[2*il+0]&0xff];
                uint32_t g2v=iq3g[q3[2*il+1]&0xff];
                const uint8_t* g1=(const uint8_t*)&g1v;
                const uint8_t* g2=(const uint8_t*)&g2v;
                float scale=f16_to_f32(b->d)*(0.5f+(float)(aux32>>28))*0.5f;
                uint8_t signs=ksigns_iq2xs[(aux32>>(7*il))&127];
                for(size_t j=0;j<4;j++){
                    OUT(o,off+0+j, scale*(float)g1[j]*(signs&kmask_iq2xs[j]?-1.f:1.f));
                    OUT(o,off+4+j, scale*(float)g2[j]*(signs&kmask_iq2xs[j+4]?-1.f:1.f));
                }
            }
            break;
        }
        case VAIST_IQ1_S: {
            const block_iq1_s*b=(const void*)(d+base);
            for(size_t tid=0;tid<32;tid++){
                size_t il=tid/8; size_t ib=tid%8; size_t off=i*256+32*ib+8*il;
                float scale=f16_to_f32(b->d)*(2.0f*(float)((b->qh[ib]>>12)&7)+1.0f);
                float delta=(b->qh[ib]&0x8000)?-1.0f-IQ1S_DELTA:-1.0f+IQ1S_DELTA;
                uint32_t g=iq1s_grid_gpu[b->qs[4*ib+il]|(((b->qh[ib]>>(3*il))&7)<<8)];
                /* CUDA: grid32[0]=g; grid32[1]=(g>>4)&0x0f0f0f0f; then cast to int8* so
                 * q[j] picks nibble j (unsigned) and the arithmetic is q[j]+delta. */
                uint8_t nib[8];
                nib[0]=(uint8_t)(g&0xf);      nib[1]=(uint8_t)((g>>4)&0xf);
                nib[2]=(uint8_t)((g>>8)&0xf);  nib[3]=(uint8_t)((g>>12)&0xf);
                nib[4]=(uint8_t)((g>>16)&0xf); nib[5]=(uint8_t)((g>>20)&0xf);
                nib[6]=(uint8_t)((g>>24)&0xf); nib[7]=(uint8_t)((g>>28)&0xf);
                for(size_t j=0;j<8;j++) OUT(o,off+j, scale*((float)nib[j]+delta));
            }
            break;
        }
        case VAIST_IQ1_M: {
            const block_iq1_m*b=(const void*)(d+base);
            for(size_t tid=0;tid<32;tid++){
                size_t il=tid/8; size_t ib=tid%8; size_t off=i*256+32*ib+8*il;
                /* Reconstruct 4-bit fp16 scale from 4 uint16 scale words (ggml IQ1M layout) */
                const uint16_t* sc=(const uint16_t*)b->scales;
                uint16_t scale_u16=(uint16_t)((sc[0]>>12)|((sc[1]>>8)&0xf0)|((sc[2]>>4)&0xf00)|((sc[3])&0xf000));
                size_t ib16=2*ib+il/2;
                float scale=f16_to_f32(scale_u16)*(2.0f*(float)((sc[ib16/4]>>(3*(ib16%4)))&0x7)+1.0f);
                float delta=(b->qh[2*ib+il/2]&(0x08u<<(4*(il%2))))?-1.0f-IQ1M_DELTA:-1.0f+IQ1M_DELTA;
                uint32_t g=iq1s_grid_gpu[b->qs[4*ib+il]|(((b->qh[2*ib+il/2]>>(4*(il%2)))&7)<<8)];
                uint8_t nib[8];
                nib[0]=(uint8_t)(g&0xf);      nib[1]=(uint8_t)((g>>4)&0xf);
                nib[2]=(uint8_t)((g>>8)&0xf);  nib[3]=(uint8_t)((g>>12)&0xf);
                nib[4]=(uint8_t)((g>>16)&0xf); nib[5]=(uint8_t)((g>>20)&0xf);
                nib[6]=(uint8_t)((g>>24)&0xf); nib[7]=(uint8_t)((g>>28)&0xf);
                for(size_t j=0;j<8;j++) OUT(o,off+j, scale*((float)nib[j]+delta));
            }
            break;
        }
        case VAIST_IQ2_S:
        case VAIST_IQ3_S:
        case VAIST_IQ3_XS:
        case VAIST_IQ4_XS:
            /* iq2_s / iq3_s / iq4_nl / iq4_xs dequant require additional scale
             * decode beyond the shipped grids; deferred to next pass. */
            return VAIST_UNSUPPORTED;
        default:
            return VAIST_INVALID_ARGUMENT;
        }
    }
    return VAIST_OK;
}
