/* Standalone CPU self-test for vaist_quant dequant/quant (no Vulkan). */
#include "vaist_quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
static int fail=0;
#define CK(c,...) do{ if(!(c)){ fprintf(stderr,"FAIL: " __VA_ARGS__); fprintf(stderr,"\n"); fail=1; } }while(0)

int main(void){
    /* ---- q4_0 round trip: [1..8, -1..-8] ---- */
    float in[32]; int i;
    for(i=0;i<16;i++){ in[i]= (float)(i+1); in[16+i]= (float)(-(i+1)); }
    unsigned char buf[256]; size_t used=0;
    CK(vaist_quantize_f32(VAIST_Q4_0, in, 32, buf, sizeof(buf), &used)==VAIST_OK, "q4_0 quantize");
    float out[32]={0};
    CK(vaist_dequantize_f32(VAIST_Q4_0, buf, used, out, 32)==VAIST_OK, "q4_0 dequantize");
    float maxerr=0; for(i=0;i<32;i++){ float e=fabsf(in[i]-out[i]); if(e>maxerr)maxerr=e; }
    CK(maxerr<1.5f, "q4_0 round-trip maxerr=%g (expect <1.5)", maxerr);

    /* ---- q8_0 round trip ---- */
    float v[32]; for(i=0;i<32;i++) v[i]=(float)(i*7-100);
    CK(vaist_quantize_f32(VAIST_Q8_0, v, 32, buf, sizeof(buf), &used)==VAIST_OK, "q8_0 quantize");
    memset(out,0,sizeof(out));
    CK(vaist_dequantize_f32(VAIST_Q8_0, buf, used, out, 32)==VAIST_OK, "q8_0 dequantize");
    maxerr=0; for(i=0;i<32;i++){ float e=fabsf(v[i]-out[i]); if(e>maxerr)maxerr=e; }
    CK(maxerr<0.5f, "q8_0 round-trip maxerr=%g (expect <0.5)", maxerr);

    /* ---- q8_1 round trip ---- */
    CK(vaist_quantize_f32(VAIST_Q8_1, v, 32, buf, sizeof(buf), &used)==VAIST_OK, "q8_1 quantize");
    memset(out,0,sizeof(out));
    CK(vaist_dequantize_f32(VAIST_Q8_1, buf, used, out, 32)==VAIST_OK, "q8_1 dequantize");
    maxerr=0; for(i=0;i<32;i++){ float e=fabsf(v[i]-out[i]); if(e>maxerr)maxerr=e; }
    CK(maxerr<0.5f, "q8_1 round-trip maxerr=%g", maxerr);

    /* ---- block sizes (GGUF byte-exact, pack(1) llama.cpp layout) ---- */
    CK(vaist_quant_block_bytes(VAIST_Q8_0)==34, "q8_0 size=%zu", vaist_quant_block_bytes(VAIST_Q8_0));
    CK(vaist_quant_block_bytes(VAIST_Q4_0)==18, "q4_0 size=%zu", vaist_quant_block_bytes(VAIST_Q4_0));
    CK(vaist_quant_block_bytes(VAIST_Q4_1)==20, "q4_1 size=%zu", vaist_quant_block_bytes(VAIST_Q4_1));

    /* ---- K-quant dequant: hand-crafted vectors ---- */
    /* Q4_K: one 256-elem block. d=1.0, min=1.0 via dm half2.
     * get_scale_min_k4(0): sc=scales[0]&63=1, m=scales[4]&63=1.
     * out = d*sc*(qs&0xf) - dmin*m = 1*1*qs_lo - 1*1. qs=0x12 -> lo=2 -> out=1; hi=1 -> out=0. */
    {
        vaist_block_q4_K blk; memset(&blk,0,sizeof blk);
        /* dm = half2(1.0f, 1.0f), all 6 scale-pairs = (sc=1, min=1) */
        blk.dm=((uint32_t)0x3c00)|((uint32_t)0x3c00<<16);
        for(i=0;i<12;i++) blk.scales[i]=1;  /* sc and min both 1 */
        for(i=0;i<128;i++) blk.qs[i]=0x12;  /* lo=2 -> out=1.0*2-1=1.0; hi=1 -> out=1*1-1=0.0 */
        float ko[256]; memset(ko,0xff,sizeof ko);
        VaistStatus st=vaist_dequantize_f32(VAIST_Q4_K, &blk, sizeof blk, ko, 256);
        CK(st==VAIST_OK, "q4_K dequant status=%d", st);
        /* il=0, ir=0..7: 4 vals each, all lo -> ko[0..31]=1.0 */
        int ok=1;
        for(i=0;i<32;i++){ if(fabsf(ko[i]-1.0f)>0.01f){ ok=0; break; } }
        CK(ok, "q4_K dequant lo=%.2f (want 1.0)", ko[0]);
        /* hi path: ko[32] = d2*(1)-m2 = 1*1-1 = 0.0 */
        for(i=0;i<32;i++){ if(fabsf(ko[32+i])>0.01f){ ok=0; break; } }
        CK(ok, "q4_K dequant hi=%.2f (want 0.0)", ko[32]);
    }
     /* Q6_K structural: d=1.0, scales[0]=4, ql=0x21, qh=0.
      * ip=0,il=0: sc[0]=4; lo=(ql[0]&0xf | 0)-32 = 1-32=-31; out = d*sc*(-31) = 1*4*(-31) = -124. */
    {
        vaist_block_q6_K blk; memset(&blk,0,sizeof blk);
        blk.d=0x3c00;
        for(i=0;i<64;i++) blk.ql[i]=0x21;
        for(i=0;i<16;i++) blk.scales[i]=4;
        float ko[256]; memset(ko,0xff,sizeof ko);
        VaistStatus st=vaist_dequantize_f32(VAIST_Q6_K, &blk, sizeof blk, ko, 256);
        CK(st==VAIST_OK, "q6_K dequant status=%d", st);
        CK(fabsf(ko[0]-(-124.0f))<0.5f, "q6_K dequant[0]=%.2f (want -124)", ko[0]);
    }
    /* Q2_K structural: dm=half2(2.0,1.0). qs encodes 2-bit values (4 vals/byte).
      * scales[0]=0x11 -> sc_lo=1, mn_hi=1. d1=2.0*1=2.0, m1=1.0*1=1.0.
      * qs[0]=0x55 -> bits: (01,01,01,01). idx values all = 1.
      * OUT = d1*(idx) - m1 = 2.0*1 - 1.0 = 1.0. */
    {
        vaist_block_q2_K blk; memset(&blk,0,sizeof blk);
        /* dm = half2(2.0, 1.0); 2.0 = 0x4000, 1.0 = 0x3c00 */
        blk.dm=((uint32_t)0x4000)|((uint32_t)0x3c00<<16);
        /* scales[0] = 0x11: sc_lo=1, mn_hi=1 */
        blk.scales[0]=0x11;
        /* qs[0]=0x55 -> 4 values each = idx 1 (bits 01) */
        blk.qs[0]=0x55;
        float ko[256]; memset(ko,0xff,sizeof ko);
        VaistStatus st=vaist_dequantize_f32(VAIST_Q2_K, &blk, sizeof blk, ko, 256);
        CK(st==VAIST_OK, "q2_K dequant status=%d", st);
        /* ko[0..3] = 2.0*1 - 1.0 = 1.0 */
        CK(fabsf(ko[0]-1.0f)<0.01f, "q2_K dequant[0]=%.2f (want 1.0)", ko[0]);
        CK(fabsf(ko[3]-1.0f)<0.01f, "q2_K dequant[3]=%.2f (want 1.0)", ko[3]);
        /* q2_K block size matches GGUF layout */
        /* q2_K block size = 16 (scales) + 64 (qs) + 4 (half2 dm) = 84 */
        CK(vaist_quant_block_bytes(VAIST_Q2_K)==84, "q2_K size=%zu", vaist_quant_block_bytes(VAIST_Q2_K));
    }

    /* ---- fp16 codec sanity ---- */
    /* Can't reach f16_to_f32 from header; rely on q4_0 round trip which exercises it. */
    printf("vaist_quant self-test: %s\n", fail?"FAIL":"PASS");
    return fail;
}
