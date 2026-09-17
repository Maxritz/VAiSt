/* Debug: mul_mat_q round-trip */
#include "vaist_blas.h"
#include "vaist_quant.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

int main(void){
    /* Small: k=32, n=32, m=2. Weight = q8_0 quantization of identity-ish matrix. */
    size_t k=32, n=32, m=2;
    float A[2*32]; /* m×k row-major */
    for(int i=0;i<2*32;i++) A[i]=(float)(i%3+1); /* small values */
    /* Build a q8_0 weight matrix (k rows × n cols), row-major, each row is 32 ints */
    float W[32*32]; memset(W,0,sizeof(W));
    for(int r=0;r<32;r++) for(int c=0;c<32;c++) W[r*32+c]=(float)((r+c)%7-3);
    /* quantize each row to q8_0 */
    unsigned char qbuf[32*256]; size_t used=0;
    for(int r=0;r<32;r++){
        vaist_quantize_f32(VAIST_Q8_0, &W[r*32], 32, qbuf+r*34, 256, &used);
    }
    float C[2*32]; memset(C,0,sizeof(C));
    /* Expected: C = A * W (m×k × k×n = m×n), row-major */
    float exp[2*32];
    for(int i=0;i<2;i++) for(int j=0;j<32;j++){
        float s=0; for(int p=0;p<32;p++) s+=A[i*32+p]*W[p*32+j];
        exp[i*32+j]=s;
    }
    VaistStatus st=vaist_blas_mul_mat_q(NULL, A, C, m, k, n, qbuf, VAIST_Q8_0);
    printf("mul_mat_q status=%d\n", (int)st);
    float maxerr=0;
    for(int i=0;i<2*32;i++){ float e=fabsf(C[i]-exp[i]); if(e>maxerr)maxerr=e; }
    printf("maxerr=%.6f (dequant+fused scalar)\n", maxerr);
    if(maxerr<0.05f) printf("mul_mat_q: PASS\n");
    else printf("mul_mat_q: FAIL\n");
    return maxerr<0.05f?0:1;
}
