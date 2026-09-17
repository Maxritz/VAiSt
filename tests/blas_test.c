/* blas_test: MatMul-free compute correctness vs scalar reference.
 * Runs on any CPU — no Vulkan required (weak-box friendly).
 * Validates ternary (2406.02528/2608.03142) and 1-bit xnor (2608.01528). */
#include "vaist_blas.h"
#include "vaist_core.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#define K 32
#define N 4

static int approx(float a,float b){ return fabsf(a-b) < 1e-3f; }

int main(void){
    int ok=1;
    VaistRuntime *rt=NULL;
    if(vaist_runtime_create(VAIST_BACKEND_CPU,&rt)!=VAIST_OK){ fprintf(stderr,"rt create fail\n"); return 1; }

    float x[K];
    int8_t sign[K*N];        /* ternary weights {-1,0,+1}, row-major: n rows x k */
    float scale[N];
    for(int i=0;i<K;i++) x[i]=((i*7)%5)-2;          /* -2..+2 */
    for(int j=0;j<N;j++){
        float s=0; for(int i=0;i<K;i++){ int8_t v=((i+j)%3)-1; sign[j*K+i]=v; s+=v*x[i]; }
        scale[j]=1.0f; (void)s;
    }

    /* dense ground truth: y[j] = sum_i sign[j,k+i]*x[i]  (scale==1) */
    float yref[N], yt[N], yb[N];
    for(int j=0;j<N;j++){ float s=0; for(int i=0;i<K;i++) s+= (float)sign[j*K+i]*x[i]; yref[j]=s; }

    /* ternary matvec */
    if(vaist_blas_matvec_ternary(rt,sign,scale,K,N,x,yt)!=VAIST_OK){ fprintf(stderr,"ternary call fail\n"); ok=0; }
    for(int j=0;j<N;j++) if(!approx(yt[j],yref[j])) ok=0;

    /* binary pack: bit = sign>=0 ? 1 : 0  → xnor parity vs x sign */
    uint8_t bits[(K*N+7)/8]; memset(bits,0,sizeof bits);
    for(int j=0;j<N;j++)for(int i=0;i<K;i++){ int off=j*K+i; bits[off/8] |= (uint8_t)(((sign[off]>=0)?1:0) << (7-(off%8))); }
    float yexp[N];
    for(int j=0;j<N;j++){ int acc=0; for(int i=0;i<K;i++){
        int off=j*K+i; int wbit=(bits[off/8]>>(7-(off%8)))&1; int xbit=x[i]>=0?1:0;
        acc += (wbit==xbit)?1:-1; } yexp[j]=(float)acc*scale[j]; }
    if(vaist_blas_matvec_binary(rt,bits,scale,K,N,x,yb)!=VAIST_OK){ fprintf(stderr,"binary call fail\n"); ok=0; }
    for(int j=0;j<N;j++) if(!approx(yb[j],yexp[j])) ok=0;

    /* path selector: ternary/binary weight => MATMUL_FREE */
    VaistComputePath p=vaist_blas_best_path(rt,K,K,N,VAIST_W_TERNARY_I8);
    if(p!=VAIST_PATH_MATMUL_FREE){ fprintf(stderr,"path ternary wrong: %d\n",(int)p); ok=0; }
    p=vaist_blas_best_path(rt,K,K,N,VAIST_W_BINARY_I1);
    if(p!=VAIST_PATH_MATMUL_FREE){ fprintf(stderr,"path binary wrong: %d\n",(int)p); ok=0; }

    vaist_runtime_destroy(rt);
    printf("blas_test: %s\n", ok?"PASS":"FAIL");
    return ok?0:1;
}
