/* vaist_blas_vulkan_test: validates the vaist_blas Vulkan dispatch path.
 *
 * Runs on any box that builds VAiSt. With a real Vulkan device present the
 * vaist_blas entry points dispatch the shaders/vkblas/* kernels; without one
 * they transparently fall back to the scalar CPU kernels (weak-box friendly).
 * In both cases the result is compared to an independent scalar reference and
 * must match. Exit 0 == pass. */
#include "vaist_blas.h"
#include "vaist_runtime.h"
#include "vaist_core.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(_WIN32)
#include <windows.h>
#endif

static int approxf(float a,float b){ return fabsf(a-b) < 1e-2f; }

int main(int argc, char **argv){
#if defined(_WIN32)
    /* Probe-only entry: child of vaist_vk_device_probe_child(). The parent passes
     * the flag as a command-line argument (argv[1]=="VAIST_VK_PROBE=1") because
     * SetEnvironmentVariableA does not propagate to CreateProcessW's inherited
     * env block on Windows. Attempt a real vkCreateDevice and signal viability via
     * exit code (no BLAS work, no diagnostics). Exit 0 => a logical device was
     * created. */
    if(argc>=2 && strcmp(argv[1],"VAIST_VK_PROBE=1")==0){
        int ok=0;
        VaistRuntime *rt=NULL;
        if(vaist_runtime_create(VAIST_BACKEND_VULKAN,&rt)==VAIST_OK){
            void *dev=NULL,*q=NULL; uint32_t qf=0;
            ok = (vaist_runtime_vk_state(rt,&dev,&q,&qf)==VAIST_OK) ? 1:0;
            vaist_runtime_destroy(rt);
        }
        return ok?0:1;
    }
#endif
    int ok=1;
    VaistRuntime *rt=NULL;
    VaistRuntimeInfo info;
    int gpu_used=0;

    if(vaist_runtime_create(VAIST_BACKEND_AUTO,&rt)!=VAIST_OK)
        if(vaist_runtime_create(VAIST_BACKEND_CPU,&rt)!=VAIST_OK){
            fprintf(stderr,"vaist_blas_vulkan_test: runtime create fail\n");
            return 1;
        }
    vaist_runtime_info(rt,&info);
    printf("vaist_blas_vulkan_test: backend=%u vulkan_available=%u\n",
           (unsigned)info.backend, (unsigned)info.vulkan_available);
    fflush(stdout);

    /* ---- 32x32x32 tiled GEMM: C += A*B (start C=0 => C = A*B) ---- */
    {
        size_t m=32,n=32,k=32;
        float *A=calloc(m*k,sizeof(float)),*B=calloc(k*n,sizeof(float));
        float *C=calloc(m*n,sizeof(float)),*ref=calloc(m*n,sizeof(float));
        if(!A||!B||!C||!ref){ ok=0; goto done_gemm; }
        for(size_t i=0;i<m*k;i++) A[i]=(float)(int)((i%9)-4);
        for(size_t i=0;i<k*n;i++) B[i]=(float)(int)((i%7)-3);
        if(vaist_blas_gemm(rt,A,B,C,m,k,n,NULL,VAIST_W_FP16)!=VAIST_OK){ ok=0; }
        for(size_t i=0;i<m;i++)for(size_t j=0;j<n;j++){
            float s=0.0f; for(size_t p=0;p<k;p++) s+=A[i*k+p]*B[p*n+j]; ref[i*n+j]=s;
        }
        for(size_t i=0;i<m*n;i++){
            if(!approxf(C[i],ref[i])){ ok=0; fprintf(stderr,"gemm mismatch @%zu: %f exp %f\n",i,C[i],ref[i]); }
        }
    done_gemm:
        free(A);free(B);free(C);free(ref);
    }

    /* ---- 64x32 ternary matvec (k=64, n=32) ---- */
    {
        size_t k=64,n=32;
        int8_t *w=calloc(k*n,1);
        float *scale=calloc(n,sizeof(float)),*x=calloc(k,sizeof(float));
        float *y=calloc(n,sizeof(float)),*ref=calloc(n,sizeof(float));
        if(!w||!scale||!x||!y||!ref){ ok=0; goto done_ter; }
        for(size_t i=0;i<k*n;i++) w[i]=(int8_t)((int)((i%3)-1)); /* {-1,0,1} */
        for(size_t i=0;i<k;i++) x[i]=(float)(int)((i%5)-2);
        for(size_t j=0;j<n;j++) scale[j]=1.0f+0.1f*(float)j;
        if(vaist_blas_matvec_ternary(rt,w,scale,k,n,x,y)!=VAIST_OK){ ok=0; }
        for(size_t j=0;j<n;j++){
            float s=0.0f; for(size_t i=0;i<k;i++) s+=(float)w[j*k+i]*x[i];
            ref[j]=scale[j]*s;
        }
        for(size_t j=0;j<n;j++){
            if(!approxf(y[j],ref[j])){ ok=0; fprintf(stderr,"ternary mismatch @%zu: %f exp %f\n",j,y[j],ref[j]); }
        }
    done_ter:
        free(w);free(scale);free(x);free(y);free(ref);
    }

    /* ---- 64x32 binary matvec (k=64 mult of 8, n=32) ---- */
    {
        size_t k=64,n=32,k8=k/8;
        uint8_t *w=calloc(k8*n,1);
        int8_t *sign=calloc(k*n,1);
        float *scale=calloc(n,sizeof(float)),*x=calloc(k,sizeof(float));
        float *y=calloc(n,sizeof(float)),*ref=calloc(n,sizeof(float));
        if(!w||!sign||!scale||!x||!y||!ref){ ok=0; goto done_bin; }
        for(size_t i=0;i<k*n;i++) sign[i]=(int8_t)((int)((i%3)-1));
        for(size_t j=0;j<n;j++)for(size_t i=0;i<k;i++)
            w[j*k8+i/8] |= (uint8_t)((sign[j*k+i]>=0?1:0) << (7u-(i&7u)));
        for(size_t i=0;i<k;i++) x[i]=(float)(int)((i%5)-2);
        for(size_t j=0;j<n;j++) scale[j]=1.0f+0.1f*(float)j;
        if(vaist_blas_matvec_binary(rt,w,scale,k,n,x,y)!=VAIST_OK){ ok=0; }
        /* reference: y[j]=scale[j]*sum_i (2*bit-1 with bit=signbit(x[i])) */
        for(size_t j=0;j<n;j++){
            int acc=0;
            for(size_t i=0;i<k;i++){ int off=(int)(j*k+i);
                int xbit = x[i]>=0?1:0; acc += (sign[off]>=0?1:0)==xbit ? 1 : -1; }
            ref[j]=scale[j]*(float)acc;
        }
        for(size_t j=0;j<n;j++){
            if(!approxf(y[j],ref[j])){ ok=0; fprintf(stderr,"binary mismatch @%zu: %f exp %f\n",j,y[j],ref[j]); }
        }
    done_bin:
        free(w);free(sign);free(scale);free(x);free(y);free(ref);
    }

    /* Detect whether any GPU dispatch was actually used (informational). */
    if(info.backend==VAIST_BACKEND_VULKAN){
        VaistDeviceCaps dc;
        gpu_used = (vaist_runtime_device_caps(rt,&dc)==VAIST_OK && dc.vulkan_api_version!=0u);
    }
    printf("vaist_blas_vulkan_test: %s (backend=%u, %s)\n",
           ok?"PASS":"FAIL", (unsigned)info.backend, gpu_used?"gpu-dispatch":"cpu-fallback");
    vaist_runtime_destroy(rt);
    return ok?0:1;
}
