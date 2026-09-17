#include "vaist_linalg.h"
#include <stdio.h>
#include <math.h>
static int approxf(float a,float b){ return fabsf(a-b)<1e-3f; }
int main(void){
    int ok=1, info=0;
    /* SPD: A = LL^T, L lower-tri = [2 0 0; 1 3 0; 0 1 4] -> A = [4,2,0;2,10,3;0,3,17] row-major */
    float A[9]={4,2,0, 2,10,3, 0,3,17};
    if(vaist_linalg_spotrf(0,A,3,3,VAIST_UPLO_UPPER,&info)!=VAIST_OK||info!=0){ fprintf(stderr,"chol fail\n"); ok=0; }
    /* check recovered diagonal (UPPER branch writes to col-major diag A[i,i]) */
    if(!approxf(A[0+0*3],2.0f)||!approxf(A[1+1*3],3.0f)||!approxf(A[2+2*3],4.0f)){ fprintf(stderr,"chol diag wrong\n"); ok=0; }

    /* LU solve: A x = b, A=[7,2,1;2,6,3;1,3,7] (row-major), b=[1,2,3] */
    float Af[9]={7,2,1, 2,6,3, 1,3,7}, LU[9], rhs[9]={1,2,3, 0,0,0, 0,0,0};
    int ipiv[3];
    if(vaist_linalg_sgetrf(0,Af,LU,ipiv,3,3,3,&info)!=VAIST_OK||info!=0){ fprintf(stderr,"lu fail\n"); ok=0; }
    if(vaist_linalg_sgetrs(0,LU,rhs,ipiv,3,1,3,3,0)!=VAIST_OK){ fprintf(stderr,"getrs fail\n"); ok=0; }
    /* expected solution of A x = [1,2,3]: x = [0.0526, 0.1340, 0.3636] */
    float ref[3]={0.0526316f, 0.1339709f, 0.3636364f};
    for(int i=0;i<3;i++) if(!approxf(rhs[i],ref[i])){ fprintf(stderr,"lu solve wrong[%d]=%f\n",i,rhs[i]); ok=0; }

    /* syevd: 3x3 symmetric -> eigenvalues of diag(1,2,3) shuffled */
    float S[9]={2,1,0, 1,3,1, 0,1,4}, W[3];
    if(vaist_linalg_ssyevd(0,VAIST_JOBS_NONE,VAIST_UPLO_LOWER,S,3,3,W,0,0,&info)!=VAIST_OK||info!=0){ fprintf(stderr,"syevd fail\n"); ok=0; }
    /* eigenvalues of [[2,1,0],[1,3,1],[0,1,4]] ascending */
    if(!approxf(W[0],1.2679f)||!approxf(W[1],3.0f)||!approxf(W[2],4.7321f)){ fprintf(stderr,"syevd eig wrong [%.4f,%.4f,%.4f]\n",W[0],W[1],W[2]); ok=0; }

    printf("linalg_test: %s\n", ok?"PASS":"FAIL");
    return ok?0:1;
}
