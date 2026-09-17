#include "vaist_linalg.hpp"
int main(){
    /* 4x4 SPD matrix: A = L L^T with L = [2,0,0,0;1,3,0,0;0,1,4,0;0,0,1,5] */
    float A[16]={4,0,0,0, 1,9,0,0, 0,1,16,0, 0,0,1,25};  /* row-major lower used by chol tile */
    int info=0;
    if(vaist_linalg_spotrf(0,A,4,4,VAIST_UPLO_LOWER,&info)!=VAIST_OK||info!=0) return 1;
    /* recovered R diagonal should be [2,3,4,5] */
    if(!(A[0+0*4]>1.99f&&A[0+0*4]<2.01f)) return 2;
    if(!(A[1+1*4]>2.99f&&A[1+1*4]<3.01f)) return 3;
    if(!(A[2+2*4]>3.99f&&A[2+2*4]<4.01f)) return 4;
    if(!(A[3+3*4]>4.99f&&A[3+3*4]<5.01f)) return 5;
    return 0;
}
