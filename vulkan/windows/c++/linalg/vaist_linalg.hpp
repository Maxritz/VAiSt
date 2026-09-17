#include "vaist_linalg.h"
namespace vaist {
inline VaistStatus potrf(VaistRuntime*rt,float*A,int n,int lda,int uplo,int*info){ return vaist_linalg_spotrf(rt,A,n,lda,uplo,info); }
inline VaistStatus potri(VaistRuntime*rt,float*A,int n,int lda,int uplo,int*info){ return vaist_linalg_spotri(rt,A,n,lda,uplo,info); }
inline VaistStatus potrs(VaistRuntime*rt,const float*A,float*B,int n,int nrhs,int lda,int ldb,int uplo){ return vaist_linalg_spotrs(rt,A,B,n,nrhs,lda,ldb,uplo); }
inline VaistStatus getrf(VaistRuntime*rt,float*A,float*X,int*ipiv,int m,int n,int lda,int*info){ return vaist_linalg_sgetrf(rt,A,X,ipiv,m,n,lda,info); }
inline VaistStatus getrs(VaistRuntime*rt,const float*A,float*X,int*ipiv,int n,int nrhs,int lda,int ldb,int trans){ return vaist_linalg_sgetrs(rt,A,X,ipiv,n,nrhs,lda,ldb,trans); }
inline VaistStatus geqrf(VaistRuntime*rt,float*A,float*tau,int m,int n,int lda,int*info){ return vaist_linalg_sgeqrf(rt,A,tau,m,n,lda,info); }
inline VaistStatus orgqr(VaistRuntime*rt,float*A,int m,int n,int k,int lda,int*tau,int*info){ return vaist_linalg_sorgqr(rt,A,m,n,k,lda,tau,info); }
}
