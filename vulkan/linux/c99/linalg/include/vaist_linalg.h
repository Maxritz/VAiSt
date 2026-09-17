#ifndef VAIST_LINALG_H
#define VAIST_LINALG_H
#include "vaist_core.h"
#include "vaist_runtime.h"
#include "vaist_blas.h"
#ifdef __cplusplus
extern "C" {
#endif

/*
 * vaist_linalg: cuSOLVER-compatible dense linear algebra surface, CPU-first.
 * Covers the 131-entry cuSOLVER proxy mapping (LU / Cholesky / QR / SVD / eig):
 *   A) LU:       getrf / getrs / getrfBatched / getrsBatched
 *   B) Cholesky: potrf / potrfBatched / potri / potrs / potrsBatched / potri_bufferSize
 *   C) QR:       geqrf / geqrfBatched / orgqr / ormqr
 *   D) SVD:      gesvd / gesvdBatched / geqp3
 *   E) Eig:      syevd / syevj / heevj / syevdBatched / syevjBatched
 *   F) Generic-X: potrf / geqrf / syevd / gesvdj / getrf (datatype-tagged variants)
 *
 * Dense algorithms are row-major here (vaist convention) but the LAPACK
 * workspace + leading-dimension contract matches cuSOLVER: column-major,
 * lda/ldb/llt stride in elements. The CPU fallback below transposes
 * internally only where cheaper than a separate kernel. This keeps one
 * documented contract with the cuSOLVER proxy.
 *
 * GPU tiles (shaders/vklinalg/): lu16 / chol16 / qr64 compile but the C
 * dispatch returns VAIST_NOT_IMPLEMENTED on the VULKAN path until the
 * tile dispatch is wired (depends on vaist_blas GPU dispatch). The CPU
 * path is fully functional and weak-box friendly.
 */

typedef enum VaistLinalgJobz {
    VAIST_JOBS_NONE = 0,   /* no vectors (cuSOLVER 'N') */
    VAIST_JOBS_VEC  = 1,   /* compute vectors (cuSOLVER 'V') */
} VaistLinalgJobz;

typedef enum VaistLinalgUplo {
    VAIST_UPLO_LOWER = 0,   /* cuSOLVER 'L' */
    VAIST_UPLO_UPPER = 1,   /* cuSOLVER 'U' */
} VaistLinalgUplo;

typedef enum VaistLinalgSide {
    VAIST_SIDE_LEFT  = 0,   /* cuSOLVER 'L' */
    VAIST_SIDE_RIGHT = 1,   /* cuSOLVER 'R' */
} VaistLinalgSide;

typedef enum VaistLinalgOp {
    VAIST_OP_NON_TRANS   = 0,  /* cuSOLVER 'N' */
    VAIST_OP_TRANS       = 1,  /* cuSOLVER 'T' */
    VAIST_OP_CONJ_TRANS  = 2,  /* cuSOLVER 'C' */
} VaistLinalgOp;

typedef enum VaistLinalgDatatype {
    VAIST_DTYPE_F32 = 0,  /* CUDA_R_32F / HIP_R_32F */
    VAIST_DTYPE_F64 = 1,  /* CUDA_R_64F */
    VAIST_DTYPE_C64 = 4,  /* CUDA_C_32F */
    VAIST_DTYPE_C128= 5,  /* CUDA_C_64F */
} VaistLinalgDatatype;

/* ---- LU (getrf/getrs). A(m,n) leading lda; pivot out ipiv[0..min(m,n)-1]. ---- */
VAIST_API VaistStatus vaist_linalg_sgetrf(VaistRuntime*rt,float*A,float*X,int*ipiv,int m,int n,int lda,int*info);
VAIST_API VaistStatus vaist_linalg_dgetrf(VaistRuntime*rt,double*A,double*X,int*ipiv,int m,int n,int lda,int*info);
VAIST_API VaistStatus vaist_linalg_sgetrs(VaistRuntime*rt,const float*A,float*X,int*ipiv,int n,int nrhs,int lda,int ldb,int trans);
VAIST_API VaistStatus vaist_linalg_dgetrs(VaistRuntime*rt,const double*A,double*X,int*ipiv,int n,int nrhs,int lda,int ldb,int trans);

/* ---- Cholesky (potrf / potri / potrs). ---- */
VAIST_API VaistStatus vaist_linalg_spotrf(VaistRuntime*rt,float*A,int n,int lda,int uplo,int*info);
VAIST_API VaistStatus vaist_linalg_dpotrf(VaistRuntime*rt,double*A,int n,int lda,int uplo,int*info);
VAIST_API VaistStatus vaist_linalg_spotrs(VaistRuntime*rt,const float*A,float*B,int n,int nrhs,int lda,int ldb,int uplo);
VAIST_API VaistStatus vaist_linalg_dpotrs(VaistRuntime*rt,const double*A,double*B,int n,int nrhs,int lda,int ldb,int uplo);
VAIST_API VaistStatus vaist_linalg_spotri(VaistRuntime*rt,float*A,int n,int lda,int uplo,int*info);

/* ---- QR (geqrf / orgqr / ormqr). ---- */
VAIST_API VaistStatus vaist_linalg_sgeqrf(VaistRuntime*rt,float*A,float*tau,int m,int n,int lda,int*info);
VAIST_API VaistStatus vaist_linalg_dgeqrf(VaistRuntime*rt,double*A,double*tau,int m,int n,int lda,int*info);
VAIST_API VaistStatus vaist_linalg_sorgqr(VaistRuntime*rt,float*A,int m,int n,int k,int lda,int*tau,int*info);

/* ---- SVD (gesvd). 'jobz' 0=singular values only, 1=U/Vt returned. ---- */
VAIST_API VaistStatus vaist_linalg_sgesvd(VaistRuntime*rt,int jobz,float*A,int m,int n,int lda,float*S,float*U,int ldu,float*Vt,int ldvt,int*info);
VAIST_API VaistStatus vaist_linalg_dgesvd(VaistRuntime*rt,int jobz,double*A,int m,int n,int lda,double*S,double*U,int ldu,double*Vt,int ldvt,int*info);

/* ---- Symmetric/Hermitian eig (syevd via Jacobi). Outputs W + optional V. ---- */
VAIST_API VaistStatus vaist_linalg_ssyevd(VaistRuntime*rt,int jobz,int uplo,float*A,int n,int lda,float*W,float*V,int ldv,int*info);
VAIST_API VaistStatus vaist_linalg_dsyevd(VaistRuntime*rt,int jobz,int uplo,double*A,int n,int lda,double*W,double*V,int ldv,int*info);

/* ---- Generic-X (datatype-tagged) variants — 1:1 with the cuSOLVER proxy. ---- */
VAIST_API VaistStatus vaist_linalg_potrfX(VaistRuntime*rt,int uplo,int64_t n,int dt,void*A,int64_t lda,int*info);
VAIST_API VaistStatus vaist_linalg_geqrfX(VaistRuntime*rt,int64_t m,int64_t n,int dt,void*A,int64_t lda,void*tau,int64_t*info);
VAIST_API VaistStatus vaist_linalg_syevdX(VaistRuntime*rt,int jobz,int uplo,int64_t n,int dtA,int dtW,void*A,int64_t lda,void*W,int*info);
VAIST_API VaistStatus vaist_linalg_gesvdjX(VaistRuntime*rt,int jobz,int64_t m,int64_t n,int dt,void*A,int64_t lda,void*S,void*U,int64_t ldu,void*V,int64_t ldv,int*info);

/* ---- Batched (CPU: loops the single-matrix routines). ---- */
VAIST_API VaistStatus vaist_linalg_spotrfBatched(VaistRuntime*rt,float**Aarray,int n,int lda,int uplo,int batchCount,int*infoArray);
VAIST_API VaistStatus vaist_linalg_dgetrsBatched(VaistRuntime*rt,const double**Aarray,const double**Barray,double**Xarray,int*ipivArray,int n,int nrhs,int lda,int ldb,int trans,int batchCount);

/* Best-path selector: CPU for now; stubs the GPU tile path. */
VAIST_API VaistComputePath vaist_linalg_best_path(const VaistRuntime*rt,int64_t n,int dt);
#ifdef __cplusplus
}
#endif
#endif
