#include "vaist_xpu.h"
#include "vaist_core.h"
#include <cstdio>
int main(){
    int rc = 0;
    float A[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float B[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float C[4];
    vaist_xpu_vector_params_t params;
    params.count = 4;
    vaist_xpu_vector_add(A, B, C, &params);
    if (!(C[0] == 6.0f && C[3] == 12.0f)) { std::fprintf(stderr,"[T] FAIL:test_xpu vector\n"); rc = 1; }
    // Perf hints (OpenVINO performance_mode analogue): same transpose, any tile
    const vaist_xpu_perf_hint_t hints[3] = {VAIST_XPU_HINT_DEFAULT, VAIST_XPU_HINT_LATENCY, VAIST_XPU_HINT_THROUGHPUT};
    for (int i = 0; i < 3; i++) {
        float M[6] = {1, 2, 3, 4, 5, 6};   // 2x3
        float T[6] = {0};                   // 3x2
        vaist_xpu_matrix_params_t mp;
        mp.rows = 2; mp.cols = 3;
        if (vaist_xpu_set_perf_hint(hints[i]) != VAIST_OK) { std::fprintf(stderr,"[T] FAIL:test_xpu set-hint=%d\n",(int)hints[i]); rc = 1; }
        if (vaist_xpu_get_perf_hint() != hints[i]) { std::fprintf(stderr,"[T] FAIL:test_xpu get-hint=%d\n",(int)hints[i]); rc = 1; }
        if (vaist_xpu_matrix_transpose(M, T, &mp) != VAIST_OK) { std::fprintf(stderr,"[T] FAIL:test_xpu transpose-hint=%d\n",(int)hints[i]); rc = 1; }
        // T[j*2+i] must equal M[i*3+j]: expect {1,4,2,5,3,6}
        float E[6] = {1, 4, 2, 5, 3, 6};
        for (int k = 0; k < 6; k++) {
            if (T[k] != E[k]) { std::fprintf(stderr,"[T] FAIL:test_xpu transpose-hint=%d k=%d got=%f\n",(int)hints[i],k,T[k]); rc = 1; break; }
        }
    }
    if (vaist_xpu_set_perf_hint((vaist_xpu_perf_hint_t)99) != VAIST_INVALID_ARGUMENT) { std::fprintf(stderr,"[T] FAIL:test_xpu bad-hint-accepted\n"); rc = 1; }
    std::fprintf(stderr,"[T] %s:test_xpu rc=%d C0=%f C3=%f\n",rc?"FAIL":"PASS",rc,C[0],C[3]);
    return rc;
}
