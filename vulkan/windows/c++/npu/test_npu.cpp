#include "vaist_npu.h"
#include "vaist_core.h"
#include <cstdio>
#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
static int check_gemm(const char*tag, int use_sparse) {
    // A(1x4)={1,2,0,4} * B(4x1)={1,0,3,1} -> 1*1+2*0+0*3+4*1=5
    int8_t A[4] = {1, 2, 0, 4};
    int8_t B[4] = {1, 0, 3, 1};
    int32_t C[1] = {0};
    vaist_npu_gemm_params_t params = {0};
    params.M = 1; params.N = 1; params.K = 4;
    if (use_sparse) params.flags = VAIST_NPU_FLAG_SPARSE;
    VaistStatus s = vaist_npu_gemm_int8(A, B, C, 1, 1, 4, &params);
    int rc = (s == VAIST_OK && C[0] == 5) ? 0 : 1;
    std::fprintf(stderr,"[T] %s:test_npu_%s rc=%d s=%d C0=%d\n",rc?"FAIL":"PASS",tag,rc,(int)s,(int)C[0]);
    return rc;
}
int main(){
#if defined(_WIN32)
    int rc = 0;
    if (!vaist_npu_is_available()) { std::fprintf(stderr,"[T] FAIL:test_npu unavailable\n"); return 1; }
    std::fprintf(stderr,"[T] npu_is_available=1\n");
    {
        // Real Level Zero detection: Intel AI Boost NPU (this box: 0x7D1D)
        uint32_t did = vaist_npu_device_id();
        std::fprintf(stderr,"[T] npu_device_id=0x%04x\n",(unsigned)did);
        if (did == 0) { std::fprintf(stderr,"[T] FAIL:test_npu no-device-id\n"); return 1; }
    }
    // Perf hints (OpenVINO performance_mode analogue): same math, any tile
    const vaist_npu_perf_hint_t hints[4] = {VAIST_NPU_HINT_DEFAULT, VAIST_NPU_HINT_LATENCY, VAIST_NPU_HINT_THROUGHPUT, VAIST_NPU_HINT_EFFICIENCY};
    for (int i = 0; i < 4; i++) {
        if (vaist_npu_set_perf_hint(hints[i]) != VAIST_OK) { std::fprintf(stderr,"[T] FAIL:test_npu set-hint=%d\n",(int)hints[i]); rc = 1; }
        if (vaist_npu_get_perf_hint() != hints[i]) { std::fprintf(stderr,"[T] FAIL:test_npu get-hint=%d\n",(int)hints[i]); rc = 1; }
        rc |= check_gemm("gemm", 0);
    }
    // Sparse path (OpenVINO sparse-weights analogue): zeros skipped, same sum
    rc |= check_gemm("sparse", 1);
    // Invalid hint rejected
    if (vaist_npu_set_perf_hint((vaist_npu_perf_hint_t)99) != VAIST_INVALID_ARGUMENT) { std::fprintf(stderr,"[T] FAIL:test_npu bad-hint-accepted\n"); rc = 1; }
    // Static-shape validation (OpenVINO NPU static-shapes-only analogue)
    {
        vaist_npu_conv2d_params_t bad = {0};
        bad.stride = 0;
        if (vaist_npu_conv2d_int8(0, 0, 0, &bad) != VAIST_INVALID_ARGUMENT) { std::fprintf(stderr,"[T] FAIL:test_npu bad-conv-accepted\n"); rc = 1; }
    }
    std::fprintf(stderr,"[T] %s:test_npu rc=%d\n",rc?"FAIL":"PASS",rc);
    return rc;
#else
    return 1;
#endif
}
