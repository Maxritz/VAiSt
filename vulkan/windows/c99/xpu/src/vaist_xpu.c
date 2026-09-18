#include "vaist_xpu.h"
#include "vaist_core.h"
#include <string.h>
#include <stdio.h>
#ifndef DBG_TRACE
#define DBG_TRACE(...) do { fprintf(stderr, "[T] %s:%d %s: ", __FILE__, __LINE__, __func__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } while(0)
#endif

// x86-64 XPU capabilities
typedef struct {
    const char* cpu_vendor;
    uint32_t logical_cores;
    uint32_t cache_line_size;
    int avx2_available;
    int sse4_2_available;
} vaist_x86_64_caps_t;

static vaist_x86_64_caps_t g_xpu_caps = {0};
static int xpu_initialized = 0;
static vaist_xpu_perf_hint_t g_perf_hint = VAIST_XPU_HINT_DEFAULT;

VAIST_API VaistStatus vaist_xpu_set_perf_hint(vaist_xpu_perf_hint_t hint) {
    if (hint < VAIST_XPU_HINT_DEFAULT || hint > VAIST_XPU_HINT_THROUGHPUT)
        return VAIST_INVALID_ARGUMENT;
    g_perf_hint = hint;
    DBG_TRACE("xpu_set_perf_hint hint=%d", (int)hint);
    return VAIST_OK;
}

VAIST_API vaist_xpu_perf_hint_t vaist_xpu_get_perf_hint(void) {
    return g_perf_hint;
}

/* Transpose tile per hint: LATENCY favors small tiles (cache footprint),
 * THROUGHPUT/DEFAULT favor large tiles. */
static size_t xpu_transpose_tile(void) {
    return (g_perf_hint == VAIST_XPU_HINT_LATENCY) ? 16 : 32;
}

int vaist_xpu_is_available(void) {
#if defined(__x86_64__) || defined(_M_X64)
    return 1;
#else
    return 0;
#endif
}

vaist_xpu_optimization_level vaist_xpu_get_optimization_level(void) {
    return VAIST_OPTIMIZATION_BASIC;
}

static void vaist_xpu_init(void) {
    if (xpu_initialized)
        return;
    
    memset(&g_xpu_caps, 0, sizeof(g_xpu_caps));
    g_xpu_caps.cpu_vendor = "Intel/AMD";
    g_xpu_caps.logical_cores = 10;
    g_xpu_caps.cache_line_size = 64;
    g_xpu_caps.avx2_available = 1;
    g_xpu_caps.sse4_2_available = 1;
    
    xpu_initialized = 1;
}

// Vector add
VaistStatus vaist_xpu_vector_add(const float* A, const float* B, float* C,
                                 vaist_xpu_vector_params_t* params) {
    if (!A || !B || !C)
        return VAIST_INVALID_ARGUMENT;
    if (!params)
        return VAIST_INVALID_ARGUMENT;
    
    vaist_xpu_init();
    
    for (size_t i = 0; i < params->count; i++)
        C[i] = A[i] + B[i];
    
    return VAIST_OK;
}

// Matrix transpose with cache-optimized tiling
VaistStatus vaist_xpu_matrix_transpose(const float* A, float* B,
                                       vaist_xpu_matrix_params_t* params) {
    if (!A || !B || !params)
        return VAIST_INVALID_ARGUMENT;
    
    vaist_xpu_init();
    
    size_t rows = params->rows;
    size_t cols = params->cols;

    // Cache-optimized tiling, tile edge from perf hint
    const size_t TILE_SIZE = xpu_transpose_tile();
    DBG_TRACE("xpu_transpose rows=%lu cols=%lu tile=%lu",
        (unsigned long)rows, (unsigned long)cols, (unsigned long)TILE_SIZE);
    
    for (size_t i = 0; i < rows; i += TILE_SIZE) {
        for (size_t j = 0; j < cols; j += TILE_SIZE) {
            size_t i_end = (i + TILE_SIZE < rows) ? i + TILE_SIZE : rows;
            size_t j_end = (j + TILE_SIZE < cols) ? j + TILE_SIZE : cols;
            
            for (size_t ii = i; ii < i_end; ii++) {
                for (size_t jj = j; jj < j_end; jj++) {
                    B[jj * rows + ii] = A[ii * cols + jj];
                }
            }
        }
    }
    
    return VAIST_OK;
}

// Parallel reduce sum
VaistStatus vaist_xpu_reduce_sum(const float* A, float* B,
                                 vaist_xpu_reduce_params_t* params) {
    if (!A || !B || !params)
        return VAIST_INVALID_ARGUMENT;
    
    vaist_xpu_init();
    
    size_t count = params->count;
    size_t stride = params->stride;
    
    for (size_t i = 0; i < count; i += stride) {
        float sum = 0.0f;
        for (size_t j = 0; j < stride && i + j < count; j++) {
            sum += A[i + j];
        }
        B[i] = sum;
    }
    
    return VAIST_OK;
}