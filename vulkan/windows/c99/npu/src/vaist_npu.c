#include "vaist_npu.h"
#include "vaist_core.h"
#include <string.h>
#include <stdio.h>
#ifndef DBG_TRACE
#define DBG_TRACE(...) do { fprintf(stderr, "[T] %s:%d %s: ", __FILE__, __LINE__, __func__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } while(0)
#endif

// Intel AI Boost NPU capabilities
typedef struct {
    const char* name;
    uint32_t arch_version;
    uint32_t tensor_cores_available;
    uint32_t int8_support;
    uint32_t sparsity_support;
    uint32_t npu_peak_tops_int8;
} vaist_npu_caps_t;

/* ---- Level Zero dynamic binding (no link dependency): the Intel AI Boost
 * NPU is exposed inbox as a VPU-type Level Zero device (verified: type=5,
 * vendor=0x8086, e.g. device=0x7D1D on Meteor Lake). Only the verified
 * struct prefix (through deviceId) is read; name/clock fields are driver
 * garbage on the NPU driver and must not be trusted. ---- */
typedef int32_t vaist_npu_ze_result_t;
typedef uint32_t vaist_npu_ze_init_flags_t;
typedef struct _vaist_npu_ze_driver_t *vaist_npu_ze_driver_t;
typedef struct _vaist_npu_ze_device_t *vaist_npu_ze_device_t;
typedef struct {
    uint32_t stype;
    void* pNext;
    int32_t type;
    uint32_t vendorId;
    uint32_t deviceId;
    uint32_t flags;
    unsigned char reserved[360 - 32];
} vaist_npu_ze_device_props_t;
#define VAIST_NPU_ZE_SUCCESS 0
#define VAIST_NPU_ZE_INTEL_VENDOR 0x8086u
#define VAIST_NPU_ZE_TYPE_VPU 5
#ifdef _WIN32
#define VAIST_NPU_ZE_CALL __stdcall
#else
#define VAIST_NPU_ZE_CALL
#endif
typedef vaist_npu_ze_result_t (VAIST_NPU_ZE_CALL *vaist_npu_pfn_zeInit)(vaist_npu_ze_init_flags_t flags);
typedef vaist_npu_ze_result_t (VAIST_NPU_ZE_CALL *vaist_npu_pfn_zeDriverGet)(uint32_t* pCount, vaist_npu_ze_driver_t* phDrivers);
typedef vaist_npu_ze_result_t (VAIST_NPU_ZE_CALL *vaist_npu_pfn_zeDeviceGet)(vaist_npu_ze_driver_t hDriver, uint32_t* pCount, vaist_npu_ze_device_t* phDevices);
typedef vaist_npu_ze_result_t (VAIST_NPU_ZE_CALL *vaist_npu_pfn_zeDeviceGetProperties)(vaist_npu_ze_device_t hDevice, vaist_npu_ze_device_props_t* pProps);
#if defined(_WIN32)
#include <windows.h>
static void* npu_lib_open(const char* n) { return (void*)LoadLibraryA(n); }
static void* npu_lib_sym(void* h, const char* n) { return h ? (void*)GetProcAddress((HMODULE)h, n) : NULL; }
static void npu_lib_close(void* h) { if (h) FreeLibrary((HMODULE)h); }
static const char* npu_ze_libname(void) { return "ze_loader.dll"; }
#else
#include <dlfcn.h>
static void* npu_lib_open(const char* n) { return dlopen(n, RTLD_NOW | RTLD_LOCAL); }
static void* npu_lib_sym(void* h, const char* n) { return h ? dlsym(h, n) : NULL; }
static void npu_lib_close(void* h) { if (h) dlclose(h); }
static const char* npu_ze_libname(void) { return "libze_loader.so.1"; }
#endif

static int g_l0_probed = 0;
static int g_l0_npu_present = 0;
static uint32_t g_l0_npu_device_id = 0;

static void vaist_npu_l0_probe(void) {
    void* lib;
    vaist_npu_pfn_zeInit pInit;
    vaist_npu_pfn_zeDriverGet pDriverGet;
    vaist_npu_pfn_zeDeviceGet pDeviceGet;
    vaist_npu_pfn_zeDeviceGetProperties pProps;
    vaist_npu_ze_driver_t drivers[8];
    uint32_t ndrivers = 8, i;
    if (g_l0_probed)
        return;
    g_l0_probed = 1;
    lib = npu_lib_open(npu_ze_libname());
    if (!lib) { DBG_TRACE("npu_l0_probe path=no-loader -> absent"); return; }
    pInit = (vaist_npu_pfn_zeInit)npu_lib_sym(lib, "zeInit");
    pDriverGet = (vaist_npu_pfn_zeDriverGet)npu_lib_sym(lib, "zeDriverGet");
    pDeviceGet = (vaist_npu_pfn_zeDeviceGet)npu_lib_sym(lib, "zeDeviceGet");
    pProps = (vaist_npu_pfn_zeDeviceGetProperties)npu_lib_sym(lib, "zeDeviceGetProperties");
    if (!pInit || !pDriverGet || !pDeviceGet || !pProps) {
        DBG_TRACE("npu_l0_probe path=no-exports -> absent");
        npu_lib_close(lib);
        return;
    }
    if (pInit(0) != VAIST_NPU_ZE_SUCCESS) {
        DBG_TRACE("npu_l0_probe path=init-fail -> absent");
        npu_lib_close(lib);
        return;
    }
    ndrivers = 8;
    if (pDriverGet(&ndrivers, drivers) != VAIST_NPU_ZE_SUCCESS) {
        DBG_TRACE("npu_l0_probe path=driver-get-fail -> absent");
        npu_lib_close(lib);
        return;
    }
    for (i = 0; i < ndrivers && i < 8; i++) {
        vaist_npu_ze_device_t devs[16];
        uint32_t ndev = 16, j;
        if (pDeviceGet(drivers[i], &ndev, devs) != VAIST_NPU_ZE_SUCCESS)
            continue;
        for (j = 0; j < ndev && j < 16; j++) {
            vaist_npu_ze_device_props_t pr;
            memset(&pr, 0, sizeof(pr));
            pr.stype = 0x1;
            if (pProps(devs[j], &pr) != VAIST_NPU_ZE_SUCCESS)
                continue;
            if (pr.vendorId == VAIST_NPU_ZE_INTEL_VENDOR && pr.type == VAIST_NPU_ZE_TYPE_VPU) {
                g_l0_npu_present = 1;
                g_l0_npu_device_id = pr.deviceId;
                DBG_TRACE("npu_l0_probe found device=0x%04x", (unsigned)pr.deviceId);
            }
        }
    }
    if (!g_l0_npu_present)
        DBG_TRACE("npu_l0_probe path=no-vpu-device -> absent");
    npu_lib_close(lib);
}

static vaist_npu_caps_t g_npu_caps = {0};
static int npu_initialized = 0;
static vaist_npu_perf_hint_t g_perf_hint = VAIST_NPU_HINT_DEFAULT;

VAIST_API VaistStatus vaist_npu_set_perf_hint(vaist_npu_perf_hint_t hint) {
    if (hint < VAIST_NPU_HINT_DEFAULT || hint > VAIST_NPU_HINT_EFFICIENCY)
        return VAIST_INVALID_ARGUMENT;
    g_perf_hint = hint;
    DBG_TRACE("npu_set_perf_hint hint=%d", (int)hint);
    return VAIST_OK;
}

VAIST_API vaist_npu_perf_hint_t vaist_npu_get_perf_hint(void) {
    return g_perf_hint;
}

/* Tile edge per hint: LATENCY/EFFICIENCY favor small tiles (first result
 * fast, low power), THROUGHPUT/DEFAULT favor large tiles. */
static size_t npu_gemm_tile(void) {
    if (g_perf_hint == VAIST_NPU_HINT_LATENCY || g_perf_hint == VAIST_NPU_HINT_EFFICIENCY)
        return 32;
    if (g_perf_hint == VAIST_NPU_HINT_THROUGHPUT)
        return 256;
    return 128;
}

int vaist_npu_is_available(void) {
    vaist_npu_l0_probe();
    DBG_TRACE("npu_is_available present=%d device=0x%04x", g_l0_npu_present, (unsigned)g_l0_npu_device_id);
    return g_l0_npu_present;
}

VAIST_API uint32_t vaist_npu_device_id(void) {
    vaist_npu_l0_probe();
    return g_l0_npu_device_id;
}

static void vaist_npu_init(void) {
    if (npu_initialized)
        return;
    memset(&g_npu_caps, 0, sizeof(g_npu_caps));
    g_npu_caps.name = "Intel_AI_Boost";
    g_npu_caps.arch_version = 1;
    g_npu_caps.tensor_cores_available = 1;
    g_npu_caps.int8_support = 1;
    g_npu_caps.sparsity_support = 1;
    g_npu_caps.npu_peak_tops_int8 = 11;
    npu_initialized = 1;
}

// INT8 GEMM optimized for Intel AI Boost
VaistStatus vaist_npu_gemm_int8(const int8_t* A, const int8_t* B, int32_t* C,
                                size_t M, size_t N, size_t K,
                                const vaist_npu_gemm_params_t* params) {
    size_t T, i, j, k, i_end, j_end, k_end;
    int sparse = 0;
    if (!A || !B || !C || M == 0 || N == 0 || K == 0)
        return VAIST_INVALID_ARGUMENT;

    vaist_npu_init();
    if (params && (params->flags & VAIST_NPU_FLAG_SPARSE))
        sparse = 1;
    T = npu_gemm_tile();
    DBG_TRACE("npu_gemm MxNxK=%lu/%lu/%lu tile=%lu sparse=%d",
        (unsigned long)M, (unsigned long)N, (unsigned long)K, (unsigned long)T, sparse);

    // Tiled INT8 GEMM - no hardware dependencies
    for (i = 0; i < M; i += T) {
        for (j = 0; j < N; j += T) {
            for (k = 0; k < K; k += T) {
                size_t ii, jj, kk;
                i_end = (i + T < M) ? i + T : M;
                j_end = (j + T < N) ? j + T : N;
                k_end = (k + T < K) ? k + T : K;
                for (ii = i; ii < i_end; ii++) {
                    for (jj = j; jj < j_end; jj++) {
                        int32_t sum = C[ii * N + jj];
                        for (kk = k; kk < k_end; kk++) {
                            int8_t av = A[ii * K + kk];
                            if (sparse && av == 0)
                                continue; /* 0 * b == 0 */
                            sum += (int32_t)av * (int32_t)B[kk * N + jj];
                        }
                        C[ii * N + jj] = sum;
                    }
                }
            }
        }
    }
    DBG_TRACE("npu_gemm -> OK C0=%d", (int)C[0]);
    return VAIST_OK;
}

// INT8 Conv2D
VaistStatus vaist_npu_conv2d_int8(const int8_t* input, const int8_t* kernel, int32_t* output,
                                  const vaist_npu_conv2d_params_t* params) {
    if (!input || !kernel || !output || !params)
        return VAIST_INVALID_ARGUMENT;
    if (params->stride <= 0 || params->dilation <= 0 || params->in_h <= 0 ||
        params->in_w <= 0 || params->in_c <= 0 || params->kernel_h <= 0 ||
        params->kernel_w <= 0 || params->out_c <= 0)
        return VAIST_INVALID_ARGUMENT;
    
    vaist_npu_init();
    
    int out_h = (params->in_h - params->kernel_h + 2 * params->pad) / params->stride + 1;
    int out_w = (params->in_w - params->kernel_w + 2 * params->pad) / params->stride + 1;
    if (out_h <= 0 || out_w <= 0)
        return VAIST_INVALID_ARGUMENT;
    DBG_TRACE("npu_conv2d in=%dx%dx%d k=%dx%d out_c=%d out=%dx%d",
        params->in_h, params->in_w, params->in_c,
        params->kernel_h, params->kernel_w, params->out_c, out_h, out_w);
    
    memset(output, 0, out_h * out_w * params->out_c * sizeof(int32_t));
    
    for (int oc = 0; oc < params->out_c; oc++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                for (int kh = 0; kh < params->kernel_h; kh++) {
                    for (int kw = 0; kw < params->kernel_w; kw++) {
                        for (int ic = 0; ic < params->in_c; ic++) {
                            int ih = oh * params->stride - params->pad + kh * params->dilation;
                            int iw = ow * params->stride - params->pad + kw * params->dilation;
                            if (ih >= 0 && ih < params->in_h && iw >= 0 && iw < params->in_w) {
                                int8_t in_val = input[(ih * params->in_w + iw) * params->in_c + ic];
                                int8_t k_val = kernel[(kh * params->kernel_w * params->out_c + kw * params->out_c + ic) * params->out_c + oc];
                                output[(oh * out_w + ow) * params->out_c + oc] += (int32_t)in_val * (int32_t)k_val;
                            }
                        }
                    }
                }
            }
        }
    }
    return VAIST_OK;
}

// ReLU activation
VaistStatus vaist_npu_relu_int32(const int32_t* input, int32_t* output, size_t count) {
    if (!input || !output || count == 0)
        return VAIST_INVALID_ARGUMENT;
    
    vaist_npu_init();
    
    for (size_t i = 0; i < count; i++)
        output[i] = (input[i] > 0) ? input[i] : 0;

    DBG_TRACE("npu_relu count=%lu -> OK", (unsigned long)count);
    return VAIST_OK;
}

// Device probe - no child process for NPU module
VaistStatus vaist_npu_device_probe(void) {
    if (!npu_initialized)
        vaist_npu_init();
    
    if (!vaist_npu_is_available())
        return VAIST_DEVICE_ERROR;
    
    return VAIST_OK;
}