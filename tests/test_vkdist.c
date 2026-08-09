/**
 * \file test_vkdist.c
 * \brief Loopback vertical-slice harness for the vkdist distributed layer.
 *
 * Bootstraps a minimal Vulkan instance + physical/logical device and a
 * VkBLASContext (mirroring test_vkblas), then runs the FULL vkdist path in one
 * process over loopback TCP:
 *
 *   1. vkdist_server_start binds an ephemeral port on 127.0.0.1.
 *   2. A server thread accepts one connection and runs vkdist_server_run
 *      (which hosts the Vulkan device and serves RPCs until BYE).
 *   3. The main thread connects as a client, registers three 8x8 f32 buffers
 *      (A, B, C), uploads A=ones and B=ramp, remotely dispatches
 *      vkblas_sgemm (alpha=1, beta=0), reads C back and compares to a CPU
 *      reference, then a second sgemm (alpha=1, beta=0.5) that reads C in
 *      place (beta path), reads back and compares again.
 *   4. The client sends BYE; the server thread joins and its VkResult must be
 *      VK_SUCCESS.
 *
 * Threading: POSIX threads (winpthreads). This MinGW-W64 build is the "posix"
 * threading model, so pthread.h is available and links with -lpthread; the same
 * source compiles unmodified on Linux. (Win32 CreateThread would also work but
 * is Windows-only; pthreads keeps the harness portable.)
 *
 * Build (Windows):
 *   gcc -std=c99 -IC:/VulkanSDK/1.4.357.0/Include -IF:/VAiT/include \
 *       F:/VAiT/tests/test_vkdist.c vkdist.o vkruntime.o vkblas.o \
 *       C:/VulkanSDK/1.4.357.0/Lib/vulkan-1.lib -lws2_32 -lpthread \
 *       -o test_vkdist.exe
 *
 * Exit status: 0 when every run test passes, 1 on any real failure.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* winsock2.h must precede any header that might pull in windows.h, so the
   winsock.h/winsock2.h redefinition conflict cannot occur. */
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#endif

#include <pthread.h>

#include "../include/vkdist/vkdist.h"
#include "../include/vkblas/vkblas.h"

/* ===========================================================================
 * Test parameters
 * ========================================================================== */

#define TEST_GEMM_M 8    /**< Rows of op(A) and C.                          */
#define TEST_GEMM_N 8    /**< Cols of op(B) and C.                          */
#define TEST_GEMM_K 8    /**< Cols of op(A), rows of op(B).                 */
#define TEST_LDA    8    /**< Leading dimension of A (>= m).                */
#define TEST_LDB    8    /**< Leading dimension of B (>= k).                */
#define TEST_LDC    8    /**< Leading dimension of C (>= m).                */
#define TEST_TOLERANCE 1e-3f /**< f32 comparison tolerance.                 */

/* ===========================================================================
 * Server thread plumbing
 * ========================================================================== */

typedef struct {
    int             listen_fd;
    VkPhysicalDevice pd;
    VkDevice        dev;
    VkBLASContext  *blas;
    VkResult        result;
} server_arg_t;

static void *server_thread_fn(void *arg)
{
    server_arg_t *a = (server_arg_t *)arg;
    int conn = vkdist_server_accept(a->listen_fd);
    if (conn < 0) {
        printf("  server      : accept failed\n");
        a->result = VK_ERROR_UNKNOWN;
        return NULL;
    }
    printf("  server      : accepted connection, serving RPCs\n");
    a->result = vkdist_server_run(a->pd, a->dev, a->blas, conn);
    printf("  server      : session ended (VkResult=%d)\n", (int)a->result);
    return NULL;
}

/* ===========================================================================
 * Bootstrap helpers (mirror test_vkblas.c)
 * ========================================================================== */

static VkResult create_instance(const char *app_name, VkInstance *out_instance)
{
    VkApplicationInfo app_info;
    memset(&app_info, 0, sizeof(app_info));
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = app_name;
    app_info.applicationVersion = 1;
    app_info.pEngineName = "vait";
    app_info.engineVersion = 1;
    app_info.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo create_info;
    memset(&create_info, 0, sizeof(create_info));
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;
    create_info.enabledLayerCount = 0;
    create_info.ppEnabledLayerNames = NULL;
    create_info.enabledExtensionCount = 0;
    create_info.ppEnabledExtensionNames = NULL;
    return vkCreateInstance(&create_info, NULL, out_instance);
}

static VkResult find_physical_device(VkInstance instance,
                                     VkPhysicalDevice *out_physical_device)
{
    uint32_t count = 0;
    VkResult r = vkEnumeratePhysicalDevices(instance, &count, NULL);
    if (r != VK_SUCCESS || count == 0) return VK_ERROR_INITIALIZATION_FAILED;

    VkPhysicalDevice *devices =
        (VkPhysicalDevice *)malloc(count * sizeof(VkPhysicalDevice));
    if (!devices) return VK_ERROR_OUT_OF_HOST_MEMORY;

    r = vkEnumeratePhysicalDevices(instance, &count, devices);
    if (r == VK_SUCCESS) *out_physical_device = devices[0];
    free(devices);
    return r;
}

static VkBool32 query_shader_int64(VkPhysicalDevice physical_device)
{
    VkPhysicalDeviceFeatures2 features2;
    memset(&features2, 0, sizeof(features2));
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    vkGetPhysicalDeviceFeatures2(physical_device, &features2);
    return features2.features.shaderInt64;
}

static VkBool32 queue_family_supports_compute(VkPhysicalDevice physical_device,
                                              uint32_t family)
{
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, NULL);
    if (count == 0) return VK_FALSE;

    VkQueueFamilyProperties *props =
        (VkQueueFamilyProperties *)malloc(count * sizeof(*props));
    if (!props) return VK_FALSE;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, props);

    VkBool32 supported = VK_FALSE;
    if (family < count) {
        supported =
            (props[family].queueFlags & VK_QUEUE_COMPUTE_BIT) ? VK_TRUE : VK_FALSE;
    }
    free(props);
    return supported;
}

static VkBool32 device_extension_available(VkPhysicalDevice physical_device,
                                           const char *name)
{
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(physical_device, NULL, &count, NULL);
    if (count == 0) return VK_FALSE;

    VkExtensionProperties *props =
        (VkExtensionProperties *)malloc(count * sizeof(*props));
    if (!props) return VK_FALSE;
    vkEnumerateDeviceExtensionProperties(physical_device, NULL, &count, props);

    VkBool32 found = VK_FALSE;
    for (uint32_t i = 0; i < count; i++) {
        if (strcmp(props[i].extensionName, name) == 0) { found = VK_TRUE; break; }
    }
    free(props);
    return found;
}

/* Create a logical device on queue family 0 with the feature set VKBLAS needs
   (shaderInt64 + the coop-matrix extension when the device advertises it). */
static VkResult create_device(VkPhysicalDevice physical_device,
                              VkDevice *out_device)
{
    float priority = 1.0f;

    VkDeviceQueueCreateInfo queue_info;
    memset(&queue_info, 0, sizeof(queue_info));
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = 0;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;

    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop_supported;
    VkPhysicalDeviceFeatures2 supported;
    memset(&coop_supported, 0, sizeof(coop_supported));
    coop_supported.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    memset(&supported, 0, sizeof(supported));
    supported.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    supported.pNext = &coop_supported;
    vkGetPhysicalDeviceFeatures2(physical_device, &supported);

    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop_enable;
    VkPhysicalDeviceFeatures2 features2;
    memset(&coop_enable, 0, sizeof(coop_enable));
    coop_enable.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    memset(&features2, 0, sizeof(features2));
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.features.shaderInt64 = VK_TRUE;

    void *pNext = NULL;
    if (coop_supported.cooperativeMatrix) {
        coop_enable.cooperativeMatrix = VK_TRUE;
        coop_enable.pNext = pNext;
        pNext = &coop_enable;
    }
    features2.pNext = pNext;

    const char *extensions[1];
    uint32_t ext_count = 0;
    if (coop_supported.cooperativeMatrix &&
        device_extension_available(physical_device,
                                   VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) {
        extensions[ext_count++] = VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME;
    }

    VkDeviceCreateInfo create_info;
    memset(&create_info, 0, sizeof(create_info));
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.pNext = &features2;
    create_info.queueCreateInfoCount = 1;
    create_info.pQueueCreateInfos = &queue_info;
    create_info.enabledExtensionCount = ext_count;
    create_info.ppEnabledExtensionNames = extensions;
    create_info.pEnabledFeatures = NULL;

    return vkCreateDevice(physical_device, &create_info, NULL, out_device);
}

/* ===========================================================================
 * Network helpers (test-side; not part of the library)
 * ========================================================================== */

/* Query the port a listen socket actually bound to (port 0 = ephemeral). */
static int get_bound_port(int fd)
{
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
#ifdef _WIN32
    SOCKET s = (SOCKET)fd;
    int alen = (int)sizeof(addr);
#else
    int s = fd;
    socklen_t alen = (socklen_t)sizeof(addr);
#endif
    if (getsockname(s, (struct sockaddr *)&addr, &alen) != 0)
        return -1;
    return (int)ntohs(addr.sin_port);
}

/* ===========================================================================
 * Reference computation + comparison
 * ========================================================================== */

/* CPU reference: D = alpha * A * B + beta * C (f32 path, double-accumulated),
   column-major. C is assumed to hold the previous result for the beta pass. */
static void compute_reference_f32(const float *A, int32_t lda,
                                  const float *B, int32_t ldb,
                                  float alpha, float beta,
                                  const float *C, int32_t ldc,
                                  float *D, int32_t ldd,
                                  int32_t m, int32_t n, int32_t k)
{
    for (int32_t row = 0; row < m; row++) {
        for (int32_t c = 0; c < n; c++) {
            double acc = 0.0;
            for (int32_t t = 0; t < k; t++) {
                acc += (double)A[row + t * lda] * (double)B[t + c * ldb];
            }
            D[row + c * ldd] = (float)(alpha * acc + beta * (double)C[row + c * ldc]);
        }
    }
}

static int check_output(const char *name, const float *got,
                        const float *expected, uint32_t count, float tolerance)
{
    int pass = 1;
    uint32_t mismatches = 0;
    for (uint32_t i = 0; i < count; i++) {
        float diff = fabsf(got[i] - expected[i]);
        if (diff > tolerance) {
            if (mismatches < 8) {
                printf("    mismatch[%u]: got %.6f expected %.6f (diff %.3e)\n",
                       i, got[i], expected[i], diff);
            }
            mismatches++;
            pass = 0;
        }
    }
    printf("  %-28s : %s\n", name, pass ? "PASS" : "FAIL");
    return pass;
}

/* ===========================================================================
 * main
 * ========================================================================== */

int main(void)
{
    const int32_t m = TEST_GEMM_M;
    const int32_t n = TEST_GEMM_N;
    const int32_t k = TEST_GEMM_K;
    const VkDeviceSize szA = (VkDeviceSize)(m * k) * sizeof(float);
    const VkDeviceSize szB = (VkDeviceSize)(k * n) * sizeof(float);
    const VkDeviceSize szC = (VkDeviceSize)(m * n) * sizeof(float);
    const uint32_t elem_count = (uint32_t)(m * n);

    static float A[TEST_GEMM_M * TEST_GEMM_K];
    static float B[TEST_GEMM_K * TEST_GEMM_N];
    static float C_init[TEST_GEMM_M * TEST_GEMM_N];
    static float ref1[TEST_GEMM_M * TEST_GEMM_N];
    static float ref2[TEST_GEMM_M * TEST_GEMM_N];
    static float out1[TEST_GEMM_M * TEST_GEMM_N];
    static float out2[TEST_GEMM_M * TEST_GEMM_N];

    int overall_pass = 1;

    /* ── 1. Vulkan bootstrap ───────────────────────────────────────────── */
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    VkDevice dev = VK_NULL_HANDLE;
    VkBLASContext *blas = NULL;

    VkResult r = create_instance("test_vkdist", &instance);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkdist: vkCreateInstance failed (%d)\n", (int)r);
        return 1;
    }
    r = find_physical_device(instance, &pd);
    if (r != VK_SUCCESS) {
        printf("test_vkdist: SKIP (no physical device found)\n");
        vkDestroyInstance(instance, NULL);
        return 0;
    }
    if (query_shader_int64(pd) == VK_FALSE) {
        printf("test_vkdist: SKIP (shaderInt64 not supported)\n");
        vkDestroyInstance(instance, NULL);
        return 0;
    }
    if (queue_family_supports_compute(pd, 0) == VK_FALSE) {
        printf("test_vkdist: SKIP (queue family 0 lacks compute)\n");
        vkDestroyInstance(instance, NULL);
        return 0;
    }
    r = create_device(pd, &dev);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkdist: vkCreateDevice failed (%d)\n", (int)r);
        goto cleanup;
    }
    r = vkblas_create_context(pd, dev, &blas);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkdist: vkblas_create_context failed (%d)\n", (int)r);
        goto cleanup;
    }
    printf("test_vkdist: device ready (arch=%s, tier=%u)\n",
           vkblas_get_arch_name(blas), vkblas_get_arch_index(blas));

    /* ── 2. Server socket (ephemeral port on loopback) ─────────────────── */
    int listen_fd = vkdist_server_start(0, "127.0.0.1");
    if (listen_fd < 0) {
        fprintf(stderr, "test_vkdist: vkdist_server_start failed\n");
        goto cleanup;
    }
    int port = get_bound_port(listen_fd);
    if (port <= 0) {
        fprintf(stderr, "test_vkdist: cannot resolve bound port\n");
        vkdist_close(listen_fd);
        goto cleanup;
    }
    printf("test_vkdist: server listening on 127.0.0.1:%d\n", port);

    /* ── 3. Server thread (accept + serve until BYE) ───────────────────── */
    server_arg_t arg;
    arg.listen_fd = listen_fd;
    arg.pd = pd;
    arg.dev = dev;
    arg.blas = blas;
    arg.result = VK_ERROR_UNKNOWN;

    pthread_t thread;
    int trc = pthread_create(&thread, NULL, server_thread_fn, &arg);
    if (trc != 0) {
        fprintf(stderr, "test_vkdist: pthread_create failed (%d)\n", trc);
        vkdist_close(listen_fd);
        goto cleanup;
    }

    /* ── 4. Client connect + HELLO handshake ───────────────────────────── */
    int fd = vkdist_client_connect("127.0.0.1", (uint16_t)port);
    if (fd < 0) {
        fprintf(stderr, "test_vkdist: vkdist_client_connect failed\n");
        overall_pass = 0;
        pthread_join(thread, NULL);
        vkdist_close(listen_fd);
        goto cleanup;
    }
    printf("  client      : connected, handshake OK\n");

    /* ── 5. Register three remote buffers ───────────────────────────────── */
    uint64_t hA = 0, hB = 0, hC = 0;
    overall_pass &= (vkdist_register_buffer(fd, szA, &hA) == VK_SUCCESS) &&
                    (hA != 0) ? 1 : 0;
    overall_pass &= (vkdist_register_buffer(fd, szB, &hB) == VK_SUCCESS) &&
                    (hB != 0) ? 1 : 0;
    overall_pass &= (vkdist_register_buffer(fd, szC, &hC) == VK_SUCCESS) &&
                    (hC != 0) ? 1 : 0;
    printf("  client      : registered A=%llu B=%llu C=%llu\n",
           (unsigned long long)hA, (unsigned long long)hB,
           (unsigned long long)hC);

    /* ── 6. Fill inputs + CPU references (column-major) ────────────────── */
    for (int32_t row = 0; row < m; row++) {
        for (int32_t t = 0; t < k; t++)
            A[row + t * TEST_LDA] = 1.0f;
    }
    for (int32_t t = 0; t < k; t++) {
        for (int32_t c = 0; c < n; c++)
            B[t + c * TEST_LDB] = (float)(1 + t + 8 * c);
    }
    for (uint32_t i = 0; i < elem_count; i++)
        C_init[i] = 1.0f;

    /* ref1 = A*B (alpha=1, beta=0). ref2 = A*B + 0.5*ref1 (in-place C). */
    compute_reference_f32(A, TEST_LDA, B, TEST_LDB, 1.0f, 0.0f, C_init,
                          TEST_LDC, ref1, TEST_LDC, m, n, k);
    compute_reference_f32(A, TEST_LDA, B, TEST_LDB, 1.0f, 0.5f, ref1,
                          TEST_LDC, ref2, TEST_LDC, m, n, k);

    /* ── 7. Upload A, B, C ─────────────────────────────────────────────── */
    overall_pass &= (vkdist_upload(fd, hA, A, 0, szA) == VK_SUCCESS) ? 1 : 0;
    overall_pass &= (vkdist_upload(fd, hB, B, 0, szB) == VK_SUCCESS) ? 1 : 0;
    overall_pass &= (vkdist_upload(fd, hC, C_init, 0, szC) == VK_SUCCESS) ? 1 : 0;
    printf("  client      : uploaded A,B,C (%llu bytes each)\n",
           (unsigned long long)szA);

    /* ── 8. Remote GEMM #1: alpha=1, beta=0 (no C read) ────────────────── */
    float alpha = 1.0f;
    float beta0 = 0.0f;
    VkResult vkr = vkdist_sgemm(fd, m, n, k, &alpha, hA, hB, hC, &beta0,
                                TEST_LDA, TEST_LDB, TEST_LDC);
    if (vkr != VK_SUCCESS) {
        fprintf(stderr, "test_vkdist: sgemm#1 failed (VkResult=%d)\n", (int)vkr);
        overall_pass = 0;
    }
    vkr = vkdist_readback(fd, hC, 0, szC, out1);
    if (vkr != VK_SUCCESS) {
        fprintf(stderr, "test_vkdist: readback#1 failed (VkResult=%d)\n", (int)vkr);
        overall_pass = 0;
    }
    overall_pass &= check_output("sgemm beta=0  (remote C)",
                                 out1, ref1, elem_count, TEST_TOLERANCE);

    /* ── 9. Remote GEMM #2: alpha=1, beta=0.5 (in-place C read) ────────── */
    float beta05 = 0.5f;
    vkr = vkdist_sgemm(fd, m, n, k, &alpha, hA, hB, hC, &beta05,
                       TEST_LDA, TEST_LDB, TEST_LDC);
    if (vkr != VK_SUCCESS) {
        fprintf(stderr, "test_vkdist: sgemm#2 failed (VkResult=%d)\n", (int)vkr);
        overall_pass = 0;
    }
    vkr = vkdist_readback(fd, hC, 0, szC, out2);
    if (vkr != VK_SUCCESS) {
        fprintf(stderr, "test_vkdist: readback#2 failed (VkResult=%d)\n", (int)vkr);
        overall_pass = 0;
    }
    overall_pass &= check_output("sgemm beta=0.5 (in-place C read)",
                                 out2, ref2, elem_count, TEST_TOLERANCE);

    /* ── 10. BYE + join server thread ──────────────────────────────────── */
    vkdist_close(fd);
    pthread_join(thread, NULL);
    vkdist_close(listen_fd);
    if (arg.result != VK_SUCCESS) {
        fprintf(stderr, "test_vkdist: server session ended with VkResult=%d\n",
                (int)arg.result);
        overall_pass = 0;
    } else {
        printf("  server      : clean shutdown (BYE honored)\n");
    }

cleanup:
    if (blas) vkblas_destroy_context(blas);
    if (dev != VK_NULL_HANDLE) vkDestroyDevice(dev, NULL);
    if (instance != VK_NULL_HANDLE) vkDestroyInstance(instance, NULL);

    printf("test_vkdist: %s\n", overall_pass ? "PASS" : "FAIL");
    return overall_pass ? 0 : 1;
}
