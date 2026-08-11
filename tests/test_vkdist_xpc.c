/**
 * \file test_vkdist_xpc.c
 * \brief Cross-PC vkdist smoke test: server on one machine, client on another.
 *
 * Server mode:  test_vkdist_xpc server <port>
 * Client mode:  test_vkdist_xpc client <ip> <port>
 *
 * The client registers/upload an f32 GEMM (C = alpha*A*B + beta*C) on the
 * remote server GPU, dispatches via vkdist_sgemm, reads the result back, and
 * verifies it against a CPU reference.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <vulkan/vulkan.h>
#include <vkruntime/vkruntime.h>
#include <vkblas/vkblas.h>
#include <vkdist/vkdist.h>

static VkBool32 device_extension_available(VkPhysicalDevice pd,
                                           const char *name)
{
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(pd, NULL, &count, NULL);
    if (count == 0) return VK_FALSE;
    VkExtensionProperties *props =
        (VkExtensionProperties *)malloc(count * sizeof(*props));
    vkEnumerateDeviceExtensionProperties(pd, NULL, &count, props);
    VkBool32 found = VK_FALSE;
    for (uint32_t i = 0; i < count; i++) {
        if (strcmp(props[i].extensionName, name) == 0) { found = VK_TRUE; break; }
    }
    free(props);
    return found;
}

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
                                   VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME))
        extensions[ext_count++] = VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME;

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
 * Shared GEMM data (client generates, server never sees host matrices)
 * ========================================================================== */

#define XM 64
#define XN 64
#define XK 64

static void compute_reference(const float *A, int32_t lda,
                              const float *B, int32_t ldb,
                              float alpha, float beta,
                              const float *C, int32_t ldc,
                              float *D, int32_t ldd,
                              int32_t m, int32_t n, int32_t k)
{
    for (int32_t row = 0; row < m; row++) {
        for (int32_t c = 0; c < n; c++) {
            double acc = 0.0;
            for (int32_t t = 0; t < k; t++)
                acc += (double)A[row + t * lda] * (double)B[t + c * ldb];
            D[row + c * ldd] = (float)(alpha * acc + beta * (double)C[row + c * ldc]);
        }
    }
}

/* ===========================================================================
 * Server mode
 * ========================================================================== */

static int run_server(uint16_t port)
{
    VkApplicationInfo appInfo = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "test_vkdist_xpc_server",
        .apiVersion = VK_API_VERSION_1_3,
    };
    VkInstanceCreateInfo instInfo = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &appInfo,
    };
    VkInstance instance;
    if (vkCreateInstance(&instInfo, NULL, &instance) != VK_SUCCESS) {
        fprintf(stderr, "server: vkCreateInstance failed\n");
        return 1;
    }

    uint32_t pdcount = 0;
    vkEnumeratePhysicalDevices(instance, &pdcount, NULL);
    if (pdcount == 0) { fprintf(stderr, "server: no GPU\n"); return 1; }
    VkPhysicalDevice pd;
    vkEnumeratePhysicalDevices(instance, &pdcount, &pd);

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(pd, &props);
    printf("server: GPU = %s\n", props.deviceName);

    VkDevice dev;
    if (create_device(pd, &dev) != VK_SUCCESS) {
        fprintf(stderr, "server: create_device failed\n");
        return 1;
    }

    VkBLASContext *blas = NULL;
    VkResult vr = vkblas_create_context(instance, pd, dev, &blas);
    if (vr != VK_SUCCESS) {
        fprintf(stderr, "server: vkblas_create_context failed (%d)\n", (int)vr);
        return 1;
    }

    int listen_fd = vkdist_server_start(port, NULL); /* INADDR_ANY */
    if (listen_fd < 0) {
        fprintf(stderr, "server: vkdist_server_start failed\n");
        return 1;
    }
    printf("server: listening on 0.0.0.0:%u\n", port);

    /* Serve connections forever — this is the worker the master talks to, so
       it must outlive a single session. vkdist_server_run owns and closes each
       accepted connection. */
    for (;;) {
        int conn_fd = vkdist_server_accept(listen_fd);
        if (conn_fd < 0) {
            fprintf(stderr, "server: accept failed\n");
            break;
        }
        printf("server: client connected\n");

        vr = vkdist_server_run(pd, dev, blas, conn_fd);
        printf("server: session ended (VkResult=%d)\n", (int)vr);
    }
    vkdist_close(listen_fd);
    vkblas_destroy_context(blas);
    vkDestroyDevice(dev, NULL);
    vkDestroyInstance(instance, NULL);
    return 0;
}

/* ===========================================================================
 * Client mode
 * ========================================================================== */

static int run_client(const char *ip, uint16_t port)
{
    static float A[XM * XK];
    static float B[XK * XN];
    static float C[XM * XN];
    static float ref[XM * XN];
    static float out[XM * XN];

    const int32_t m = XM, n = XN, k = XK;
    const float alpha = 1.0f, beta = 0.5f;
    const uint32_t elems = (uint32_t)(m * n);
    const int32_t lda = m, ldb = k, ldc = m;

    for (int32_t row = 0; row < m; row++)
        for (int32_t t = 0; t < k; t++)
            A[row + t * lda] = (float)(row + 1) * 0.01f;
    for (int32_t t = 0; t < k; t++)
        for (int32_t c = 0; c < n; c++)
            B[t + c * ldb] = (float)(1 + t + 16 * c);
    for (uint32_t i = 0; i < elems; i++)
        C[i] = 1.0f;

    compute_reference(A, lda, B, ldb, alpha, beta, C, ldc, ref, ldc, m, n, k);

    int fd = vkdist_client_connect(ip, port);
    if (fd < 0) {
        fprintf(stderr, "client: connect failed\n");
        return 1;
    }
    printf("client: connected to %s:%u\n", ip, port);

    uint64_t hA = 0, hB = 0, hC = 0;
    VkDeviceSize szA = (VkDeviceSize)sizeof(float) * m * k;
    VkDeviceSize szB = (VkDeviceSize)sizeof(float) * k * n;
    VkDeviceSize szC = (VkDeviceSize)sizeof(float) * m * n;

    VkResult vr = vkdist_register_buffer(fd, szA, &hA);
    if (vr == VK_SUCCESS) vr = vkdist_register_buffer(fd, szB, &hB);
    if (vr == VK_SUCCESS) vr = vkdist_register_buffer(fd, szC, &hC);
    if (vr != VK_SUCCESS) {
        fprintf(stderr, "client: register failed (%d)\n", (int)vr);
        return 1;
    }
    printf("client: registered A/B/C\n");

    if (vkdist_upload(fd, hA, A, 0, szA) != VK_SUCCESS ||
        vkdist_upload(fd, hB, B, 0, szB) != VK_SUCCESS ||
        vkdist_upload(fd, hC, C, 0, szC) != VK_SUCCESS) {
        fprintf(stderr, "client: upload failed\n");
        return 1;
    }
    printf("client: uploaded A/B/C (%zu KB)\n", (size_t)((szA + szB + szC) / 1024));

    vr = vkdist_sgemm(fd, m, n, k, &alpha, hA, hB, hC, &beta, lda, ldb, ldc);
    if (vr != VK_SUCCESS) {
        fprintf(stderr, "client: sgemm failed (%d)\n", (int)vr);
        return 1;
    }
    printf("client: remote sgemm dispatched (%dx%dx%d)\n", m, n, k);

    vr = vkdist_readback(fd, hC, 0, szC, out);
    if (vr != VK_SUCCESS) {
        fprintf(stderr, "client: readback failed (%d)\n", (int)vr);
        return 1;
    }
    printf("client: read back C\n");

    /* Verify: relative tolerance (values here are ~40k, so absolute f32
       rounding is ~0.01; compare against a magnitude-scaled epsilon). */
    int errors = 0;
    for (uint32_t i = 0; i < elems; i++) {
        float scale = fabsf(ref[i]) > 1.0f ? fabsf(ref[i]) : 1.0f;
        if (fabsf(out[i] - ref[i]) > 1e-3f * scale) {
            if (errors < 5)
                printf("  mismatch[%u]: got %.4f expected %.4f\n",
                       i, out[i], ref[i]);
            errors++;
        }
    }

    vkdist_close(fd);

    if (errors == 0) {
        printf("\n=== RESULT: PASS (cross-PC sgemm verified) ===\n");
        return 0;
    }
    printf("\n=== RESULT: FAIL (%d mismatches) ===\n", errors);
    return 1;
}

/* ===========================================================================
 * Master mode: connect multiple workers, query caps, partitioned GEMM
 * ========================================================================== */

static int run_master(const char *const *ips, int n_workers, uint16_t port)
{
    VkDistMaster *master = NULL;
    VkResult vr = vkdist_master_create(&master);
    if (vr != VK_SUCCESS) {
        fprintf(stderr, "master: create failed (%d)\n", (int)vr);
        return 1;
    }

    for (int i = 0; i < n_workers; i++) {
        VkDistCaps caps;
        vr = vkdist_master_add_worker(master, ips[i], "rr", port, &caps);
        if (vr != VK_SUCCESS) {
            fprintf(stderr, "master: add_worker %s failed (%d) -- SSH key gate\n",
                    ips[i], (int)vr);
            vkdist_master_destroy(master);
            return 1;
        }
        printf("master: worker[%d] %s -> %s (arch=%u vram=%llu MB subgroup=%u)\n",
               i, ips[i], caps.gpu_name, caps.arch_index,
               (unsigned long long)(caps.vram_total / (1024 * 1024)),
               caps.subgroup_size);
    }
    printf("master: %d worker(s) connected\n", n_workers);

    /* SSH gate rejection test: a host we have no key to must be refused. */
    VkResult gate = vkdist_verify_ssh_key("10.0.0.99", "rr");
    if (gate == VK_SUCCESS) {
        fprintf(stderr, "master: WARN 10.0.0.99 unexpectedly keyed\n");
    } else {
        printf("master: SSH gate correctly rejected untrusted host (VkResult=%d)\n",
               (int)gate);
    }

    static float A[XM * XK];
    static float B[XK * XN];
    static float C[XM * XN];
    static float ref[XM * XN];
    static float out[XM * XN];

    const int32_t m = XM, n = XN, k = XK;
    const float alpha = 1.0f, beta = 0.5f;
    const uint32_t elems = (uint32_t)(m * n);
    const int32_t lda = m, ldb = k, ldc = m;

    for (int32_t row = 0; row < m; row++)
        for (int32_t t = 0; t < k; t++)
            A[row + t * lda] = (float)(row + 1) * 0.01f;
    for (int32_t t = 0; t < k; t++)
        for (int32_t c = 0; c < n; c++)
            B[t + c * ldb] = (float)(1 + t + 16 * c);
    for (uint32_t i = 0; i < elems; i++)
        C[i] = 1.0f;

    compute_reference(A, lda, B, ldb, alpha, beta, C, ldc, ref, ldc, m, n, k);

    /* sgemm reads C for the beta term, then overwrites it with the result. */
    vr = vkdist_master_sgemm(master, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
    if (vr != VK_SUCCESS) {
        fprintf(stderr, "master: sgemm failed (%d)\n", (int)vr);
        vkdist_master_destroy(master);
        return 1;
    }
    printf("master: partitioned sgemm across %d worker(s) done\n", n_workers);

    int errors = 0;
    for (uint32_t i = 0; i < elems; i++) {
        float scale = fabsf(ref[i]) > 1.0f ? fabsf(ref[i]) : 1.0f;
        if (fabsf(C[i] - ref[i]) > 1e-3f * scale) {
            if (errors < 5)
                printf("  mismatch[%u]: got %.4f expected %.4f\n",
                       i, C[i], ref[i]);
            errors++;
        }
    }

    vkdist_master_destroy(master);

    if (errors == 0) {
        printf("\n=== RESULT: PASS (master/worker sgemm verified) ===\n");
        return 0;
    }
    printf("\n=== RESULT: FAIL (%d mismatches) ===\n", errors);
    return 1;
}

int main(int argc, char **argv)
{
    if (argc >= 3 && strcmp(argv[1], "server") == 0) {
        return run_server((uint16_t)atoi(argv[2]));
    }
    if (argc >= 4 && strcmp(argv[1], "client") == 0) {
        return run_client(argv[2], (uint16_t)atoi(argv[3]));
    }
    if (argc >= 4 && strcmp(argv[1], "master") == 0) {
        const char *ips[8];
        int n = 0;
        for (int i = 2; i < argc - 1 && n < 8; i++)
            ips[n++] = argv[i];
        return run_master(ips, n, (uint16_t)atoi(argv[argc - 1]));
    }
    fprintf(stderr, "usage: test_vkdist_xpc server <port>\n"
                    "       test_vkdist_xpc client <ip> <port>\n"
                    "       test_vkdist_xpc master <ip> [<ip> ...] <port>\n");
    return 1;
}
