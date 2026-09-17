/**
 * \file test_vkkv.c
 * \brief Public-API test harness for the VKKV library.
 *
 * Bootstraps a minimal Vulkan instance + physical/logical device, creates a
 * VkKVTransfer, and validates the host-fit / GPU-apply pipeline numerically:
 *
 *   (a) Builds synthetic calibration data: n=64 samples, src_dim=8,
 *       tgt_dim=8, n_heads=2. X_h ~ U(-1,1), Y_h = X_h * A_h + noise where
 *       A_0 is a random 8x8 matrix and A_1 = I.
 *   (b) vkkv_fit_cpu, then vkkv_apply per head on a held-out source block,
 *       comparing TARGET to X_heldout * A_h (relative error < 1e-1, which
 *       reflects the small ridge bias + calibration noise).
 *   (c) Pure-algebra path: re-fit on clean identity data (Y = X, no noise);
 *       apply must reproduce src within 1e-3 (validates the whole chain
 *       independent of the noise model).
 *   (d) Per-head selection: applying head 0 and head 1 to the SAME input
 *       yields distinct outputs (head 0 mapper != head 1 mapper).
 *
 * Exit status: 0 when all checks pass (or the harness is skipped for lack of
 * a compute-capable device), 1 on any real failure.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../include/vkkv/vkkv.h"
#include "../include/vkruntime/vkruntime.h"

/* ===========================================================================
 * Test parameters
 * ========================================================================== */

#define N_HEADS      2u   /**< Number of heads to map.                        */
#define SRC_DIM      8u   /**< Source feature dimension.                      */
#define TGT_DIM      8u   /**< Target feature dimension.                      */
#define N_SAMPLES    64u  /**< Calibration samples per head.                  */
#define N_HELDOUT    32u  /**< Held-out rows (sequence positions).            */
#define RIDGE_LAMBDA 1e-4f /**< Small ridge penalty (negligible bias).        */
#define NOISE_STD    0.02f /**< Calibration noise amplitude.                  */
#define TOL_NOISE    0.1f  /**< Relative error budget for the noisy case.     */
#define TOL_ALGEBRA  1e-3f /**< Absolute tolerance for the pure-algebra case. */

/* ===========================================================================
 * Deterministic PRNG (xorshift64) + Gaussian sampler
 * ========================================================================== */

static uint64_t g_seed = 0x9E3779B97F4A7C15ull;

static double rnd_u01(void)
{
    g_seed ^= g_seed << 13;
    g_seed ^= g_seed >> 7;
    g_seed ^= g_seed << 17;
    return (double)(g_seed >> 11) / 9007199254740992.0; /* [0,1) */
}

static double rnd_uniform(double lo, double hi)
{
    return lo + (hi - lo) * rnd_u01();
}

static double rnd_gauss(void)
{
    double u1 = rnd_u01();
    if (u1 < 1e-12) u1 = 1e-12;
    double u2 = rnd_u01();
    return sqrt(-2.0 * log(u1)) * cos(6.2831853071795864769 * u2);
}

/* ===========================================================================
 * Small dense-matrix helpers (row-major float)
 * ========================================================================== */

/**
 * \brief C = A * B with A [n x d], B [d x m], C [n x m], all row-major.
 */
static void mat_mul(const float *A, const float *B, float *C,
                    uint32_t n, uint32_t d, uint32_t m)
{
    for (uint32_t r = 0; r < n; r++) {
        for (uint32_t c = 0; c < m; c++) {
            double s = 0.0;
            for (uint32_t k = 0; k < d; k++) {
                s += (double)A[(size_t)r * d + k] * (double)B[(size_t)k * m + c];
            }
            C[(size_t)r * m + c] = (float)s;
        }
    }
}

/**
 * \brief Relative Frobenius error ||a - b||_F / ||b||_F.
 */
static double rel_fro_err(const float *a, const float *b, uint32_t count)
{
    double na = 0.0, nb = 0.0;
    for (uint32_t i = 0; i < count; i++) {
        double d = (double)a[i] - (double)b[i];
        na += d * d;
        nb += (double)b[i] * (double)b[i];
    }
    return sqrt(na) / (sqrt(nb) + 1e-300);
}

/**
 * \brief Maximum absolute error between two float arrays.
 */
static double max_abs_err(const float *a, const float *b, uint32_t count)
{
    double m = 0.0;
    for (uint32_t i = 0; i < count; i++) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > m) m = d;
    }
    return m;
}

/* ===========================================================================
 * Vulkan bootstrap (mirrors tests/test_vkmath.c)
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

    VkPhysicalDeviceFeatures features;
    memset(&features, 0, sizeof(features));
    features.shaderInt64 = VK_TRUE;

    VkDeviceCreateInfo create_info;
    memset(&create_info, 0, sizeof(create_info));
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.queueCreateInfoCount = 1;
    create_info.pQueueCreateInfos = &queue_info;
    create_info.enabledExtensionCount = 0;
    create_info.ppEnabledExtensionNames = NULL;
    create_info.pEnabledFeatures = &features;
    return vkCreateDevice(physical_device, &create_info, NULL, out_device);
}

static VkResult create_command_pool_and_buffer(VkDevice device,
                                               VkCommandPool *out_pool,
                                               VkCommandBuffer *out_cmd)
{
    VkCommandPoolCreateInfo pool_info;
    memset(&pool_info, 0, sizeof(pool_info));
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = 0;

    VkResult r = vkCreateCommandPool(device, &pool_info, NULL, out_pool);
    if (r != VK_SUCCESS) return r;

    VkCommandBufferAllocateInfo alloc_info;
    memset(&alloc_info, 0, sizeof(alloc_info));
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = *out_pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;
    r = vkAllocateCommandBuffers(device, &alloc_info, out_cmd);
    if (r != VK_SUCCESS) {
        vkDestroyCommandPool(device, *out_pool, NULL);
        *out_pool = VK_NULL_HANDLE;
    }
    return r;
}

/**
 * \brief Record one apply for head h into cmd, submit, and wait.
 */
static int run_apply(VkKVTransfer *t, VkQueue queue, VkCommandPool pool,
                     VkCommandBuffer cmd, uint32_t h, VkBuffer src, VkBuffer dst)
{
    VkCommandBufferBeginInfo begin_info;
    memset(&begin_info, 0, sizeof(begin_info));
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VkResult r = vkBeginCommandBuffer(cmd, &begin_info);
    if (r != VK_SUCCESS) return 0;
    r = vkkv_apply(t, cmd, h, src, N_HELDOUT, dst);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "  vkkv_apply head %u failed (VkResult=%d)\n",
                (unsigned)h, (int)r);
        return 0;
    }
    r = vkEndCommandBuffer(cmd);
    if (r != VK_SUCCESS) return 0;

    VkSubmitInfo submit_info;
    memset(&submit_info, 0, sizeof(submit_info));
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;

    r = vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);
    if (r != VK_SUCCESS) return 0;
    r = vkQueueWaitIdle(queue);
    if (r != VK_SUCCESS) return 0;
    vkResetCommandBuffer(cmd, 0); /* return to initial state for vkr_download */
    (void)pool;
    return 1;
}

/* ===========================================================================
 * main
 * ========================================================================== */

int main(void)
{
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkRuntime *rt = NULL;
    VkKVTransfer *tf = NULL;
    VkBuffer src_buf = VK_NULL_HANDLE, dst_buf = VK_NULL_HANDLE;
    VkDeviceMemory src_mem = VK_NULL_HANDLE, dst_mem = VK_NULL_HANDLE;

    int pass = 1;
    VkResult r;

    /* Data */
    float *X[N_HEADS], *Y[N_HEADS];
    float A[N_HEADS][SRC_DIM * TGT_DIM];
    float *X_held[N_HEADS];
    float *y_expected[N_HEADS];
    float *Xc[N_HEADS], *Yc[N_HEADS];
    float y_got0[N_HELDOUT * TGT_DIM];
    float y_got1[N_HELDOUT * TGT_DIM];
    float y_got[N_HELDOUT * TGT_DIM];
    memset(y_got0, 0, sizeof(y_got0));
    memset(y_got1, 0, sizeof(y_got1));
    memset(y_got, 0, sizeof(y_got));
    memset(A, 0, sizeof(A));

    for (uint32_t i = 0; i < N_HEADS; i++) {
        X[i] = NULL; Y[i] = NULL; X_held[i] = NULL;
        y_expected[i] = NULL; Xc[i] = NULL; Yc[i] = NULL;
    }

    /* ── 1. Instance ────────────────────────────────────────────────────── */
    r = create_instance("test_vkkv", &instance);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkkv: vkCreateInstance failed (%d)\n", (int)r);
        return 1;
    }

    /* ── 2. Physical device ─────────────────────────────────────────────── */
    r = find_physical_device(instance, &pd);
    if (r != VK_SUCCESS) {
        printf("test_vkkv: SKIP (no physical device found)\n");
        vkDestroyInstance(instance, NULL);
        return 0;
    }

    /* ── 3. Queue family gate ───────────────────────────────────────────── */
    if (queue_family_supports_compute(pd, 0) == VK_FALSE) {
        printf("test_vkkv: SKIP (queue family 0 lacks compute)\n");
        vkDestroyInstance(instance, NULL);
        return 0;
    }

    /* ── 4. Logical device + queue ──────────────────────────────────────── */
    r = create_device(pd, &device);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkkv: vkCreateDevice failed (%d)\n", (int)r);
        goto cleanup;
    }
    vkGetDeviceQueue(device, 0, 0, &queue);

    /* ── 5. Command pool + buffer (test's own; apply recording) ─────────── */
    r = create_command_pool_and_buffer(device, &pool, &cmd);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkkv: command pool/buffer failed (%d)\n", (int)r);
        goto cleanup;
    }

    /* ── 6. Test runtime (buffer alloc + staging) ───────────────────────── */
    r = vkr_create_runtime(pd, device, queue, &rt);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkkv: vkr_create_runtime failed (%d)\n", (int)r);
        goto cleanup;
    }

    /* ── 7. Transfer ────────────────────────────────────────────────────── */
    r = vkkv_create_transfer(pd, device, N_HEADS, SRC_DIM, TGT_DIM,
                             RIDGE_LAMBDA, &tf);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkkv: vkkv_create_transfer failed (%d)\n", (int)r);
        goto cleanup;
    }

    /* ── 8. Device buffers for held-out source / target ─────────────────── */
    {
        VkDeviceSize bytes = (VkDeviceSize)N_HELDOUT * SRC_DIM * sizeof(float);
        VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                 | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                                 | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        r = vkr_malloc(rt, bytes, usage, &src_buf, &src_mem);
        if (r != VK_SUCCESS) { pass = 0; goto cleanup; }
        r = vkr_malloc(rt, bytes, usage, &dst_buf, &dst_mem);
        if (r != VK_SUCCESS) { pass = 0; goto cleanup; }
    }

    /* ── 9. (a) Synthetic calibration data ──────────────────────────────── */
    for (uint32_t h = 0; h < N_HEADS; h++) {
        X[h] = (float *)malloc((size_t)N_SAMPLES * SRC_DIM * sizeof(float));
        Y[h] = (float *)malloc((size_t)N_SAMPLES * TGT_DIM * sizeof(float));
        X_held[h] = (float *)malloc((size_t)N_HELDOUT * SRC_DIM * sizeof(float));
        y_expected[h] = (float *)malloc((size_t)N_HELDOUT * TGT_DIM * sizeof(float));
        Xc[h] = (float *)malloc((size_t)N_SAMPLES * SRC_DIM * sizeof(float));
        Yc[h] = (float *)malloc((size_t)N_SAMPLES * TGT_DIM * sizeof(float));
        if (!X[h] || !Y[h] || !X_held[h] || !y_expected[h] || !Xc[h] || !Yc[h]) {
            fprintf(stderr, "test_vkkv: host allocation failed\n");
            pass = 0;
            goto cleanup;
        }
    }

    /* Known per-head transforms: A0 random, A1 = I (identity). */
    for (uint32_t k = 0; k < SRC_DIM; k++) {
        for (uint32_t j = 0; j < TGT_DIM; j++) {
            A[0][k * TGT_DIM + j] = (float)rnd_uniform(-0.5, 0.5);
            A[1][k * TGT_DIM + j] = (k == j) ? 1.0f : 0.0f;
        }
    }

    /* Calibration: X ~ U(-1,1); Y_h = X * A_h + gauss * NOISE_STD. */
    for (uint32_t h = 0; h < N_HEADS; h++) {
        for (uint32_t s = 0; s < N_SAMPLES; s++) {
            for (uint32_t k = 0; k < SRC_DIM; k++) {
                X[h][s * SRC_DIM + k] = (float)rnd_uniform(-1.0, 1.0);
            }
        }
        mat_mul(X[h], A[h], Y[h], N_SAMPLES, SRC_DIM, TGT_DIM);
        for (uint32_t s = 0; s < N_SAMPLES * TGT_DIM; s++) {
            Y[h][s] += (float)(rnd_gauss() * NOISE_STD);
        }
    }

    /* Held-out source block (same for every head; expected = X_held * A_h). */
    for (uint32_t h = 0; h < N_HEADS; h++) {
        for (uint32_t s = 0; s < N_HELDOUT; s++) {
            for (uint32_t k = 0; k < SRC_DIM; k++) {
                X_held[h][s * SRC_DIM + k] = (float)rnd_uniform(-1.0, 1.0);
            }
        }
        mat_mul(X_held[h], A[h], y_expected[h], N_HELDOUT, SRC_DIM, TGT_DIM);
    }

    /* ── 10. (b) Fit + apply, compare to the true transform ─────────────── */
    printf("test_vkkv: fitting %u heads (%u x %u) with lambda=%g\n",
           (unsigned)N_HEADS, (unsigned)SRC_DIM, (unsigned)TGT_DIM,
           (double)RIDGE_LAMBDA);
    r = vkkv_fit_cpu(tf, (const float *const *)X, (const float *const *)Y,
                     N_SAMPLES);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkkv: vkkv_fit_cpu failed (%d)\n", (int)r);
        pass = 0;
        goto cleanup;
    }

    for (uint32_t h = 0; h < N_HEADS; h++) {
        VkDeviceSize bytes = (VkDeviceSize)N_HELDOUT * SRC_DIM * sizeof(float);
        r = vkr_upload(rt, cmd, queue, X_held[h], src_buf, 0, bytes);
        if (r != VK_SUCCESS) {
            fprintf(stderr, "test_vkkv: upload failed (%d)\n", (int)r);
            pass = 0;
            goto cleanup;
        }
        if (!run_apply(tf, queue, pool, cmd, h, src_buf, dst_buf)) {
            pass = 0;
            goto cleanup;
        }
        r = vkr_download(rt, cmd, queue, dst_buf, 0, y_got, bytes);
        if (r != VK_SUCCESS) {
            fprintf(stderr, "test_vkkv: download failed (%d)\n", (int)r);
            pass = 0;
            goto cleanup;
        }

        double rel = rel_fro_err(y_got, y_expected[h], N_HELDOUT * TGT_DIM);
        int ok = rel < (double)TOL_NOISE;
        printf("  apply head %u vs X*A%u : rel_err=%.4e %s\n",
               (unsigned)h, (unsigned)h, rel, ok ? "PASS" : "FAIL");
        pass &= ok;

        if (h == 0) memcpy(y_got0, y_got, sizeof(y_got0));
        else        memcpy(y_got1, y_got, sizeof(y_got1));
    }

    /* ── 11. (d) Per-head selection: distinct mappers → distinct outputs ── */
    {
        double diff = 0.0;
        for (uint32_t i = 0; i < N_HELDOUT * TGT_DIM; i++) {
            double d = (double)y_got0[i] - (double)y_got1[i];
            diff += d * d;
        }
        diff = sqrt(diff);
        int ok = diff > 1.0;
        printf("  head0 vs head1 on same input : frobenius diff=%.4e %s\n",
               diff, ok ? "PASS" : "FAIL");
        pass &= ok;
    }

    /* ── 12. (c) Pure-algebra path: clean identity fit, apply ~ identity ── */
    for (uint32_t h = 0; h < N_HEADS; h++) {
        for (uint32_t s = 0; s < N_SAMPLES; s++) {
            for (uint32_t k = 0; k < SRC_DIM; k++) {
                Xc[h][s * SRC_DIM + k] = (float)rnd_uniform(-1.0, 1.0);
                Yc[h][s * TGT_DIM + k] = Xc[h][s * SRC_DIM + k]; /* A = I */
            }
        }
    }
    r = vkkv_fit_cpu(tf, (const float *const *)Xc, (const float *const *)Yc,
                     N_SAMPLES);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkkv: refit failed (%d)\n", (int)r);
        pass = 0;
        goto cleanup;
    }

    {
        VkDeviceSize bytes = (VkDeviceSize)N_HELDOUT * SRC_DIM * sizeof(float);
        r = vkr_upload(rt, cmd, queue, X_held[0], src_buf, 0, bytes);
        if (r != VK_SUCCESS) { pass = 0; goto cleanup; }
        if (!run_apply(tf, queue, pool, cmd, 0, src_buf, dst_buf)) {
            pass = 0;
            goto cleanup;
        }
        r = vkr_download(rt, cmd, queue, dst_buf, 0, y_got, bytes);
        if (r != VK_SUCCESS) { pass = 0; goto cleanup; }

        double maxerr = max_abs_err(y_got, X_held[0], N_HELDOUT * SRC_DIM);
        int ok = maxerr < (double)TOL_ALGEBRA;
        printf("  identity apply reproduces src : max_abs_err=%.4e %s\n",
               maxerr, ok ? "PASS" : "FAIL");
        pass &= ok;
    }

cleanup:
    for (uint32_t i = 0; i < N_HEADS; i++) {
        free(X[i]); free(Y[i]); free(X_held[i]);
        free(y_expected[i]); free(Xc[i]); free(Yc[i]);
    }
    if (src_buf) vkr_free(rt, src_buf, src_mem);
    if (dst_buf) vkr_free(rt, dst_buf, dst_mem);
    if (tf) vkkv_destroy_transfer(tf);
    if (rt) vkr_destroy_runtime(rt);
    if (cmd) vkFreeCommandBuffers(device, pool, 1, &cmd);
    if (pool) vkDestroyCommandPool(device, pool, NULL);
    if (device) vkDestroyDevice(device, NULL);
    if (instance) vkDestroyInstance(instance, NULL);

    printf("test_vkkv: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
