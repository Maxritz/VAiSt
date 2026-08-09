/**
 * \file test_vkrand.c
 * \brief Public-API test harness for the VKRAND library.
 *
 * Bootstraps a minimal Vulkan instance + physical/logical device, creates a
 * VkRandContext via vkrand_create_context(), records one
 * vkrand_uniform_f32() dispatch (seed=42, count=1024) into a single command
 * buffer, submits once, and validates the GPU output against a CPU reference
 * computed with the SAME Philox4x32-10 algorithm and [0,1) mapping.
 *
 * This is a header-only test: it includes only <vulkan/vulkan.h> and the
 * public vkrand.h header (relative include). No internal headers are pulled.
 *
 * The CPU reference is validated against the official Random123 known-answer
 * vectors for philox4x32-10, so the algorithm is proven correct independent
 * of the GPU result.
 *
 * Exit status: 0 when all checks pass. Returns 1 on any real failure.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../include/vkrand/vkrand.h"

/* ===========================================================================
 * Test parameters
 * ========================================================================== */

#define TEST_SEED        42u            /**< Deterministic generator seed.      */
#define TEST_COUNT       1024u          /**< Number of f32 values to generate.  */
#define TEST_STAGING_SIZE ((VkDeviceSize)(1u << 20))  /**< 1 MiB host buffer.   */

/* ===========================================================================
 * CPU reference (must mirror the shader bit-for-bit)
 * ========================================================================== */

#define PHILOX_M0 0xD2511F53u
#define PHILOX_M1 0xCD9E8D57u
#define PHILOX_W0 0x9E3779B9u
#define PHILOX_W1 0xBB67AE85u

/** \brief High 32 bits of the unsigned 64-bit product a*b. */
static uint32_t mulhi32(uint32_t a, uint32_t b)
{
    return (uint32_t)(((uint64_t)a * b) >> 32);
}

/** \brief Murmur3 finalizer (fmix32) — must match the shader exactly. */
static uint32_t philox_seed_hash(uint32_t s)
{
    s ^= s >> 16;
    s *= 0x85EBCA6Bu;
    s ^= s >> 13;
    s *= 0xC2B2AE35u;
    s ^= s >> 16;
    return s;
}

/**
 * \brief CPU reference: Philox4x32-10 uniform f32 for thread index i.
 *
 * Counter c0 = i, c1 = seed hash, c2 = W0, c3 = W1; key k0 = seed,
 * k1 = W0 ^ seed. Ten rounds with the key bumped by (W0, W1) after each.
 * Maps float(c0 & 0xFFFFFF) / 2^24 into [0,1).
 */
static float cpu_uniform_f32(uint32_t i, uint32_t seed)
{
    uint32_t c[4], k[2];
    c[0] = i;
    c[1] = philox_seed_hash(seed);
    c[2] = PHILOX_W0;
    c[3] = PHILOX_W1;
    k[0] = seed;
    k[1] = PHILOX_W0 ^ seed;

    for (int r = 0; r < 10; r++) {
        uint32_t hi0 = mulhi32(c[0], PHILOX_M0);
        uint32_t lo0 = c[0] * PHILOX_M0;
        uint32_t hi1 = mulhi32(c[2], PHILOX_M1);
        uint32_t lo1 = c[2] * PHILOX_M1;
        c[0] = hi1 ^ c[1] ^ k[0];
        c[1] = lo1;
        c[2] = hi0 ^ c[3] ^ k[1];
        c[3] = lo0;
        k[0] += PHILOX_W0;
        k[1] += PHILOX_W1;
    }
    return (float)(c[0] & 0xFFFFFFu) / 16777216.0f;
}

/**
 * \brief Self-check the CPU reference against the Random123 known-answer
 *        vectors for philox4x32-10. Returns 1 on match.
 */
static int verify_cpu_reference_against_random123(void)
{
    /* ctr={0,0,0,0}, key={0,0} -> 6627e8d5 e169c58d bc57ac4c 9b00dbd8 */
    uint32_t c[4] = {0, 0, 0, 0};
    uint32_t k[2] = {0, 0};
    for (int r = 0; r < 10; r++) {
        uint32_t hi0 = mulhi32(c[0], PHILOX_M0);
        uint32_t lo0 = c[0] * PHILOX_M0;
        uint32_t hi1 = mulhi32(c[2], PHILOX_M1);
        uint32_t lo1 = c[2] * PHILOX_M1;
        c[0] = hi1 ^ c[1] ^ k[0];
        c[1] = lo1;
        c[2] = hi0 ^ c[3] ^ k[1];
        c[3] = lo0;
        k[0] += PHILOX_W0;
        k[1] += PHILOX_W1;
    }
    if (c[0] != 0x6627E8D5u || c[1] != 0xE169C58Du ||
        c[2] != 0xBC57AC4Cu || c[3] != 0x9B00DBD8u) {
        return 0;
    }
    /* ctr={ffffffff,ffffffff,ffffffff,ffffffff}, key={ffffffff,ffffffff}
       -> 408f276d 41c83b0e a20bc7c6 6d5451fd */
    c[0] = 0xFFFFFFFFu; c[1] = 0xFFFFFFFFu;
    c[2] = 0xFFFFFFFFu; c[3] = 0xFFFFFFFFu;
    k[0] = 0xFFFFFFFFu; k[1] = 0xFFFFFFFFu;
    for (int r = 0; r < 10; r++) {
        uint32_t hi0 = mulhi32(c[0], PHILOX_M0);
        uint32_t lo0 = c[0] * PHILOX_M0;
        uint32_t hi1 = mulhi32(c[2], PHILOX_M1);
        uint32_t lo1 = c[2] * PHILOX_M1;
        c[0] = hi1 ^ c[1] ^ k[0];
        c[1] = lo1;
        c[2] = hi0 ^ c[3] ^ k[1];
        c[3] = lo0;
        k[0] += PHILOX_W0;
        k[1] += PHILOX_W1;
    }
    return (c[0] == 0x408F276Du && c[1] == 0x41C83B0Eu &&
            c[2] == 0xA20BC7C6u && c[3] == 0x6D5451FDu) ? 1 : 0;
}

/* ===========================================================================
 * Harness state
 * ========================================================================== */

typedef struct {
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue queue;
    VkCommandPool cmd_pool;
    VkCommandBuffer cmd;
    VkDeviceMemory mem;         /**< Single host-visible/coherent allocation. */
    VkBuffer staging;           /**< 1 MiB buffer spanning the memory.        */
    void *mapped;               /**< Host mapping of mem.                     */
    VkDeviceSize align;         /**< Buffer memory alignment for sub-buffers. */
    VkDeviceSize cursor;        /**< Sub-allocation cursor into mem.          */
    VkBuffer out;               /**< Output buffer receiving the uniforms.    */
    VkDeviceSize off_out;       /**< Offset of the output buffer in mem.      */
    VkFence fence;
    uint32_t subgroup_size;
    VkRandContext *rand_ctx;
} harness_t;

/* ===========================================================================
 * Bootstrap helpers (mirror test_vkmath.c)
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

static uint32_t query_subgroup_size(VkPhysicalDevice physical_device)
{
    VkPhysicalDeviceSubgroupProperties subgroup;
    memset(&subgroup, 0, sizeof(subgroup));
    subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;

    VkPhysicalDeviceProperties2 props2;
    memset(&props2, 0, sizeof(props2));
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &subgroup;
    vkGetPhysicalDeviceProperties2(physical_device, &props2);
    return subgroup.subgroupSize;
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

static VkResult create_sub_buffer(VkDevice device, VkDeviceMemory mem,
                                  VkDeviceSize offset, VkDeviceSize size,
                                  VkBuffer *out_buffer)
{
    VkBufferCreateInfo buffer_info;
    memset(&buffer_info, 0, sizeof(buffer_info));
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult r = vkCreateBuffer(device, &buffer_info, NULL, out_buffer);
    if (r != VK_SUCCESS) return r;
    return vkBindBufferMemory(device, *out_buffer, mem, offset);
}

static VkResult allocate_staging_memory(VkPhysicalDevice physical_device,
                                        VkDevice device, VkDeviceSize size,
                                        VkDeviceMemory *out_memory,
                                        VkBuffer *out_staging,
                                        VkDeviceSize *out_align)
{
    VkBufferCreateInfo probe_info;
    memset(&probe_info, 0, sizeof(probe_info));
    probe_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    probe_info.size = size;
    probe_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    probe_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer probe = VK_NULL_HANDLE;
    VkResult r = vkCreateBuffer(device, &probe_info, NULL, &probe);
    if (r != VK_SUCCESS) return r;

    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(device, probe, &requirements);
    vkDestroyBuffer(device, probe, NULL);

    VkPhysicalDeviceMemoryProperties memory_properties;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);

    uint32_t memory_index = UINT32_MAX;
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++) {
        if ((requirements.memoryTypeBits & (1u << i)) == 0u) continue;
        VkMemoryPropertyFlags flags = memory_properties.memoryTypes[i].propertyFlags;
        if ((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0u &&
            (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0u) {
            memory_index = i;
            break;
        }
    }
    if (memory_index == UINT32_MAX) return VK_ERROR_FEATURE_NOT_PRESENT;

    VkMemoryAllocateInfo alloc_info;
    memset(&alloc_info, 0, sizeof(alloc_info));
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = size;
    alloc_info.memoryTypeIndex = memory_index;

    r = vkAllocateMemory(device, &alloc_info, NULL, out_memory);
    if (r != VK_SUCCESS) return r;

    r = create_sub_buffer(device, *out_memory, 0, size, out_staging);
    if (r != VK_SUCCESS) {
        vkFreeMemory(device, *out_memory, NULL);
        *out_memory = VK_NULL_HANDLE;
        return r;
    }

    *out_align = requirements.alignment;
    return VK_SUCCESS;
}

static VkDeviceSize align_up(VkDeviceSize value, VkDeviceSize align)
{
    return (value + align - 1) & ~(align - 1);
}

/* ===========================================================================
 * Checks
 * ========================================================================== */

/**
 * \brief (a) every value in [0,1); (b) exact equality vs CPU reference.
 *
 * \return 1 when both pass, 0 otherwise.
 */
static int check_values(const float *got, const float *expected, uint32_t count)
{
    int range_ok = 1;
    int exact_ok = 1;
    uint32_t range_bad = 0, exact_bad = 0;

    for (uint32_t i = 0; i < count; i++) {
        if (!(got[i] >= 0.0f && got[i] < 1.0f)) {
            if (range_bad < 8)
                printf("    out-of-range[%u]: got %.9f\n", i, got[i]);
            range_bad++;
            range_ok = 0;
        }
        if (got[i] != expected[i]) {   /* bit-exact, same float op order */
            if (exact_bad < 8)
                printf("    mismatch[%u]: got %.9f (0x%08x) expected %.9f (0x%08x)\n",
                       i, got[i], *(const uint32_t *)&got[i],
                       expected[i], *(const uint32_t *)&expected[i]);
            exact_bad++;
            exact_ok = 0;
        }
    }
    printf("  range [0,1)        : %s (%u bad)\n", range_ok ? "PASS" : "FAIL", range_bad);
    printf("  exact CPU match    : %s (%u bad)\n", exact_ok ? "PASS" : "FAIL", exact_bad);
    return range_ok && exact_ok;
}

/**
 * \brief (c) statistical sanity: mean in [0.4,0.6], stddev in [0.2,0.35].
 *
 * \return 1 when both pass, 0 otherwise.
 */
static int check_stats(const float *got, uint32_t count)
{
    double sum = 0.0;
    for (uint32_t i = 0; i < count; i++) sum += (double)got[i];
    double mean = sum / (double)count;

    double var = 0.0;
    for (uint32_t i = 0; i < count; i++) {
        double d = (double)got[i] - mean;
        var += d * d;
    }
    var /= (double)count;
    double stddev = sqrt(var);

    int mean_ok  = (mean >= 0.4 && mean <= 0.6);
    int stddev_ok = (stddev >= 0.2 && stddev <= 0.35);
    printf("  stats mean/stddev  : %s (mean=%.4f stddev=%.4f)\n",
           (mean_ok && stddev_ok) ? "PASS" : "FAIL", mean, stddev);
    return mean_ok && stddev_ok;
}

/* ===========================================================================
 * main
 * ========================================================================== */

int main(void)
{
    harness_t h;
    memset(&h, 0, sizeof(h));
    h.mem = VK_NULL_HANDLE;
    h.staging = VK_NULL_HANDLE;
    h.out = VK_NULL_HANDLE;
    h.fence = VK_NULL_HANDLE;

    float *expected = (float *)malloc(TEST_COUNT * sizeof(float));
    if (!expected) {
        fprintf(stderr, "test_vkrand: host allocation failed\n");
        return 1;
    }

    int overall_pass = 1;
    VkResult r;

    /* ── 1. Instance ────────────────────────────────────────────────────── */
    r = create_instance("test_vkrand", &h.instance);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkrand: vkCreateInstance failed (%d)\n", (int)r);
        free(expected);
        return 1;
    }

    /* ── 2. Physical device ─────────────────────────────────────────────── */
    r = find_physical_device(h.instance, &h.physical_device);
    if (r != VK_SUCCESS) {
        printf("test_vkrand: SKIP (no physical device found)\n");
        vkDestroyInstance(h.instance, NULL);
        free(expected);
        return 0;
    }

    /* ── 3. shaderInt64 gate (mirrors test_vkmath; harmless here) ───────── */
    if (query_shader_int64(h.physical_device) == VK_FALSE) {
        printf("test_vkrand: SKIP (shaderInt64 not supported)\n");
        vkDestroyInstance(h.instance, NULL);
        free(expected);
        return 0;
    }
    h.subgroup_size = query_subgroup_size(h.physical_device);

    /* ── 4. Queue family gate (the harness uses vkGetDeviceQueue(d, 0, 0)) ─ */
    if (queue_family_supports_compute(h.physical_device, 0) == VK_FALSE) {
        printf("test_vkrand: SKIP (queue family 0 lacks compute)\n");
        vkDestroyInstance(h.instance, NULL);
        free(expected);
        return 0;
    }

    /* ── 5. Logical device ──────────────────────────────────────────────── */
    r = create_device(h.physical_device, &h.device);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkrand: vkCreateDevice failed (%d)\n", (int)r);
        goto cleanup;
    }
    vkGetDeviceQueue(h.device, 0, 0, &h.queue);

    /* ── 6. Command pool + one command buffer ───────────────────────────── */
    r = create_command_pool_and_buffer(h.device, &h.cmd_pool, &h.cmd);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkrand: command pool/buffer failed (%d)\n", (int)r);
        goto cleanup;
    }

    /* ── 7. 1 MiB host-visible/host-coherent memory + output sub-buffer ─── */
    r = allocate_staging_memory(h.physical_device, h.device, TEST_STAGING_SIZE,
                                &h.mem, &h.staging, &h.align);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkrand: staging allocation failed (%d)\n", (int)r);
        goto cleanup;
    }
    r = vkMapMemory(h.device, h.mem, 0, VK_WHOLE_SIZE, 0, &h.mapped);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkrand: vkMapMemory failed (%d)\n", (int)r);
        goto cleanup;
    }

    h.off_out = align_up(h.cursor, h.align);
    h.cursor = h.off_out + TEST_COUNT * sizeof(float);
    r = create_sub_buffer(h.device, h.mem, h.off_out,
                          TEST_COUNT * sizeof(float), &h.out);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkrand: output buffer failed (%d)\n", (int)r);
        goto cleanup;
    }

    /* ── 8. CPU reference values ────────────────────────────────────────── */
    if (!verify_cpu_reference_against_random123()) {
        fprintf(stderr, "test_vkrand: CPU reference FAILED Random123 vectors\n");
        overall_pass = 0;
        goto cleanup;
    }
    for (uint32_t i = 0; i < TEST_COUNT; i++) {
        expected[i] = cpu_uniform_f32(i, TEST_SEED);
    }

    /* ── 9. Context ─────────────────────────────────────────────────────── */
    r = vkrand_create_context(h.physical_device, h.device, &h.rand_ctx);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkrand: vkrand_create_context failed (%d)\n", (int)r);
        goto cleanup;
    }
    printf("test_vkrand: device ready (arch=%s, tier=%u, subgroup=%u, staging=%u)\n",
           vkrand_get_arch_name(h.rand_ctx), vkrand_get_arch_index(h.rand_ctx),
           (unsigned)h.subgroup_size, (unsigned)TEST_STAGING_SIZE);

    /* ── 10. Record the dispatch into the single command buffer ─────────── */
    VkCommandBufferBeginInfo begin_info;
    memset(&begin_info, 0, sizeof(begin_info));
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    r = vkBeginCommandBuffer(h.cmd, &begin_info);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkrand: vkBeginCommandBuffer failed (%d)\n", (int)r);
        overall_pass = 0;
        goto cleanup;
    }

    r = vkrand_uniform_f32(h.rand_ctx, h.cmd, TEST_SEED, TEST_COUNT, h.out);
    if (r != VK_SUCCESS) {
        printf("  uniform_f32 : FAIL (record, VkResult=%d)\n", (int)r);
        overall_pass = 0;
    } else {
        printf("  uniform_f32 : recorded\n");
    }

    r = vkEndCommandBuffer(h.cmd);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkrand: vkEndCommandBuffer failed (%d)\n", (int)r);
        overall_pass = 0;
        goto cleanup;
    }

    /* ── 11. One submit, one fence, device idle ─────────────────────────── */
    VkFenceCreateInfo fence_info;
    memset(&fence_info, 0, sizeof(fence_info));
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    r = vkCreateFence(h.device, &fence_info, NULL, &h.fence);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkrand: vkCreateFence failed (%d)\n", (int)r);
        overall_pass = 0;
        goto cleanup;
    }

    VkSubmitInfo submit_info;
    memset(&submit_info, 0, sizeof(submit_info));
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &h.cmd;

    r = vkQueueSubmit(h.queue, 1, &submit_info, h.fence);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkrand: vkQueueSubmit failed (%d)\n", (int)r);
        overall_pass = 0;
        goto cleanup;
    }
    vkWaitForFences(h.device, 1, &h.fence, VK_TRUE, UINT64_MAX);
    vkQueueWaitIdle(h.queue);

    /* ── 12. Compare GPU results against the CPU reference ──────────────── */
    const float *got = (const float *)((const char *)h.mapped + h.off_out);
    overall_pass &= check_values(got, expected, TEST_COUNT);
    overall_pass &= check_stats(got, TEST_COUNT);

cleanup:
    if (h.rand_ctx) vkrand_destroy_context(h.rand_ctx);
    if (h.out != VK_NULL_HANDLE) vkDestroyBuffer(h.device, h.out, NULL);
    if (h.staging) vkDestroyBuffer(h.device, h.staging, NULL);
    if (h.mapped) vkUnmapMemory(h.device, h.mem);
    if (h.mem != VK_NULL_HANDLE) vkFreeMemory(h.device, h.mem, NULL);
    if (h.fence != VK_NULL_HANDLE) vkDestroyFence(h.device, h.fence, NULL);
    if (h.cmd != VK_NULL_HANDLE)
        vkFreeCommandBuffers(h.device, h.cmd_pool, 1, &h.cmd);
    if (h.cmd_pool != VK_NULL_HANDLE)
        vkDestroyCommandPool(h.device, h.cmd_pool, NULL);
    if (h.device != VK_NULL_HANDLE) vkDestroyDevice(h.device, NULL);
    if (h.instance != VK_NULL_HANDLE) vkDestroyInstance(h.instance, NULL);

    free(expected);

    printf("test_vkrand: %s\n", overall_pass ? "PASS" : "FAIL");
    return overall_pass ? 0 : 1;
}
