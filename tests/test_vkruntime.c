/**
 * \file test_vkruntime.c
 * \brief Public-API test harness for the VKRuntime library.
 *
 * Bootstraps a minimal Vulkan instance + physical/logical device, creates a
 * VkRuntime via vkr_create_runtime(), then validates:
 *   (a) capability queries (arch index, subgroup size, device handle);
 *   (b) pooled allocator: 1 MiB + 1000 varying-size sub-allocations, all
 *       distinct, then vkr_free of everything;
 *   (c) upload/download round-trip through a device-local buffer;
 *   (d) command-pool + descriptor-pool helpers;
 *   (e) vkr_detect_capabilities() consistency with the runtime's arch
 *       queries, and nonzero subgroup size;
 *   (f) vkr_create_descriptor_pool / vkr_create_pipeline_cache /
 *       vkr_create_pipeline_layout creation helpers return VK_SUCCESS.
 *
 * This is a header-only test: it includes only <vulkan/vulkan.h> and the
 * public vkruntime.h header (relative include). No internal headers are pulled.
 *
 * Exit status: 0 when all checks pass. Returns 1 on any failure.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/vkruntime/vkruntime.h"

/* ===========================================================================
 * Test parameters
 * ========================================================================== */

#define TEST_ALLOC_COUNT  1000u    /**< Number of sub-allocations to churn.   */
#define TEST_BIG_SIZE     (1u << 20)             /**< 1 MiB head buffer.       */
#define TEST_RT_BYTES     (256u * 1024u)         /**< Round-trip buffer size.  */
#define TEST_RT_FLOATS    (TEST_RT_BYTES / 4u)   /**< 65536 floats.            */
#define TEST_OFFSET       (TEST_RT_BYTES / 2u)   /**< Non-zero offset region.  */
#define TEST_OFFSET_SIZE  (4u * 1024u)           /**< 4 KiB at the offset.     */

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

static int report(int pass, const char *name)
{
    printf("  %-28s : %s\n", name, pass ? "PASS" : "FAIL");
    return pass;
}

/* ===========================================================================
 * main
 * ========================================================================== */

int main(void)
{
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool harness_pool = VK_NULL_HANDLE;
    VkCommandBuffer harness_cmd = VK_NULL_HANDLE;
    VkRuntime *rt = NULL;

    VkBuffer *bufs = NULL;
    VkDeviceMemory *mems = NULL;

    int overall_pass = 1;
    VkResult r;

    /* ── 1. Instance ────────────────────────────────────────────────────── */
    r = create_instance("test_vkruntime", &instance);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkruntime: vkCreateInstance failed (%d)\n", (int)r);
        return 1;
    }

    /* ── 2. Physical device ─────────────────────────────────────────────── */
    r = find_physical_device(instance, &physical_device);
    if (r != VK_SUCCESS) {
        printf("test_vkruntime: SKIP (no physical device found)\n");
        vkDestroyInstance(instance, NULL);
        return 0;
    }

    /* ── 3. Logical device on queue family 0 (shaderInt64 enabled) ──────── */
    r = create_device(physical_device, &device);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkruntime: vkCreateDevice failed (%d)\n", (int)r);
        goto cleanup;
    }
    vkGetDeviceQueue(device, 0, 0, &queue);

    /* ── 4. Runtime ─────────────────────────────────────────────────────── */
    r = vkr_create_runtime(physical_device, device, queue, &rt);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkruntime: vkr_create_runtime failed (%d)\n", (int)r);
        goto cleanup;
    }
    printf("test_vkruntime: device ready (arch=%u, %s, subgroup=%u)\n",
           (unsigned)vkr_get_arch_index(rt), vkr_get_arch_name(rt),
           (unsigned)vkr_get_subgroup_size(rt));

    /* ── 5. harness command pool + buffer (used by upload/download) ─────── */
    r = vkr_create_command_pool(rt, 0, &harness_pool);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkruntime: command pool failed (%d)\n", (int)r);
        goto cleanup;
    }
    {
        VkCommandBufferAllocateInfo alloc_info;
        memset(&alloc_info, 0, sizeof(alloc_info));
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = harness_pool;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = 1;
        r = vkAllocateCommandBuffers(device, &alloc_info, &harness_cmd);
        if (r != VK_SUCCESS) {
            fprintf(stderr, "test_vkruntime: command buffer failed (%d)\n", (int)r);
            goto cleanup;
        }
    }

    /* ── 6. (a) capability query sanity ─────────────────────────────────── */
    printf("  --- capability queries ---\n");
    {
        uint32_t arch = vkr_get_arch_index(rt);
        overall_pass &= report(arch <= 2, "arch index in {0,1,2}");
        overall_pass &= report(vkr_get_arch_name(rt) != NULL, "arch name non-null");
        overall_pass &= report(vkr_get_subgroup_size(rt) > 0, "subgroup size > 0");
        overall_pass &= report(vkr_has_subgroup(rt) == VK_TRUE, "subgroup compute");
        overall_pass &= report(vkr_get_device(rt) == device, "device handle match");
    }

    /* ── 7. (b) pooled allocator ────────────────────────────────────────── */
    printf("  --- pooled allocator ---\n");
    {
        VkBuffer big = VK_NULL_HANDLE;
        VkDeviceMemory big_mem = VK_NULL_HANDLE;
        VkBufferUsageFlags dev_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                     | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                                     | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        r = vkr_malloc(rt, TEST_BIG_SIZE, dev_usage, &big, &big_mem);
        if (r != VK_SUCCESS) {
            fprintf(stderr, "test_vkruntime: 1 MiB vkr_malloc failed (%d)\n", (int)r);
            overall_pass = 0;
        } else {
            overall_pass &= report(big != VK_NULL_HANDLE && big_mem != VK_NULL_HANDLE,
                                   "1 MiB alloc");
        }

        bufs = (VkBuffer *)calloc(TEST_ALLOC_COUNT, sizeof(VkBuffer));
        mems = (VkDeviceMemory *)calloc(TEST_ALLOC_COUNT, sizeof(VkDeviceMemory));
        if (!bufs || !mems) {
            fprintf(stderr, "test_vkruntime: host allocation failed\n");
            overall_pass = 0;
            goto cleanup;
        }

        int all_ok = 1;
        for (uint32_t i = 0; i < TEST_ALLOC_COUNT; i++) {
            VkDeviceSize size = 1024 + (VkDeviceSize)(i % 64) * 1024; /* 1KB..64KB */
            VkBuffer b = VK_NULL_HANDLE;
            VkDeviceMemory m = VK_NULL_HANDLE;
            VkResult ar = vkr_malloc(rt, size, dev_usage, &b, &m);
            if (ar != VK_SUCCESS) {
                printf("  alloc[%u] failed (VkResult=%d)\n", (unsigned)i, (int)ar);
                all_ok = 0;
                break;
            }
            bufs[i] = b;
            mems[i] = m;
        }
        overall_pass &= report(all_ok, "1000 sub-allocations succeed");

        /* distinctness: no two buffer handles equal */
        int distinct = 1;
        for (uint32_t i = 0; i < TEST_ALLOC_COUNT && all_ok; i++) {
            for (uint32_t j = i + 1; j < TEST_ALLOC_COUNT; j++) {
                if (bufs[i] == bufs[j]) { distinct = 0; break; }
            }
            if (!distinct) break;
        }
        overall_pass &= report(distinct, "all buffers distinct");

        /* vkr_free of every allocation */
        for (uint32_t i = 0; i < TEST_ALLOC_COUNT; i++) {
            if (bufs[i]) vkr_free(rt, bufs[i], mems[i]);
        }
        if (big) vkr_free(rt, big, big_mem);
        overall_pass &= report(1, "vkr_free all allocations");
    }

    /* ── 8. (c) upload/download round-trip ──────────────────────────────── */
    printf("  --- upload / download round-trip ---\n");
    {
        VkBuffer dev = VK_NULL_HANDLE;
        VkDeviceMemory dev_mem = VK_NULL_HANDLE;
        VkBufferUsageFlags dev_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                     | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                                     | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        float *ref = (float *)malloc(TEST_RT_BYTES);
        float *out = (float *)malloc(TEST_RT_BYTES);
        unsigned char *ref_off = (unsigned char *)malloc(TEST_OFFSET_SIZE);
        unsigned char *out_off = (unsigned char *)malloc(TEST_OFFSET_SIZE);
        if (!ref || !out || !ref_off || !out_off) {
            fprintf(stderr, "test_vkruntime: host allocation failed\n");
            overall_pass = 0;
            goto cleanup;
        }

        r = vkr_malloc(rt, TEST_RT_BYTES, dev_usage, &dev, &dev_mem);
        if (r != VK_SUCCESS) {
            fprintf(stderr, "test_vkruntime: round-trip vkr_malloc failed (%d)\n", (int)r);
            overall_pass = 0;
            free(ref); free(out); free(ref_off); free(out_off);
            goto cleanup;
        }

        for (uint32_t i = 0; i < TEST_RT_FLOATS; i++) {
            ref[i] = (float)((int)(i % 1000)) / 4.0f;   /* deterministic pattern */
        }
        for (uint32_t i = 0; i < TEST_OFFSET_SIZE; i++) {
            ref_off[i] = (unsigned char)((i * 131) % 251);
        }
        memset(out, 0, TEST_RT_BYTES);
        memset(out_off, 0, TEST_OFFSET_SIZE);

        /* full-buffer upload at offset 0 */
        int up_ok = vkr_upload(rt, harness_cmd, queue, ref, dev, 0, TEST_RT_BYTES)
                    == VK_SUCCESS;
        overall_pass &= report(up_ok, "upload 256 KiB");

        int dn_ok = vkr_download(rt, harness_cmd, queue, dev, 0, out, TEST_RT_BYTES)
                    == VK_SUCCESS;
        overall_pass &= report(dn_ok, "download 256 KiB");

        overall_pass &= report(memcmp(out, ref, TEST_RT_BYTES) == 0,
                               "round-trip byte-identical");

        /* non-zero offset upload/download */
        int off_up = vkr_upload(rt, harness_cmd, queue, ref_off, dev,
                                TEST_OFFSET, TEST_OFFSET_SIZE) == VK_SUCCESS;
        overall_pass &= report(off_up, "upload at offset");

        int off_dn = vkr_download(rt, harness_cmd, queue, dev,
                                  TEST_OFFSET, out_off, TEST_OFFSET_SIZE)
                     == VK_SUCCESS;
        overall_pass &= report(off_dn, "download at offset");

        overall_pass &= report(memcmp(out_off, ref_off, TEST_OFFSET_SIZE) == 0,
                               "offset round-trip byte-identical");

        vkr_free(rt, dev, dev_mem);
        free(ref); free(out); free(ref_off); free(out_off);
    }

    /* ── 9. (d) pool helpers ────────────────────────────────────────────── */
    printf("  --- pool helpers ---\n");
    {
        VkCommandPool cmd_pool = VK_NULL_HANDLE;
        VkDescriptorPool desc_pool = VK_NULL_HANDLE;

        r = vkr_create_command_pool(rt, 0, &cmd_pool);
        overall_pass &= report(r == VK_SUCCESS, "command pool created");
        if (cmd_pool) vkDestroyCommandPool(device, cmd_pool, NULL);

        r = vkr_create_descriptor_pool(device, 64, 256, &desc_pool);
        overall_pass &= report(r == VK_SUCCESS, "descriptor pool created");
        if (desc_pool) vkDestroyDescriptorPool(device, desc_pool, NULL);

        vkr_wait_idle(rt);
        overall_pass &= report(1, "vkr_wait_idle");
    }

    /* ── 10. (e) vkr_detect_capabilities consistency ───────────────────── */
    printf("  --- vkr_detect_capabilities ---\n");
    {
        VkRuntimeCaps caps;
        memset(&caps, 0, sizeof(caps));
        r = vkr_detect_capabilities(physical_device, device, &caps);
        overall_pass &= report(r == VK_SUCCESS, "detect succeeds");

        overall_pass &= report(caps.arch_index == vkr_get_arch_index(rt),
                               "arch index matches runtime");
        int name_ok = caps.arch_name != NULL &&
                      strcmp(caps.arch_name, vkr_get_arch_name(rt)) == 0;
        overall_pass &= report(name_ok, "arch name matches runtime");
        overall_pass &= report(caps.subgroup_size > 0, "subgroup size > 0");
        overall_pass &= report(caps.subgroup_size == vkr_get_subgroup_size(rt),
                               "subgroup size matches runtime");
        overall_pass &= report(caps.has_subgroup == vkr_has_subgroup(rt),
                               "has_subgroup matches runtime");
        overall_pass &= report(caps.has_coop_matrix == vkr_has_coop_matrix(rt),
                               "has_coop_matrix matches runtime");
    }

    /* ── 11. (f) creation helpers ──────────────────────────────────────── */
    printf("  --- creation helpers ---\n");
    {
        VkPipelineCache cache = VK_NULL_HANDLE;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;

        r = vkr_create_pipeline_cache(device, &cache);
        overall_pass &= report(r == VK_SUCCESS, "pipeline cache created");

        r = vkr_create_descriptor_pool(device, 64, 256, &pool);
        overall_pass &= report(r == VK_SUCCESS, "descriptor pool created");

        /* trivial set layout (one SSBO binding) + one push-constant range */
        VkDescriptorSetLayoutBinding binding;
        memset(&binding, 0, sizeof(binding));
        binding.binding = 0;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo dslci;
        memset(&dslci, 0, sizeof(dslci));
        dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslci.bindingCount = 1;
        dslci.pBindings = &binding;
        r = vkCreateDescriptorSetLayout(device, &dslci, NULL, &set_layout);
        overall_pass &= report(r == VK_SUCCESS, "set layout created");

        if (r == VK_SUCCESS) {
            VkPushConstantRange pcr;
            memset(&pcr, 0, sizeof(pcr));
            pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            pcr.offset = 0;
            pcr.size = 16;
            r = vkr_create_pipeline_layout(device, set_layout, 1, &pcr,
                                           &layout);
            overall_pass &= report(r == VK_SUCCESS, "pipeline layout created");
        }

        if (cache) vkDestroyPipelineCache(device, cache, NULL);
        if (pool) vkDestroyDescriptorPool(device, pool, NULL);
        if (layout) vkDestroyPipelineLayout(device, layout, NULL);
        if (set_layout) vkDestroyDescriptorSetLayout(device, set_layout, NULL);
    }

    /* ── 12. (g) vkr_create_device (canonical full-feature device) ──────── */
    printf("  --- vkr_create_device ---\n");
    {
        VkDevice dev2 = VK_NULL_HANDLE;
        VkQueue q2 = VK_NULL_HANDLE;

        r = vkr_create_device(physical_device, 0, &dev2);
        overall_pass &= report(r == VK_SUCCESS, "create succeeds");
        if (r != VK_SUCCESS) goto cleanup;

        vkGetDeviceQueue(dev2, 0, 0, &q2);
        overall_pass &= report(q2 != VK_NULL_HANDLE, "queue available");

        /* capability detection must agree on the full-feature device */
        VkRuntimeCaps caps2;
        memset(&caps2, 0, sizeof(caps2));
        r = vkr_detect_capabilities(physical_device, dev2, &caps2);
        overall_pass &= report(r == VK_SUCCESS, "detect on full-feature device");

        overall_pass &= report(caps2.has_push_descriptor,
                               "push descriptors enabled");
        overall_pass &= report(caps2.has_shader_float16,
                               "shaderFloat16 enabled");
        overall_pass &= report(caps2.has_storage_buffer16,
                               "storageBuffer16 enabled");
        overall_pass &= report(caps2.has_subgroup,
                               "compute-stage subgroup enabled");
        if (caps2.has_subgroup)
            overall_pass &= report(caps2.subgroup_size == caps2.wavefront_size ||
                                   caps2.wavefront_size == 0,
                                   "subgroup/wavefront consistent");

        /* invalid queue family must fail cleanly */
        VkDevice bad = VK_NULL_HANDLE;
        r = vkr_create_device(physical_device, 0xFFFFFFu, &bad);
        overall_pass &= report(r != VK_SUCCESS && bad == VK_NULL_HANDLE,
                               "invalid queue family rejected");
        if (bad != VK_NULL_HANDLE) vkDestroyDevice(bad, NULL);

        vkDestroyDevice(dev2, NULL);
    }

cleanup:
    if (bufs) {
        for (uint32_t i = 0; i < TEST_ALLOC_COUNT; i++) {
            if (bufs[i]) vkr_free(rt, bufs[i], mems[i]);
        }
    }
    if (harness_cmd != VK_NULL_HANDLE && harness_pool != VK_NULL_HANDLE)
        vkFreeCommandBuffers(device, harness_pool, 1, &harness_cmd);
    if (harness_pool != VK_NULL_HANDLE)
        vkDestroyCommandPool(device, harness_pool, NULL);
    if (rt) vkr_destroy_runtime(rt);
    if (device != VK_NULL_HANDLE) vkDestroyDevice(device, NULL);
    if (instance != VK_NULL_HANDLE) vkDestroyInstance(instance, NULL);
    free(bufs);
    free(mems);

    printf("test_vkruntime: %s\n", overall_pass ? "PASS" : "FAIL");
    return overall_pass ? 0 : 1;
}
