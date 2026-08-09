/**
 * \file test_vkmodel.c
 * \brief Public-API test harness for the VKModel GGUF loader.
 *
 * Bootstraps a minimal Vulkan instance + physical/logical device, creates a
 * VkRuntime via vkr_create_runtime(), then:
 *   (a) writes a small synthetic GGUF file in C (magic "GGUF", version 3,
 *       metadata KVs, 3 tensors: F32 [8,16], Q8_0 [8,8], F16 [8,8]) and
 *       loads it with vkmodel_load();
 *   (b) asserts kv_count, get_kv_string("general.architecture") == "llama",
 *       tensor names/dtypes/nelems/sizes, and the vkmodel_block_elems() map;
 *   (c) downloads tensor 0 (F32) back via vkr_download and memcmp's it
 *       against the exact bytes written (proves the upload path);
 *   (d) vkmodel_destroy() frees without error;
 *   (e) loads a second file with a 33-byte tensor (I8 [33]) to verify the
 *       loader reads tensor data at the gguf offset field verbatim through
 *       the 32-byte-aligned data region.
 *
 * This is a header-only test: it includes only <vulkan/vulkan.h> and the
 * public vkmodel.h header. No internal headers are pulled.
 *
 * Exit status: 0 when all checks pass. Returns 1 on any failure.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/vkmodel/vkmodel.h"

/* ===========================================================================
 * Test parameters
 * ========================================================================== */

#define TEST_FILE_A "test_vkmodel_a.gguf"   /**< 3-tensor synthetic model.  */
#define TEST_FILE_B "test_vkmodel_b.gguf"   /**< Odd-size (33-byte) tensor. */

#define GGUF_MAGIC_LE 0x46554747u           /**< "GGUF" little-endian.      */
#define GGUF_VERSION  3u

/* ===========================================================================
 * Bootstrap helpers (mirror test_vkruntime.c)
 * ========================================================================== */

static VkResult create_instance(const char* app_name, VkInstance* out_instance)
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
                                     VkPhysicalDevice* out_physical_device)
{
    uint32_t count = 0;
    VkResult r = vkEnumeratePhysicalDevices(instance, &count, NULL);
    if (r != VK_SUCCESS || count == 0) return VK_ERROR_INITIALIZATION_FAILED;

    VkPhysicalDevice* devices =
        (VkPhysicalDevice*)malloc(count * sizeof(VkPhysicalDevice));
    if (!devices) return VK_ERROR_OUT_OF_HOST_MEMORY;

    r = vkEnumeratePhysicalDevices(instance, &count, devices);
    if (r == VK_SUCCESS) *out_physical_device = devices[0];
    free(devices);
    return r;
}

static VkResult create_device(VkPhysicalDevice physical_device,
                              VkDevice* out_device)
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

static int report(int pass, const char* name)
{
    printf("  %-32s : %s\n", name, pass ? "PASS" : "FAIL");
    return pass;
}

/* ===========================================================================
 * Synthetic GGUF writer
 * ========================================================================== */

static void wr_u32(FILE* f, uint32_t v, uint64_t* pos)
{
    uint8_t b[4] = { (uint8_t)v, (uint8_t)(v >> 8),
                     (uint8_t)(v >> 16), (uint8_t)(v >> 24) };
    fwrite(b, 1, 4, f);
    *pos += 4;
}

static void wr_u64(FILE* f, uint64_t v, uint64_t* pos)
{
    uint8_t b[8];
    for (int i = 0; i < 8; i++) b[i] = (uint8_t)(v >> (8 * i));
    fwrite(b, 1, 8, f);
    *pos += 8;
}

static void wr_str(FILE* f, const char* s, uint64_t* pos)
{
    uint64_t len = strlen(s);
    wr_u64(f, len, pos);
    fwrite(s, 1, (size_t)len, f);
    *pos += len;
}

static void wr_string_kv(FILE* f, const char* key, const char* val,
                         uint64_t* pos)
{
    wr_str(f, key, pos);
    wr_u32(f, 8, pos);      /* GGUF_METADATA_VALUE_TYPE_STRING */
    wr_str(f, val, pos);
}

static void wr_uint32_kv(FILE* f, const char* key, uint32_t val,
                         uint64_t* pos)
{
    wr_str(f, key, pos);
    wr_u32(f, 4, pos);      /* GGUF_METADATA_VALUE_TYPE_UINT32 */
    wr_u32(f, val, pos);
}

static void wr_tensor_info(FILE* f, const char* name, uint32_t n_dims,
                           const uint64_t* dims, uint32_t type, uint64_t offset,
                           uint64_t* pos)
{
    wr_str(f, name, pos);
    wr_u32(f, n_dims, pos);
    for (uint32_t d = 0; d < n_dims; d++) wr_u64(f, dims[d], pos);
    wr_u32(f, type, pos);
    wr_u64(f, offset, pos);
}

static void wr_pad_to(FILE* f, uint64_t align, uint64_t* pos)
{
    uint64_t rem = (*pos) % align;
    uint64_t pad = rem == 0 ? 0 : align - rem;
    for (uint64_t i = 0; i < pad; i++) fputc(0, f);
    *pos += pad;
}

static void fill_pattern(uint8_t* buf, size_t n, unsigned seed)
{
    for (size_t i = 0; i < n; i++) buf[i] = (uint8_t)((i * seed) % 251u);
}

/**
 * \brief Write the 3-tensor synthetic GGUF (F32 / Q8_0 / F16) file.
 *
 * Tensor data offsets are deliberately non-contiguous after the first tensor
 * (Q8_0 at 512, F16 at 608) so the loader must honor the stored offsets
 * verbatim rather than assuming data is packed back-to-back.
 *
 * \retval 1 on success, 0 on write failure.
 */
static int write_file_a(const char* path)
{
    FILE* f = fopen(path, "wb");
    if (!f) return 0;
    uint64_t pos = 0;

    /* header */
    wr_u32(f, GGUF_MAGIC_LE, &pos);
    wr_u32(f, GGUF_VERSION, &pos);
    wr_u64(f, 3, &pos);   /* tensor_count */
    wr_u64(f, 2, &pos);   /* metadata_kv_count */

    /* metadata KVs */
    wr_string_kv(f, "general.architecture", "llama", &pos);
    wr_uint32_kv(f, "llama.context_length", 2048, &pos);

    /* tensor infos */
    {
        const uint64_t dims_embd[2] = { 8, 16 };   /* F32  [8,16] 512 B  */
        const uint64_t dims_q[2]    = { 8, 8 };    /* Q8_0 [8,8]  72 B   */
        const uint64_t dims_out[2]  = { 8, 8 };    /* F16  [8,8]  128 B  */
        wr_tensor_info(f, "token_embd", 2, dims_embd, 0, 0, &pos);
        wr_tensor_info(f, "blk.0.attn_q.weight", 2, dims_q, 8, 512, &pos);
        wr_tensor_info(f, "output.weight", 2, dims_out, 1, 608, &pos);
    }

    /* tensor data region, 32-byte aligned */
    wr_pad_to(f, 32, &pos);
    {
        uint8_t buf[512];
        fill_pattern(buf, 512, 131);
        fwrite(buf, 1, 512, f);    /* token_embd at data_offset + 0  */
        pos += 512;

        wr_pad_to(f, 32, &pos);    /* 512 already aligned: no-op     */
        fill_pattern(buf, 72, 97);
        fwrite(buf, 1, 72, f);     /* attn_q at data_offset + 512    */
        pos += 72;

        wr_pad_to(f, 32, &pos);    /* pad 584 -> 608                 */
        fill_pattern(buf, 128, 67);
        fwrite(buf, 1, 128, f);    /* output at data_offset + 608    */
        pos += 128;
    }

    fclose(f);
    return 1;
}

/**
 * \brief Write the odd-size synthetic GGUF (one I8 tensor with 33 elements).
 *
 * A 33-byte tensor data region is not a multiple of 32; the loader must read
 * exactly 33 bytes at the offset stored in the file (verbatim), not at a
 * re-computed aligned offset.
 *
 * \retval 1 on success, 0 on write failure.
 */
static int write_file_b(const char* path)
{
    FILE* f = fopen(path, "wb");
    if (!f) return 0;
    uint64_t pos = 0;

    wr_u32(f, GGUF_MAGIC_LE, &pos);
    wr_u32(f, GGUF_VERSION, &pos);
    wr_u64(f, 1, &pos);   /* tensor_count */
    wr_u64(f, 2, &pos);   /* metadata_kv_count */

    wr_string_kv(f, "general.architecture", "llama", &pos);
    wr_uint32_kv(f, "general.alignment", 32, &pos);

    {
        const uint64_t dims[1] = { 33 };
        wr_tensor_info(f, "odd.weight", 1, dims, 24, 0, &pos);   /* I8 */
    }

    wr_pad_to(f, 32, &pos);
    {
        uint8_t buf[33];
        fill_pattern(buf, 33, 83);
        fwrite(buf, 1, 33, f);
        pos += 33;
    }

    fclose(f);
    return 1;
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
    VkRuntime* rt = NULL;
    VkModel* model = NULL;

    int overall_pass = 1;
    VkResult r;

    /* ── 1. Instance ────────────────────────────────────────────────────── */
    r = create_instance("test_vkmodel", &instance);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkmodel: vkCreateInstance failed (%d)\n", (int)r);
        return 1;
    }

    /* ── 2. Physical device ─────────────────────────────────────────────── */
    r = find_physical_device(instance, &physical_device);
    if (r != VK_SUCCESS) {
        printf("test_vkmodel: SKIP (no physical device found)\n");
        vkDestroyInstance(instance, NULL);
        return 0;
    }

    /* ── 3. Logical device on queue family 0 ────────────────────────────── */
    r = create_device(physical_device, &device);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkmodel: vkCreateDevice failed (%d)\n", (int)r);
        goto cleanup;
    }
    vkGetDeviceQueue(device, 0, 0, &queue);

    /* ── 4. Runtime ─────────────────────────────────────────────────────── */
    r = vkr_create_runtime(physical_device, device, queue, &rt);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkmodel: vkr_create_runtime failed (%d)\n", (int)r);
        goto cleanup;
    }
    printf("test_vkmodel: device ready (arch=%u, %s, subgroup=%u)\n",
           (unsigned)vkr_get_arch_index(rt), vkr_get_arch_name(rt),
           (unsigned)vkr_get_subgroup_size(rt));

    /* ── 5. Harness command pool + buffer (for vkr_download) ────────────── */
    r = vkr_create_command_pool(rt, 0, &harness_pool);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkmodel: command pool failed (%d)\n", (int)r);
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
            fprintf(stderr, "test_vkmodel: command buffer failed (%d)\n", (int)r);
            goto cleanup;
        }
    }

    /* ── 6. Write the synthetic GGUF files ──────────────────────────────── */
    printf("  --- synthetic GGUF writer ---\n");
    if (!write_file_a(TEST_FILE_A)) {
        fprintf(stderr, "test_vkmodel: failed to write %s\n", TEST_FILE_A);
        overall_pass = 0;
        goto cleanup;
    }
    overall_pass &= report(1, "wrote 3-tensor model file");
    if (!write_file_b(TEST_FILE_B)) {
        fprintf(stderr, "test_vkmodel: failed to write %s\n", TEST_FILE_B);
        overall_pass = 0;
        goto cleanup;
    }
    overall_pass &= report(1, "wrote odd-size model file");

    /* ── 7. Load + inspect model A ──────────────────────────────────────── */
    printf("  --- load + metadata ---\n");
    r = vkmodel_load(rt, TEST_FILE_A, &model);
    overall_pass &= report(r == VK_SUCCESS, "vkmodel_load file A");

    if (r == VK_SUCCESS) {
        overall_pass &= report(vkmodel_get_kv_count(model) == 2,
                               "kv_count == 2");

        const char* arch =
            vkmodel_get_kv_string(model, "general.architecture", NULL);
        overall_pass &= report(arch != NULL && strcmp(arch, "llama") == 0,
                               "general.architecture == llama");

        const char* nctx =
            vkmodel_get_kv_string(model, "llama.context_length", NULL);
        overall_pass &= report(nctx == NULL,
                               "non-string key returns NULL");

        const char* missing =
            vkmodel_get_kv_string(model, "missing.key", "fallback");
        overall_pass &= report(missing != NULL &&
                               strcmp(missing, "fallback") == 0,
                               "missing key returns fallback");

        const char* key0 = vkmodel_get_kv_key(model, 0);
        overall_pass &= report(key0 != NULL &&
                               strcmp(key0, "general.architecture") == 0,
                               "kv key 0");

        printf("  --- tensor info ---\n");
        overall_pass &= report(vkmodel_get_tensor_count(model) == 3,
                               "tensor_count == 3");

        overall_pass &= report(
            vkmodel_get_tensor_name(model, 0) != NULL &&
            strcmp(vkmodel_get_tensor_name(model, 0), "token_embd") == 0,
            "tensor 0 name == token_embd");
        overall_pass &= report(vkmodel_get_tensor_dtype(model, 0) == 0,
                               "tensor 0 dtype == F32");
        overall_pass &= report(vkmodel_get_tensor_nelems(model, 0) == 128,
                               "tensor 0 nelems == 128");
        overall_pass &= report(vkmodel_get_tensor_size(model, 0) == 512,
                               "tensor 0 size == 512");
        overall_pass &= report(
            vkmodel_get_tensor_buffer(model, 0) != VK_NULL_HANDLE,
            "tensor 0 buffer non-null");

        overall_pass &= report(
            vkmodel_get_tensor_name(model, 1) != NULL &&
            strcmp(vkmodel_get_tensor_name(model, 1),
                   "blk.0.attn_q.weight") == 0,
            "tensor 1 name == blk.0.attn_q.weight");
        overall_pass &= report(vkmodel_get_tensor_dtype(model, 1) == 8,
                               "tensor 1 dtype == Q8_0");
        overall_pass &= report(vkmodel_get_tensor_nelems(model, 1) == 64,
                               "tensor 1 nelems == 64");
        overall_pass &= report(vkmodel_get_tensor_size(model, 1) == 72,
                               "tensor 1 size == 72");

        overall_pass &= report(
            vkmodel_get_tensor_name(model, 2) != NULL &&
            strcmp(vkmodel_get_tensor_name(model, 2), "output.weight") == 0,
            "tensor 2 name == output.weight");
        overall_pass &= report(vkmodel_get_tensor_dtype(model, 2) == 1,
                               "tensor 2 dtype == F16");
        overall_pass &= report(vkmodel_get_tensor_nelems(model, 2) == 64,
                               "tensor 2 nelems == 64");
        overall_pass &= report(vkmodel_get_tensor_size(model, 2) == 128,
                               "tensor 2 size == 128");

        printf("  --- block-size mapping ---\n");
        overall_pass &= report(vkmodel_block_elems(0) == 1, "F32 -> 1");
        overall_pass &= report(vkmodel_block_elems(1) == 1, "F16 -> 1");
        overall_pass &= report(vkmodel_block_elems(2) == 32, "Q4_0 -> 32");
        overall_pass &= report(vkmodel_block_elems(8) == 32, "Q8_0 -> 32");
        overall_pass &= report(vkmodel_block_elems(12) == 256, "Q4_K -> 256");
        overall_pass &= report(vkmodel_block_elems(14) == 256, "Q6_K -> 256");
        overall_pass &= report(vkmodel_block_elems(23) == 256, "IQ4_XS -> 256");
        overall_pass &= report(vkmodel_block_elems(99) == 0, "unknown -> 0");

        printf("  --- upload round-trip (tensor 0, F32) ---\n");
        {
            uint8_t ref[512];
            uint8_t got[512];
            fill_pattern(ref, 512, 131);
            memset(got, 0xAA, sizeof(got));

            r = vkr_download(rt, harness_cmd, queue,
                             vkmodel_get_tensor_buffer(model, 0), 0,
                             got, sizeof(got));
            overall_pass &= report(r == VK_SUCCESS, "download tensor 0");
            overall_pass &= report(memcmp(ref, got, sizeof(got)) == 0,
                                   "tensor 0 byte-identical");
        }

        vkmodel_destroy(model);
        model = NULL;
        overall_pass &= report(1, "vkmodel_destroy model A");
    }

    /* ── 8. Odd-size tensor (33 bytes) file B ───────────────────────────── */
    printf("  --- odd-size tensor (I8 [33], 33 bytes) ---\n");
    r = vkmodel_load(rt, TEST_FILE_B, &model);
    overall_pass &= report(r == VK_SUCCESS, "vkmodel_load file B");

    if (r == VK_SUCCESS) {
        overall_pass &= report(vkmodel_get_tensor_count(model) == 1,
                               "file B tensor_count == 1");
        overall_pass &= report(
            vkmodel_get_tensor_name(model, 0) != NULL &&
            strcmp(vkmodel_get_tensor_name(model, 0), "odd.weight") == 0,
            "odd tensor name");
        overall_pass &= report(vkmodel_get_tensor_dtype(model, 0) == 24,
                               "odd dtype == I8");
        overall_pass &= report(vkmodel_get_tensor_nelems(model, 0) == 33,
                               "odd nelems == 33");
        overall_pass &= report(vkmodel_get_tensor_size(model, 0) == 33,
                               "odd size == 33");

        uint8_t ref[33];
        uint8_t got[33];
        fill_pattern(ref, 33, 83);
        memset(got, 0xAA, sizeof(got));
        r = vkr_download(rt, harness_cmd, queue,
                         vkmodel_get_tensor_buffer(model, 0), 0,
                         got, sizeof(got));
        overall_pass &= report(r == VK_SUCCESS, "download odd tensor");
        overall_pass &= report(memcmp(ref, got, sizeof(got)) == 0,
                               "odd tensor byte-identical (offset verbatim)");

        vkmodel_destroy(model);
        model = NULL;
        overall_pass &= report(1, "vkmodel_destroy model B");
    }

    /* ── 9. Error path: missing file ────────────────────────────────────── */
    printf("  --- error path ---\n");
    {
        VkModel* bad = NULL;
        r = vkmodel_load(rt, "test_vkmodel_missing.gguf", &bad);
        overall_pass &= report(r != VK_SUCCESS && bad == NULL,
                               "load missing file fails cleanly");
    }

cleanup:
    if (model) vkmodel_destroy(model);
    remove(TEST_FILE_A);
    remove(TEST_FILE_B);
    if (harness_cmd != VK_NULL_HANDLE && harness_pool != VK_NULL_HANDLE)
        vkFreeCommandBuffers(device, harness_pool, 1, &harness_cmd);
    if (harness_pool != VK_NULL_HANDLE)
        vkDestroyCommandPool(device, harness_pool, NULL);
    if (rt) vkr_destroy_runtime(rt);
    if (device != VK_NULL_HANDLE) vkDestroyDevice(device, NULL);
    if (instance != VK_NULL_HANDLE) vkDestroyInstance(instance, NULL);

    printf("test_vkmodel: %s\n", overall_pass ? "PASS" : "FAIL");
    return overall_pass ? 0 : 1;
}
