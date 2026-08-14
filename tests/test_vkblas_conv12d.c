/**
 * \file test_vkblas_conv12d.c
 * \brief Native Vulkan 1D/2D convolution tests (register-blocked direct rb2).
 *
 * conv1d (identity, 1x1 kernel) and conv2d (3x3, pad=1, all-ones input) with
 * host references. Uses host-visible buffers and a single command buffer.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vulkan/vulkan.h>
#include <vkruntime/vkruntime.h>
#include <vkblas/vkblas.h>

static VkInstance      g_instance;
static VkPhysicalDevice g_phys_dev;
static VkDevice        g_device;
static VkQueue         g_queue;
static VkCommandPool   g_pool;
static VkCommandBuffer g_cmd;
static VkFence         g_fence;
static VkBLASContext*  g_ctx;

static void die(const char *msg, VkResult r) {
    fprintf(stderr, "FAIL: %s (VkResult %d)\n", msg, r);
    exit(1);
}

static VkBuffer make_buffer(VkDeviceSize size, VkDeviceMemory *out_mem,
                            void **out_map)
{
    VkBufferCreateInfo bi;
    memset(&bi, 0, sizeof(bi));
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
             | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
             | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buf = VK_NULL_HANDLE;
    if (vkCreateBuffer(g_device, &bi, NULL, &buf) != VK_SUCCESS)
        die("vkCreateBuffer", VK_ERROR_INITIALIZATION_FAILED);

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(g_device, buf, &req);
    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(g_phys_dev, &props);
    uint32_t mi = UINT32_MAX;
    for (uint32_t i = 0; i < props.memoryTypeCount; i++) {
        if ((req.memoryTypeBits & (1u << i)) &&
            (props.memoryTypes[i].propertyFlags &
             (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
             (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            mi = i; break;
        }
    }
    if (mi == UINT32_MAX) die("no host-visible memory", VK_ERROR_FEATURE_NOT_PRESENT);

    VkMemoryAllocateInfo mai;
    memset(&mai, 0, sizeof(mai));
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = mi;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    if (vkAllocateMemory(g_device, &mai, NULL, &mem) != VK_SUCCESS)
        die("vkAllocateMemory", VK_ERROR_OUT_OF_DEVICE_MEMORY);
    vkBindBufferMemory(g_device, buf, mem, 0);

    void *map = NULL;
    vkMapMemory(g_device, mem, 0, VK_WHOLE_SIZE, 0, &map);
    *out_mem = mem;
    *out_map = map;
    return buf;
}

/* Begin a one-time cmd buffer, add host->shader barrier, return cmd. */
static VkCommandBuffer begin_cmd(void)
{
    VkCommandBufferBeginInfo bi;
    memset(&bi, 0, sizeof(bi));
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(g_cmd, &bi) != VK_SUCCESS)
        die("vkBeginCommandBuffer", VK_ERROR_INITIALIZATION_FAILED);
    VkMemoryBarrier mb;
    memset(&mb, 0, sizeof(mb));
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(g_cmd,
        VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &mb, 0, NULL, 0, NULL);
    return g_cmd;
}

static void end_submit(void)
{
    VkMemoryBarrier mb;
    memset(&mb, 0, sizeof(mb));
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(g_cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
        0, 1, &mb, 0, NULL, 0, NULL);
    if (vkEndCommandBuffer(g_cmd) != VK_SUCCESS)
        die("vkEndCommandBuffer", VK_ERROR_INITIALIZATION_FAILED);
    VkSubmitInfo si;
    memset(&si, 0, sizeof(si));
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &g_cmd;
    if (vkQueueSubmit(g_queue, 1, &si, g_fence) != VK_SUCCESS)
        die("vkQueueSubmit", VK_ERROR_INITIALIZATION_FAILED);
    vkWaitForFences(g_device, 1, &g_fence, VK_TRUE, UINT64_MAX);
    vkResetFences(g_device, 1, &g_fence);
}

/* === Test 1: conv1d, 1x1 kernel (identity) === */
static int test_conv1d(void)
{
    uint32_t n = 1, c = 2, li = 8, k = 2, lo = 8, kl = 1;
    float x_host[16] = {
        1,2,3,4,5,6,7,8,
        0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f
    };
    float w_host[4] = { 1,0,0,1 }; /* identity */

    VkDeviceMemory xm, wm, ym;
    void *xmap, *wmap, *ymap;
    VkBuffer x = make_buffer(sizeof(x_host), &xm, &xmap);
    VkBuffer w = make_buffer(sizeof(w_host), &wm, &wmap);
    VkBuffer y = make_buffer(sizeof(x_host), &ym, &ymap);
    memcpy(xmap, x_host, sizeof(x_host));
    memcpy(wmap, w_host, sizeof(w_host));
    memset(ymap, 0, sizeof(x_host));

    VkCommandBuffer cmd = begin_cmd();
    if (vkblas_conv1d_f32(g_ctx, n, c, li, k, lo, kl,
            0, 1, 1, 1.0f, x, w, 0.0f, y, cmd) != VK_SUCCESS)
        die("vkblas_conv1d_f32", VK_ERROR_INITIALIZATION_FAILED);
    end_submit();

    int pass = 1;
    for (int ch = 0; ch < 2 && pass; ch++)
        for (int l = 0; l < 8; l++) {
            float exp = x_host[ch * 8 + l];
            float got = ((float*)ymap)[ch * 8 + l];
            if (fabsf(got - exp) > 0.01f) {
                printf("  MISMATCH y[%d][%d]: exp=%.4f got=%.4f\n", ch, l, exp, got);
                pass = 0; break;
            }
        }
    printf("  conv1d_f32 : %s\n", pass ? "PASS" : "FAIL");

    vkDestroyBuffer(g_device, x, NULL); vkFreeMemory(g_device, xm, NULL);
    vkDestroyBuffer(g_device, w, NULL); vkFreeMemory(g_device, wm, NULL);
    vkDestroyBuffer(g_device, y, NULL); vkFreeMemory(g_device, ym, NULL);
    return pass;
}

/* === Test 2: conv2d, 3x3 kernel, stride=1, pad=1 (spatial preserved) === */
static int test_conv2d(void)
{
    uint32_t n = 1, c = 1, hi = 5, wi = 5, k = 1, dh = 5, dw = 5, kh = 3, kw = 3;
    float x_host[25];
    for (int i = 0; i < 25; i++) x_host[i] = 1.0f;
    float w_host[9] = { 0,0,0, 0,1,0, 0,0,0 }; /* center-only kernel */

    VkDeviceMemory xm, wm, ym;
    void *xmap, *wmap, *ymap;
    VkBuffer x = make_buffer(sizeof(x_host), &xm, &xmap);
    VkBuffer w = make_buffer(sizeof(w_host), &wm, &wmap);
    VkBuffer y = make_buffer(sizeof(x_host), &ym, &ymap);
    memcpy(xmap, x_host, sizeof(x_host));
    memcpy(wmap, w_host, sizeof(w_host));
    memset(ymap, 0, sizeof(x_host));

    VkCommandBuffer cmd = begin_cmd();
    if (vkblas_conv2d_f32(g_ctx, n, c, hi, wi, k, dh, dw, kh, kw,
            1, 1, 1, 1, 1, 1, 1.0f, x, w, 0.0f, y, cmd) != VK_SUCCESS)
        die("vkblas_conv2d_f32", VK_ERROR_INITIALIZATION_FAILED);
    end_submit();

    /* With a center-only kernel and pad=1, every output pixel sees an input
     * pixel under the center weight -> all outputs = 1.0. */
    int pass = 1;
    for (int i = 0; i < 25; i++) {
        if (fabsf(((float*)ymap)[i] - 1.0f) > 0.01f) {
            printf("  MISMATCH y[%d]: exp=1.0 got=%.4f\n", i, ((float*)ymap)[i]);
            pass = 0; break;
        }
    }
    printf("  conv2d_f32 : %s\n", pass ? "PASS" : "FAIL");

    vkDestroyBuffer(g_device, x, NULL); vkFreeMemory(g_device, xm, NULL);
    vkDestroyBuffer(g_device, w, NULL); vkFreeMemory(g_device, wm, NULL);
    vkDestroyBuffer(g_device, y, NULL); vkFreeMemory(g_device, ym, NULL);
    return pass;
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("Starting native Vulkan conv1d/conv2d test...\n");

    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "test_vkblas_conv12d",
        .apiVersion = VK_API_VERSION_1_3,
    };
    VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                 .pApplicationInfo = &app };
    if (vkCreateInstance(&ici, NULL, &g_instance) != VK_SUCCESS)
        die("vkCreateInstance", VK_ERROR_INITIALIZATION_FAILED);

    uint32_t npd = 0;
    vkEnumeratePhysicalDevices(g_instance, &npd, NULL);
    if (npd == 0) {
        printf("SKIP: no GPU\n");
        vkDestroyInstance(g_instance, NULL);
        return 0;
    }
    vkEnumeratePhysicalDevices(g_instance, &npd, &g_phys_dev);

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = 0, .queueCount = 1, .pQueuePriorities = &prio,
    };
    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
    };
    if (vkr_create_device(g_phys_dev, 0, &g_device) != VK_SUCCESS)
        die("vkr_create_device", VK_ERROR_INITIALIZATION_FAILED);
    vkGetDeviceQueue(g_device, 0, 0, &g_queue);

    VkCommandPoolCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = 0,
    };
    if (vkCreateCommandPool(g_device, &cpci, NULL, &g_pool) != VK_SUCCESS)
        die("vkCreateCommandPool", VK_ERROR_INITIALIZATION_FAILED);
    VkCommandBufferAllocateInfo cbai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = g_pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    if (vkAllocateCommandBuffers(g_device, &cbai, &g_cmd) != VK_SUCCESS)
        die("vkAllocateCommandBuffers", VK_ERROR_INITIALIZATION_FAILED);
    VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    vkCreateFence(g_device, &fci, NULL, &g_fence);

    if (vkblas_create_context(g_instance, g_phys_dev, g_device, &g_ctx) != VK_SUCCESS)
        die("vkblas_create_context", VK_ERROR_INITIALIZATION_FAILED);

    int p1 = test_conv1d();
    int p2 = test_conv2d();

    vkblas_destroy_context(g_ctx);
    vkDestroyFence(g_device, g_fence, NULL);
    vkFreeCommandBuffers(g_device, g_pool, 1, &g_cmd);
    vkDestroyCommandPool(g_device, g_pool, NULL);
    vkDestroyDevice(g_device, NULL);
    vkDestroyInstance(g_instance, NULL);

    printf("\n=== RESULT: %s ===\n", (p1 && p2) ? "PASS" : "FAIL");
    return (p1 && p2) ? 0 : 1;
}
