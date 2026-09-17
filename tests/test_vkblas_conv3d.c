/**
 * \file test_vkblas_conv3d.c
 * \brief Native Vulkan 3D convolution test (register-blocked direct rb2).
 *
 * Creates host-visible device buffers for x/w/y, records a vkblas_conv3d_f32
 * dispatch into a command buffer, submits, and verifies against a host
 * reference. No HIP / MIOpen involvement.
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

static void run_dispatch(VkBuffer x, VkBuffer w, VkBuffer y)
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

    VkResult r = vkblas_conv3d_f32(g_ctx, 1, 1, 2, 2, 2, 1, 1, 1, 1,
        2, 2, 2, 0, 0, 0, 1, 1, 1, 1, 1, 1,
        1.0f, x, w, 0.0f, y, g_cmd);
    if (r != VK_SUCCESS) die("vkblas_conv3d_f32", r);

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

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("Starting native Vulkan conv3d test...\n");

    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "test_vkblas_conv3d",
        .apiVersion = VK_API_VERSION_1_3,
    };
    VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                 .pApplicationInfo = &app };
    if (vkCreateInstance(&ici, NULL, &g_instance) != VK_SUCCESS)
        die("vkCreateInstance", VK_ERROR_INITIALIZATION_FAILED);

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(g_instance, &n, NULL);
    if (n == 0) {
        printf("SKIP: no GPU\n");
        vkDestroyInstance(g_instance, NULL);
        return 0;
    }
    vkEnumeratePhysicalDevices(g_instance, &n, &g_phys_dev);

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

    /* Simple 3D conv: 1x1x2x2x2 input, 2x2x2 kernel, output 1x1x1x1x1 */
    float h_in[8]  = {1, 2, 3, 4, 5, 6, 7, 8};
    float h_w[8]   = {1, 0, 0, 1, 1, 0, 1, 0};
    float h_expected = 1*1 + 2*0 + 3*0 + 4*1 + 5*1 + 6*0 + 7*1 + 8*0; /* 17 */

    VkDeviceMemory x_mem, w_mem, y_mem;
    void *x_map, *w_map, *y_map;
    VkBuffer x = make_buffer(sizeof(h_in), &x_mem, &x_map);
    VkBuffer w = make_buffer(sizeof(h_w),  &w_mem, &w_map);
    VkBuffer y = make_buffer(sizeof(float), &y_mem, &y_map);

    memcpy(x_map, h_in, sizeof(h_in));
    memcpy(w_map, h_w, sizeof(h_w));
    memset(y_map, 0, sizeof(float));

    run_dispatch(x, w, y);

    float out = ((float*)y_map)[0];
    float diff = fabsf(out - h_expected);
    int ok = diff < 1e-4f;
    printf("conv3d: out=%f expected=%f diff=%.2e => %s\n",
           out, h_expected, diff, ok ? "PASS" : "FAIL");

    vkblas_destroy_context(g_ctx);
    vkDestroyFence(g_device, g_fence, NULL);
    vkFreeCommandBuffers(g_device, g_pool, 1, &g_cmd);
    vkDestroyCommandPool(g_device, g_pool, NULL);
    vkDestroyBuffer(g_device, x, NULL); vkFreeMemory(g_device, x_mem, NULL);
    vkDestroyBuffer(g_device, w, NULL); vkFreeMemory(g_device, w_mem, NULL);
    vkDestroyBuffer(g_device, y, NULL); vkFreeMemory(g_device, y_mem, NULL);
    vkDestroyDevice(g_device, NULL);
    vkDestroyInstance(g_instance, NULL);

    return ok ? 0 : 1;
}
