/**
 * \file vkcompress.c
 * \brief VKCompress implementation: context lifecycle, buffer registry, GPU compress/decompress.
 */
#include "vkcompress_internal.h"
#include "shaders_spv.h"
#include <string.h>
#include <stdlib.h>

static const uint32_t VKCOMP_WORKGROUP_MASK = VKCOMP_WORKGROUP_SIZE - 1u;

#define VKCOMP_CHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) return _r; } while(0)

static inline uint32_t vkcomp_div_ceil(uint32_t a, uint32_t b) {
    return (a + b - 1u) / b;
}

VkResult vkcompress_create_context(VkPhysicalDevice physicalDevice,
                                   VkDevice device,
                                   VkCompressContext** pContext)
{
    if (!pContext) return VK_ERROR_INITIALIZATION_FAILED;

    VkCompressContext* ctx = (VkCompressContext*)calloc(1, sizeof(VkCompressContext));
    if (!ctx) return VK_ERROR_OUT_OF_HOST_MEMORY;

    ctx->device = device;
    ctx->physical_device = physicalDevice;
    ctx->next_id = 1;
    ctx->staging_size = 256 * 1024 * 1024; /* 256 MB staging */

    /* Descriptor set layout: binding 0 = src (read), binding 1 = compressed (write/read), binding 2 = metadata */
    VkDescriptorSetLayoutBinding bindings[3] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
    };
    VkDescriptorSetLayoutCreateInfo dslci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3, .pBindings = bindings,
    };
    VKCOMP_CHECK(vkCreateDescriptorSetLayout(device, &dslci, NULL, &ctx->set_layout));

    /* Pipeline layout */
    VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &ctx->set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &(VkPushConstantRange){
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0, .size = sizeof(vkcomp_push_constants_t),
        },
    };
    VKCOMP_CHECK(vkCreatePipelineLayout(device, &plci, NULL, &ctx->pipeline_layout));

    /* Descriptor pool: 3 storage buffers per entry, max VKCOMP_MAX_BUFFERS entries */
    VkDescriptorPoolSize pool_sizes[1] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VKCOMP_MAX_BUFFERS * 3},
    };
    VkDescriptorPoolCreateInfo dpci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = VKCOMP_MAX_BUFFERS,
        .poolSizeCount = 1, .pPoolSizes = pool_sizes,
    };
    VKCOMP_CHECK(vkCreateDescriptorPool(device, &dpci, NULL, &ctx->descriptor_pool));

    /* Create pipelines */
    for (int i = 0; i < VKCOMP_MAX_BUFFERS; i++) {
        ctx->buffers[i].id = 0;
    }

    /* Lazy-init staging buffer on first write */

    /* Subgroup detection */
    ctx->has_subgroup = VK_FALSE;
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physicalDevice, &props);
    /* Assume subgroup support on Vulkan 1.1+ (safe default; tier fallback handles it) */
    if (props.apiVersion >= VK_API_VERSION_1_1) ctx->has_subgroup = VK_TRUE;

    *pContext = ctx;
    return VK_SUCCESS;
}

void vkcompress_destroy_context(VkCompressContext* ctx)
{
    if (!ctx) return;
    VkDevice dev = ctx->device;
    if (dev == VK_NULL_HANDLE) { free(ctx); return; }

    for (uint32_t i = 0; i < ctx->buffer_count; i++) {
        vkcomp_buffer_entry_t* e = &ctx->buffers[i];
        if (e->pipeline_write) vkDestroyPipeline(dev, e->pipeline_write, NULL);
        if (e->pipeline_read) vkDestroyPipeline(dev, e->pipeline_read, NULL);
        if (e->module_write) vkDestroyShaderModule(dev, e->module_write, NULL);
        if (e->module_read) vkDestroyShaderModule(dev, e->module_read, NULL);
        if (e->comp_memory) vkFreeMemory(dev, e->comp_memory, NULL);
        if (e->meta_memory) vkFreeMemory(dev, e->meta_memory, NULL);
        if (e->comp_buffer) vkDestroyBuffer(dev, e->comp_buffer, NULL);
        if (e->meta_buffer) vkDestroyBuffer(dev, e->meta_buffer, NULL);
    }
    if (ctx->staging_memory) vkFreeMemory(dev, ctx->staging_memory, NULL);
    if (ctx->staging_buffer) vkDestroyBuffer(dev, ctx->staging_buffer, NULL);
    if (ctx->pipeline_layout) vkDestroyPipelineLayout(dev, ctx->pipeline_layout, NULL);
    if (ctx->set_layout) vkDestroyDescriptorSetLayout(dev, ctx->set_layout, NULL);
    if (ctx->descriptor_pool) vkDestroyDescriptorPool(dev, ctx->descriptor_pool, NULL);
    free(ctx);
}

static VkResult ensure_staging(VkCompressContext* ctx)
{
    if (ctx->staging_buffer) return VK_SUCCESS;

    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = ctx->staging_size,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                 | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VKCOMP_CHECK(vkCreateBuffer(ctx->device, &bci, NULL, &ctx->staging_buffer));

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(ctx->device, ctx->staging_buffer, &mr);
    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mr.size,
        .memoryTypeIndex = 0, /* caller should have ensured host-visible type 0 */
    };
    VKCOMP_CHECK(vkAllocateMemory(ctx->device, &mai, NULL, &ctx->staging_memory));
    VKCOMP_CHECK(vkBindBufferMemory(ctx->device, ctx->staging_buffer, ctx->staging_memory, 0));
    return VK_SUCCESS;
}

static VkResult ensure_shader_module(VkDevice dev, const uint32_t* spirv, size_t size,
                                      VkShaderModule* pModule)
{
    if (*pModule) return VK_SUCCESS;
    VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = size,
        .pCode = spirv,
    };
    VKCOMP_CHECK(vkCreateShaderModule(dev, &smci, NULL, pModule));
    return VK_SUCCESS;
}

static VkResult ensure_pipeline(VkCompressContext* ctx, VkPipeline* pipeline,
                                VkShaderModule module)
{
    if (*pipeline) return VK_SUCCESS;
    VkComputePipelineCreateInfo pci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = module,
            .pName = "main",
        },
        .layout = ctx->pipeline_layout,
    };
    VKCOMP_CHECK(vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &pci, NULL, pipeline));
    return VK_SUCCESS;
}

vkcomp_buffer_id_t vkcompress_register_buffer(VkCompressContext* ctx,
                                              VkDeviceSize size,
                                              const char* tag,
                                              int compression_level)
{
    if (!ctx || !size || !tag) return VKCOMP_INVALID_ID;
    if (compression_level < 0) compression_level = 0;
    if (compression_level > 9) compression_level = 9;

    if (ctx->buffer_count >= VKCOMP_MAX_BUFFERS) return VKCOMP_INVALID_ID;

    uint64_t id = ctx->next_id++;
    vkcomp_buffer_entry_t* e = &ctx->buffers[ctx->buffer_count];
    memset(e, 0, sizeof(*e));
    e->id = id;
    e->uncompressed_size = size;
    e->compression_level = compression_level;
    e->valid = 1;
    strncpy(e->tag, tag, VKCOMP_MAX_TAG_LEN - 1u);
    ctx->buffer_count++;

    /* Allocate device buffers for compressed data + metadata */
    /* Worst case compression: 5% expansion for LZ4, use uncompressed size as upper bound */
    VkDeviceSize comp_cap = size; /* conservative */
    VkDeviceSize meta_cap = vkcomp_div_ceil((uint32_t)size, VKCOMP_WORKGROUP_SIZE * 32u) * 32u;

    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = comp_cap,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    if (vkCreateBuffer(ctx->device, &bci, NULL, &e->comp_buffer) != VK_SUCCESS) return VKCOMP_INVALID_ID;

    bci.size = meta_cap;
    if (vkCreateBuffer(ctx->device, &bci, NULL, &e->meta_buffer) != VK_SUCCESS) {
        vkDestroyBuffer(ctx->device, e->comp_buffer, NULL);
        return VKCOMP_INVALID_ID;
    }

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(ctx->device, e->comp_buffer, &mr);
    /* Find device-local memory type */
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(ctx->physical_device, &memProps);
    uint32_t memTypeIndex = 0;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((mr.memoryTypeBits & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            memTypeIndex = i; break;
        }
    }

    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mr.size,
        .memoryTypeIndex = memTypeIndex,
    };
    if (vkAllocateMemory(ctx->device, &mai, NULL, &e->comp_memory) != VK_SUCCESS) {
        vkDestroyBuffer(ctx->device, e->comp_buffer, NULL);
        vkDestroyBuffer(ctx->device, e->meta_buffer, NULL);
        return VKCOMP_INVALID_ID;
    }
    vkBindBufferMemory(ctx->device, e->comp_buffer, e->comp_memory, 0);

    vkGetBufferMemoryRequirements(ctx->device, e->meta_buffer, &mr);
    /* Reuse device-local type — meta buffer has same flags */
    mai.allocationSize = mr.size;
    if (vkAllocateMemory(ctx->device, &mai, NULL, &e->meta_memory) != VK_SUCCESS) {
        vkDestroyBuffer(ctx->device, e->comp_buffer, NULL);
        vkDestroyBuffer(ctx->device, e->meta_buffer, NULL);
        vkFreeMemory(ctx->device, e->comp_memory, NULL);
        return VKCOMP_INVALID_ID;
    }
    vkBindBufferMemory(ctx->device, e->meta_buffer, e->meta_memory, 0);

    e->compressed_size = comp_cap;
    return id;
}

VkResult vkcompress_write(VkCompressContext* ctx, VkCommandBuffer cmd,
                          vkcomp_buffer_id_t id,
                          VkBuffer src, VkDeviceSize size, VkDeviceSize offset)
{
    if (!ctx || id == VKCOMP_INVALID_ID || !src || !size) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    vkcomp_buffer_entry_t* e = NULL;
    for (uint32_t i = 0; i < ctx->buffer_count; i++) {
        if (ctx->buffers[i].id == id && ctx->buffers[i].valid) {
            e = &ctx->buffers[i];
            break;
        }
    }
    if (!e) return VK_ERROR_INITIALIZATION_FAILED;

    const uint32_t* comp_spv = (e->compression_level > 3)
        ? vkcompress_spv_baseline_compress_high
        : vkcompress_spv_baseline_compress_fast;
    size_t spirv_size = (e->compression_level > 3)
        ? vkcompress_spv_baseline_compress_high_size
        : vkcompress_spv_baseline_compress_fast_size;
    VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .codeSize = spirv_size,
        .pCode = comp_spv,
    };
    VkResult r = vkCreateShaderModule(ctx->device, &smci, NULL, &e->module_write);
    if (r) return r;
    r = ensure_pipeline(ctx, &e->pipeline_write, e->module_write);
    if (r) return r;

    /* Allocate descriptor set if needed */
    if (!e->desc_set) {
        VkDescriptorSetAllocateInfo dsai = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = ctx->descriptor_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &ctx->set_layout,
        };
        VKCOMP_CHECK(vkAllocateDescriptorSets(ctx->device, &dsai, &e->desc_set));
    }

    /* Write descriptors: binding 0 = src, binding 1 = comp_buffer, binding 2 = meta_buffer */
    VkDescriptorBufferInfo buf_infos[3] = {
        { .buffer = src, .offset = offset, .range = size },
        { .buffer = e->comp_buffer, .offset = 0, .range = size },
        { .buffer = e->meta_buffer, .offset = 0, .range = VKCOMP_MAX_BUFFERS * 2 * sizeof(uint32_t) },
    };
    VkWriteDescriptorSet writes[3] = {
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = e->desc_set,
          .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &buf_infos[0] },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = e->desc_set,
          .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &buf_infos[1] },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = e->desc_set,
          .dstBinding = 2, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &buf_infos[2] },
    };
    vkUpdateDescriptorSets(ctx->device, 3, writes, 0, NULL);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, e->pipeline_write);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layout, 0, 1, &e->desc_set, 0, NULL);

    uint32_t num_blocks = vkcomp_div_ceil((uint32_t)size, VKCOMP_WORKGROUP_SIZE * sizeof(float));
    vkcomp_push_constants_t pc = { .num_blocks = num_blocks };
    vkCmdPushConstants(cmd, ctx->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, num_blocks, 1, 1);

    e->compressed_offset = 0;
    e->compressed_size = size;
    return VK_SUCCESS;
}

VkResult vkcompress_read(VkCompressContext* ctx, VkCommandBuffer cmd,
                         vkcomp_buffer_id_t id,
                         VkBuffer dst, VkDeviceSize size, VkDeviceSize offset)
{
    if (!ctx || id == VKCOMP_INVALID_ID || !dst || !size) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    vkcomp_buffer_entry_t* e = NULL;
    for (uint32_t i = 0; i < ctx->buffer_count; i++) {
        if (ctx->buffers[i].id == id && ctx->buffers[i].valid) {
            e = &ctx->buffers[i];
            break;
        }
    }
    if (!e) return VK_ERROR_INITIALIZATION_FAILED;

    const uint32_t* decomp_spv = (e->compression_level > 3)
        ? vkcompress_spv_baseline_decompress_high
        : vkcompress_spv_baseline_decompress_fast;
    size_t decomp_size = (e->compression_level > 3)
        ? vkcompress_spv_baseline_decompress_high_size
        : vkcompress_spv_baseline_decompress_fast_size;

    VKCOMP_CHECK(ensure_shader_module(ctx->device, decomp_spv, decomp_size, &e->module_read));
    VKCOMP_CHECK(ensure_pipeline(ctx, &e->pipeline_read, e->module_read));

    if (!e->desc_set) {
        VkDescriptorSetAllocateInfo dsai = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = ctx->descriptor_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &ctx->set_layout,
        };
        VKCOMP_CHECK(vkAllocateDescriptorSets(ctx->device, &dsai, &e->desc_set));
    }

    /* binding 0 = comp_buffer (read), binding 1 = dst (write), binding 2 = meta */
    VkDescriptorBufferInfo buf_infos[3] = {
        { .buffer = e->comp_buffer, .offset = 0, .range = size },
        { .buffer = dst, .offset = offset, .range = size },
        { .buffer = e->meta_buffer, .offset = 0, .range = VKCOMP_MAX_BUFFERS * 2 * sizeof(uint32_t) },
    };
    VkWriteDescriptorSet writes[3] = {
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = e->desc_set,
          .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &buf_infos[0] },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = e->desc_set,
          .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &buf_infos[1] },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = e->desc_set,
          .dstBinding = 2, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &buf_infos[2] },
    };
    vkUpdateDescriptorSets(ctx->device, 3, writes, 0, NULL);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, e->pipeline_read);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layout, 0, 1, &e->desc_set, 0, NULL);

    uint32_t num_blocks = vkcomp_div_ceil((uint32_t)size, VKCOMP_WORKGROUP_SIZE * sizeof(float));
    vkcomp_push_constants_t pc = { .num_blocks = num_blocks };
    vkCmdPushConstants(cmd, ctx->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, num_blocks, 1, 1);

    return VK_SUCCESS;
}

VkResult vkcompress_load_catalog(VkCompressContext* ctx,
                                 const char* tag_prefix,
                                 const char* catalog_path)
{
    /* Stub: future impl reads JSON catalog → pre-register buffers */
    (void)ctx; (void)tag_prefix; (void)catalog_path;
    return VK_SUCCESS;
}

VkResult vkcompress_save_catalog(VkCompressContext* ctx,
                                 const char* catalog_path)
{
    /* Stub: future impl writes .catalog JSON with buffer metadata */
    (void)ctx; (void)catalog_path;
    return VK_SUCCESS;
}
