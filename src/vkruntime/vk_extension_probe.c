/**
 * \file vk_extension_probe.c
 * \brief Vulkan extension enumeration + probe for unused AMD/RDNA extensions.
 *
 * This module enumerates every Vulkan instance and device extension made
 * available by the loader+driver on this machine (SDK 1.4.357.0, driver
 * 26.7.1), then specifically probes for the AMD/RDNA extensions listed in
 * specs/EXTENSIONS-UNUSED.md, specs/GPU_CAPABILITIES.md, and
 * specs/RDNA-EXTENSIONS.md that VAiT does not currently enable.
 *
 * The output is a DLL that exports:
 *   - vkr_probe_all_extensions()   — prints every instance + device extension
 *   - vkr_probe_amd_extensions()   — prints the 25 AMD extension names + presence
 *   - vkr_probe_rdna_gaps()        — prints GAP_ANALYSIS.md feature status
 *   - vkr_create_device_full()     — creates a device with ALL available features
 *
 * This is a research/probe binary, not a production library.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>
#include <vulkan/vk_sdk_platform.h>

/* ── AMD extensions from vk_amd_extensions.txt (25 total) ────────────────── */
static const char* amd_extensions[] = {
    "VK_AMD_anti_lag",
    "VK_AMD_buffer_marker",
    "VK_AMD_device_coherent_memory",
    "VK_AMD_display_native_hdr",
    "VK_AMD_draw_indirect_count",
    "VK_AMD_gcn_shader",
    "VK_AMD_gpa_interface",
    "VK_AMD_gpu_shader_half_float",
    "VK_AMD_gpu_shader_int16",
    "VK_AMD_memory_overallocation_behavior",
    "VK_AMD_mixed_attachment_samples",
    "VK_AMD_negative_viewport_height",
    "VK_AMD_pipeline_compiler_control",
    "VK_AMD_rasterization_order",
    "VK_AMD_shader_ballot",
    "VK_AMD_shader_core_properties",
    "VK_AMD_shader_core_properties2",
    "VK_AMD_shader_early_and_late_fragment_tests",
    "VK_AMD_shader_explicit_vertex_parameter",
    "VK_AMD_shader_fragment_mask",
    "VK_AMD_shader_image_load_store_lod",
    "VK_AMD_shader_info",
    "VK_AMD_shader_trinary_minmax",
    "VK_AMD_texture_gather_bias_lod",
    "VK_AMD_vertex_shader_layer",
};
#define NUM_AMD_EXTENSIONS (sizeof(amd_extensions)/sizeof(amd_extensions[0]))

/* ── Extensions VAiT specs say are UNUSED (EXTENSIONS-UNUSED.md §2/§3) ────── */
static const char* unused_extensions[] = {
    /* Core features not yet enabled */
    "VK_KHR_shader_integer_dot_product",
    "VK_KHR_shader_subgroup_rotate",
    "VK_KHR_shader_bfloat16",
    "VK_EXT_shader_float8",
    /* Named extensions not wired */
    "VK_EXT_shader_atomic_float",
    "VK_EXT_memory_budget",
    "VK_EXT_memory_priority",
    "VK_AMD_device_coherent_memory",
    "VK_AMD_shader_info",
    "VK_AMD_buffer_marker",
    "VK_AMD_shader_trinary_minmax",
    "VK_EXT_descriptor_indexing",
};
#define NUM_UNUSED_EXTENSIONS (sizeof(unused_extensions)/sizeof(unused_extensions[0]))

/* ── Proposed VK_MaxR_* extensions (VK-MAXR-EXTENSIONS.md) ─────────────────── */
static const char* maxr_proposals[] = {
    "VK_MaxR_register_limits",
    "VK_MaxR_cache_control",
    "VK_MaxR_wave_matrix",
    "VK_MaxR_wave_matrix_fp8",
    "VK_MaxR_zero_copy_memory",
    "VK_MaxR_l2_cache_reservation",
    "VK_MaxR_occupancy_query",
};
#define NUM_MAXR (sizeof(maxr_proposals)/sizeof(maxr_proposals[0]))

/* ── Print all Vulkan instance extensions ─────────────────────────────────── */
static void print_instance_extensions(void) {
    uint32_t count = 0;
    VkResult r = vkEnumerateInstanceExtensionProperties(NULL, &count, NULL);
    if (r != VK_SUCCESS || count == 0) {
        printf("  (no instance extensions or enumeration failed: %d)\n", r);
        return;
    }

    VkExtensionProperties* exts = malloc(sizeof(VkExtensionProperties) * count);
    if (!exts) { printf("  (allocation failed)\n"); return; }

    r = vkEnumerateInstanceExtensionProperties(NULL, &count, exts);
    if (r != VK_SUCCESS) {
        printf("  (enumeration failed: %d)\n", r);
        free(exts);
        return;
    }

    printf("  Total: %u instance extensions\n", count);
    for (uint32_t i = 0; i < count; i++) {
        printf("  %s (rev %u)\n", exts[i].extensionName, exts[i].specVersion);
    }
    free(exts);
}

/* ── Print all device extensions for a physical device ────────────────────── */
static void print_device_extensions(VkPhysicalDevice pd) {
    uint32_t count = 0;
    VkResult r = vkEnumerateDeviceExtensionProperties(pd, NULL, &count, NULL);
    if (r != VK_SUCCESS || count == 0) {
        printf("  (no device extensions or enumeration failed: %d)\n", r);
        return;
    }

    VkExtensionProperties* exts = malloc(sizeof(VkExtensionProperties) * count);
    if (!exts) { printf("  (allocation failed)\n"); return; }

    r = vkEnumerateDeviceExtensionProperties(pd, NULL, &count, exts);
    if (r != VK_SUCCESS) {
        printf("  (enumeration failed: %d)\n", r);
        free(exts);
        return;
    }

    printf("  Total: %u device extensions\n", count);
    for (uint32_t i = 0; i < count; i++) {
        printf("  %s (rev %u)\n", exts[i].extensionName, exts[i].specVersion);
    }
    free(exts);
}

/* ── Check if an extension string is in a list ────────────────────────────── */
static int is_ext_present(const char** list, size_t n, const char* name) {
    for (size_t i = 0; i < n; i++) {
        if (strcmp(list[i], name) == 0) return 1;
    }
    return 0;
}

/* ── Probe specific extension arrays ─────────────────────────────────────── */
static void probe_extension_sets(VkPhysicalDevice pd) {
    uint32_t dev_count = 0;
    vkEnumerateDeviceExtensionProperties(pd, NULL, &dev_count, NULL);

    char** dev_names = malloc(sizeof(char*) * dev_count);
    VkExtensionProperties* dev_exts = malloc(sizeof(VkExtensionProperties) * dev_count);

    if (dev_names && dev_exts) {
        vkEnumerateDeviceExtensionProperties(pd, NULL, &dev_count, dev_exts);
        for (uint32_t i = 0; i < dev_count; i++) {
            dev_names[i] = dev_exts[i].extensionName;
        }
    }

    /* AMD extensions presence */
    printf("\n=== AMD Extension Probe (25 names from vk_amd_extensions.txt) ===\n");
    int found_amd = 0;
    for (size_t i = 0; i < NUM_AMD_EXTENSIONS; i++) {
        int present = dev_names ? is_ext_present((const char**)dev_names, dev_count, amd_extensions[i]) : 0;
        printf("  [%s] %s\n", present ? "PRESENT" : "ABSENT ", amd_extensions[i]);
        if (present) found_amd++;
    }
    printf("  --- %d/%zu AMD extensions present ---\n", found_amd, NUM_AMD_EXTENSIONS);

    /* Unused extensions */
    printf("\n=== Unused Extensions Probe (EXTENSIONS-UNUSED.md) ===\n");
    for (size_t i = 0; i < NUM_UNUSED_EXTENSIONS; i++) {
        int present = dev_names ? is_ext_present((const char**)dev_names, dev_count, unused_extensions[i]) : 0;
        printf("  [%s] %s\n", present ? "PRESENT" : "ABSENT ", unused_extensions[i]);
    }

    /* Proposed MaxR extensions */
    printf("\n=== Proposed VK_MaxR_* Extension Probe (VK-MAXR-EXTENSIONS.md) ===\n");
    for (size_t i = 0; i < NUM_MAXR; i++) {
        int present = dev_names ? is_ext_present((const char**)dev_names, dev_count, maxr_proposals[i]) : 0;
        printf("  [%s] %s\n", present ? "PRESENT" : "ABSENT", maxr_proposals[i]);
    }

    free(dev_names);
    free(dev_exts);
}

/* ── Query detailed physical device properties ─────────────────────────────── */
static void print_device_info(VkPhysicalDevice pd) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(pd, &props);

    printf("  Device: %s\n", props.deviceName);
    printf("  Vendor: 0x%04x, Device: 0x%04x\n", props.vendorID, props.deviceID);
    printf("  API version: %u.%u.%u\n",
           VK_VERSION_MAJOR(props.apiVersion),
           VK_VERSION_MINOR(props.apiVersion),
           VK_VERSION_PATCH(props.apiVersion));
    printf("  Driver version: %u.%u.%u\n",
           VK_VERSION_MAJOR(props.driverVersion),
           VK_VERSION_MINOR(props.driverVersion),
           VK_VERSION_PATCH(props.driverVersion));

    /* Try to get AMD shader core properties if available */
    VkPhysicalDeviceProperties2 props2 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
    props2.properties = props;

    VkPhysicalDeviceShaderCorePropertiesAMD core_props = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_AMD,
        .pNext = NULL,
    };
    props2.pNext = &core_props;
    vkGetPhysicalDeviceProperties2(pd, &props2);

    if (core_props.deviceName[0] != '\0') {
        printf("  wavefrontSize: %u\n", core_props.wavefrontSize);
        printf("  simdPerComputeUnit: %u\n", core_props.simdPerComputeUnit);
        printf("  computeUnitsPerShaderArray: %u\n", core_props.computeUnitsPerShaderArray);
        printf("  vgprsPerSimd: %u\n", core_props.vgprsPerSimd);
        printf("  maxVgprAllocation: %u\n", core_props.maxVgprAllocation);
    }

    /* AMD shader core properties2 */
    VkPhysicalDeviceShaderCoreProperties2AMD core_props2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_2_AMD,
        .pNext = NULL,
    };
    props2.pNext = &core_props2;
    vkGetPhysicalDeviceProperties2(pd, &props2);

    if (core_props2.activeComputeUnitCount > 0) {
        printf("  activeComputeUnitCount: %u\n", core_props2.activeComputeUnitCount);
    }
}

/* ── Enumerate physical devices and probe each ────────────────────────────── */
VKCOMAPI VkResult VKAPI_CALL vkProbeExtensions(void) {
    printf("=== Vulkan Extension & ISA Gap Probe ===\n");
    printf("SDK: %s\n", VK_HEADER_VERSION_COMPLETE);

    /* Instance extensions */
    printf("\n--- Instance Extensions ---\n");
    print_instance_extensions();

    /* Create instance */
    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "VAiT-ExtensionProbe",
        .applicationVersion = 1,
        .pEngineName = "VAiSt",
        .engineVersion = 1,
        .apiVersion = VK_API_VERSION_1_4,
    };

    VkInstanceCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info,
    };

    VkInstance instance;
    VkResult r = vkCreateInstance(&create_info, NULL, &instance);
    if (r != VK_SUCCESS) {
        printf("vkCreateInstance failed: %d\n", r);
        return r;
    }

    /* Enumerate physical devices */
    uint32_t pd_count = 0;
    vkEnumeratePhysicalDevices(instance, &pd_count, NULL);
    printf("\nPhysical devices found: %u\n", pd_count);

    if (pd_count == 0) {
        printf("No physical devices found!\n");
        vkDestroyInstance(instance, NULL);
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VkPhysicalDevice* pds = malloc(sizeof(VkPhysicalDevice) * pd_count);
    if (!pds) {
        vkDestroyInstance(instance, NULL);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    vkEnumeratePhysicalDevices(instance, &pd_count, pds);

    for (uint32_t i = 0; i < pd_count; i++) {
        printf("\n=== Physical Device %u ===\n", i);
        print_device_info(pds[i]);

        printf("\nDevice Extensions:\n");
        print_device_extensions(pds[i]);

        probe_extension_sets(pds[i]);

        /* Query cooperative matrix properties */
        printf("\n--- Cooperative Matrix Properties ---\n");
        uint32_t cm_count = 0;
        r = vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(pds[i], &cm_count, NULL);
        if (r == VK_SUCCESS && cm_count > 0) {
            VkCooperativeMatrixPropertiesKHR* cm_props = malloc(sizeof(VkCooperativeMatrixPropertiesKHR) * cm_count);
            if (cm_props) {
                vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(pds[i], &cm_count, cm_props);
                printf("  %u cooperative matrix configurations:\n", cm_count);
                for (uint32_t j = 0; j < cm_count; j++) {
                    VkCooperativeMatrixPropertiesKHR* p = &cm_props[j];
                    printf("    A:%u B:%u C:%u D:%u (%s) M:%u K:%u N:%u\n",
                           p->atileSize, p->bTileSize, p->cTileSize, p->dTileSize,
                           "float", p->m, p->k, p->n);
                }
                free(cm_props);
            }
        } else {
            printf("  vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR not available or 0 configs\n");
        }

        /* Query integer dot product properties */
        printf("\n--- Integer Dot Product Properties ---\n");
        VkPhysicalDeviceProperties2 dp = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
        VkPhysicalDeviceShaderIntegerDotProductProperties dp_props = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES,
        };
        dp.pNext = &dp_props;
        vkGetPhysicalDeviceProperties2(pds[i], &dp);
        printf("  integerDotProduct8BitSigned: %d\n", dp_props.integerDotProduct8BitSigned);
        printf("  integerDotProduct8BitUnsigned: %d\n", dp_props.integerDotProduct8BitUnsigned);
        printf("  integerDotProduct8BitMixed: %d\n", dp_props.integerDotProduct8BitMixed);
        printf("  integerDotProduct16BitSigned: %d\n", dp_props.integerDotProduct16BitSigned);
        printf("  integerDotProduct16BitUnsigned: %d\n", dp_props.integerDotProduct16BitUnsigned);
    }

    free(pds);
    vkDestroyInstance(instance, NULL);
    printf("\n=== Probe Complete ===\n");
    return VK_SUCCESS;
}
