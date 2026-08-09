# Graph Report - F:\VAiT\specs  (2026-08-09)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 2187 nodes · 5172 edges · 124 communities (121 shown, 3 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 52 edges (avg confidence: 0.81)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- hipDeviceProp_t
- __HOST_DEVICE__
- amd_hip_fp16.h
- VmaAllocator_T
- vk_mem_alloc.h
- hip_runtime_api.h
- VmaAllocationCreateInfo
- VmaDefragmentationContext_T
- VMA_CALL_PRE
- VmaJsonWriter
- VmaAllocator
- VmaVulkanFunctions
- __half
- VmaBlockMetadata_TLSF
- __CG_QUALIFIER__
- thread_block
- T
- .GetSize
- VmaBlockMetadata_Linear
- VmaAllocation_T
- Microcode Formats
- RDNA4 Instruction Set Architecture
- Vector Memory Operations (VMEM)
- Wave Matrix Multiply Accumulate (WMMA)
- VkDeviceSize
- VmaBlockVector
- amd_hip_cooperative_groups.h
- VmaBlockMetadata
- hipMemPoolProps
- VkResult
- Local Data Share (LDS)
- Wave State
- VmaIntrusiveLinkedList
- AMD Vulkan Extensions
- VmaVector
- size
- Configuring Vulkan Layers
- HIP_LAUNCH_CONFIG_st
- Externally Synchronized Parameters
- VmaDeviceMemoryBlock
- iterator
- VmaWin32Handle
- hipStreamMemOpWaitValueParams_t
- .num_threads
- hipLaunchParams_t
- VmaDetailedStatistics
- VmaRawList
- VmaDedicatedAllocationList
- hipArrayMapInfo
- ItemType
- VmaAllocatorCreateInfo
- LDS & GDS Instructions
- Vulkan Specification (v1.4)
- VmaAllocationInfo
- VmaRWMutex
- grid_group
- coalesced_group
- Vulkan API (Commands & Structures)
- CreateImage
- Global Data Share (GDS)
- VmaStatistics
- hipFunctionLaunchParams_t
- VmaPoolCreateInfo
- thread_block_tile_type
- VmaSmallVector
- thread_group
- hipExternalSemaphoreSignalNodeParams
- hipFuncAttributes
- VkAllocationCallbacks
- VMA_CLASS_NO_COPY_NO_MOVE
- VmaDefragmentationMove
- VmaPoolAllocator
- __clamp_01
- T
- Normative References
- VmaVirtualBlock_T
- VmaMappingHysteresis
- VmaList
- VmaDefragmentationInfo
- VmaPool_T
- VmaSuballocation
- hipExternalMemoryMipmappedArrayDesc_st
- Implicit Valid Usage
- hipAccessPolicyWindow
- hipBatchMemOpNodeParams
- hipExternalMemoryHandleDesc_st
- hipGraphInstantiateParams
- hipMemsetParams
- hipblasBfloat16
- VmaStlAllocator
- VmaCurrentBudgetData
- AtomicTransactionalIncrement
- VmaVectorInsertSorted
- hipExternalSemaphoreHandleDesc_st
- VmaVirtualBlockCreateInfo
- RDNA 2 Instruction Set Architecture Reference Guide
- VmaPnextChainFind
- hipPointerAttribute_t
- rocfft.h
- VmaAllocationInfo2
- VmaDefragmentationStats
- VmaVirtualAllocationCreateInfo
- hipEventRecordNodeParams
- hipGraphNodeParams
- VmaDeviceMemoryCallbacks
- __shfl
- hipExternalMemoryBufferDesc_st
- hipGraphEdgeData
- Appendix C: Compressed Image Formats
- hipHostNodeParams
- hipLaunchAttribute_st
- hipExternalSemaphoreSignalParams_st
- hipExternalSemaphoreWaitParams_st
- Query Commands (vkGet*/vkEnumerate*)
- DebugLogAllocation
- VmaMutexLock
- VmaTotalStatistics
- __float22half2_rn
- hipArrayMemoryRequirements
- VmaAllocationExtraData
- is_floating_point<_Float16>

## God Nodes (most connected - your core abstractions)
1. `hipDeviceProp_t` - 110 edges
2. `VmaAllocator_T` - 108 edges
3. `__half()` - 95 edges
4. `__half2` - 82 edges
5. `device` - 82 edges
6. `VmaBlockVector` - 70 edges
7. `VmaBlockMetadata_TLSF` - 52 edges
8. `VmaVulkanFunctions` - 45 edges
9. `VmaDeviceMemoryBlock` - 45 edges
10. `size` - 45 edges

## Surprising Connections (you probably didn't know these)
- `M0 Register` --semantically_similar_to--> `M0 Miscellaneous Register`  [INFERRED] [semantically similar]
  rdna2_isa.txt → rdna4_isa.txt
- `Vector GPR (VGPR)` --semantically_similar_to--> `Vector General Purpose Registers (VGPR)`  [INFERRED] [semantically similar]
  rdna2_isa.txt → rdna4_isa.txt
- `Scalar GPR (SGPR)` --semantically_similar_to--> `Scalar General Purpose Registers (SGPR)`  [INFERRED] [semantically similar]
  rdna2_isa.txt → rdna4_isa.txt
- `Local Data Share (LDS)` --conceptually_related_to--> `Barriers and Work-group Synchronization`  [INFERRED]
  rdna2_isa.txt → rdna4_isa.txt
- `Appendix E: Layers & Extensions` --references--> `AMD Vulkan Extensions`  [INFERRED]
  vk_coop_matrix.txt → vk_amd_extensions.txt

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **LDS Access Types** — rdna2_isa_lds_direct_read, rdna2_isa_lds_parameter_read, rdna2_isa_lds_indexed_atomic, rdna2_isa_lds_lane_permute [EXTRACTED 1.00]
- **Scalar Instruction Encoding Formats** — rdna2_isa_sop2, rdna2_isa_sopk, rdna2_isa_sop1, rdna2_isa_sopc, rdna2_isa_sopp [EXTRACTED 1.00]
- **Export Dependency Completion Flow (EXPCNT / S_WAITCNT)** — rdna2_isa_export, rdna2_isa_exec_mask, rdna2_isa_vgpr, rdna2_isa_s_waitcnt [EXTRACTED 1.00]
- **Atomic Read-Modify-Write Opcode Family Across Memory Formats** — rdna2_isa_buffer_atomic, rdna2_isa_image_atomic, rdna2_isa_flat_atomic, rdna2_isa_global_atomic [EXTRACTED 0.90]
- **LDS/GDS Data Share Instruction Suite** — rdna2_isa_ds_section, rdna2_isa_ds_swizzle_b32, rdna2_isa_ds_consume, rdna2_isa_ds_append, rdna2_isa_ds_ordered_count, rdna2_isa_ds_permute_b32, rdna2_isa_ds_bpermute_b32, rdna2_isa_gws [EXTRACTED 0.90]
- **Wavefront Cross-Lane Data Movement Operations** — rdna2_isa_v_permlane16_b32, rdna2_isa_v_permlanex16_b32, rdna2_isa_ds_permute_b32, rdna2_isa_ds_bpermute_b32, rdna2_isa_ds_swizzle_b32, rdna2_isa_v_mbcnt_lo_u32_b32, rdna2_isa_v_mbcnt_hi_u32_b32 [INFERRED 0.75]
- **RDNA2 Wave Execution Model** — rdna2_isa_wave32, rdna2_isa_wave64, rdna2_isa_exec_mask, rdna2_isa_compute_shader, rdna2_isa_subvector_execution [EXTRACTED 0.90]
- **RDNA2 Memory Hierarchy** — rdna2_isa_lds, rdna2_isa_gds, rdna2_isa_device_memory, rdna2_isa_scratch_memory, rdna2_isa_buffer_resource, rdna2_isa_image_resource [INFERRED 0.80]
- **RDNA2 ISA Instruction Families** — rdna2_isa_scalar_alu, rdna2_isa_vector_alu, rdna2_isa_scalar_memory, rdna2_isa_vector_memory, rdna2_isa_flat_memory, rdna2_isa_data_share, rdna2_isa_export_instructions [EXTRACTED 0.95]
- **Scalar and Vector ALU Execution Model** — rdna4_isa_scalar_alu_operations, rdna4_isa_vector_alu_operations, rdna4_isa_wave_state [INFERRED 0.80]
- **Memory Hierarchy and Data Sharing** — rdna4_isa_cache_system, rdna4_isa_scalar_memory_operations, rdna4_isa_vector_memory_buffer_instructions, rdna4_isa_vector_memory_image_instructions, rdna4_isa_flat_global_scratch_operations [INFERRED 0.75]
- **Instruction Groups Encoded by Microcode Formats** — rdna4_isa_scalar_alu_operations, rdna4_isa_vector_alu_operations, rdna4_isa_scalar_memory_operations, rdna4_isa_vector_memory_buffer_instructions, rdna4_isa_vector_memory_image_instructions, rdna4_isa_exports, rdna4_isa_microcode_formats [INFERRED 0.80]
- **VALU Encoding Family** — rdna4_isa_valu, rdna4_isa_vop1, rdna4_isa_vop2, rdna4_isa_vopc, rdna4_isa_vop3, rdna4_isa_vop3p, rdna4_isa_vopd, rdna4_isa_dpp [EXTRACTED 0.95]
- **Memory Dependency Counters** — rdna4_isa_kmcnt, rdna4_isa_loadcnt, rdna4_isa_storecnt, rdna4_isa_dscnt, rdna4_isa_expcnt [EXTRACTED 0.95]
- **WMMA Matrix Multiply Subsystem** — rdna4_isa_wmma, rdna4_isa_swmmac, rdna4_isa_matrix_storage, rdna4_isa_sparse_matrix, rdna4_isa_global_load_tr, rdna4_isa_vop3p [EXTRACTED 0.90]
- **Layer configuration priority order** — vulkan_layers_whitepaper_vk_instance_layers, vulkan_layers_whitepaper_vk_loader_settings_json, vulkan_layers_whitepaper_vk_layer_settings_txt, vulkan_layers_whitepaper_vk_ext_layer_settings [EXTRACTED 1.00]
- **Generated layer settings artifacts** — vulkan_layers_whitepaper_vulkan_configurator, vulkan_layers_whitepaper_vk_layer_settings_txt, vulkan_layers_whitepaper_vulkan_layer_settings_hpp, vulkan_layers_whitepaper_vk_layer_settings_sh [EXTRACTED 1.00]
- **Vulkan Layer Settings library consumers** — vulkan_layers_whitepaper_vulkan_layer_settings_library, vulkan_layers_whitepaper_khronos_validation_layer, vulkan_layers_whitepaper_khronos_profiles_layer [EXTRACTED 1.00]
- **AMD Vulkan Extension Family** — vk_amd_extensions_vk_amd_anti_lag, vk_amd_extensions_vk_amd_buffer_marker, vk_amd_extensions_vk_amd_device_coherent_memory, vk_amd_extensions_vk_amd_display_native_hdr, vk_amd_extensions_vk_amd_draw_indirect_count, vk_amd_extensions_vk_amd_gcn_shader, vk_amd_extensions_vk_amd_gpa_interface, vk_amd_extensions_vk_amd_gpu_shader_half_float, vk_amd_extensions_vk_amd_gpu_shader_int16, vk_amd_extensions_vk_amd_memory_overallocation_behavior, vk_amd_extensions_vk_amd_mixed_attachment_samples, vk_amd_extensions_vk_amd_negative_viewport_height, vk_amd_extensions_vk_amd_pipeline_compiler_control, vk_amd_extensions_vk_amd_rasterization_order, vk_amd_extensions_vk_amd_shader_ballot, vk_amd_extensions_vk_amd_shader_core_properties_2, vk_amd_extensions_vk_amd_shader_core_properties, vk_amd_extensions_vk_amd_shader_early_and_late_fragment_tests, vk_amd_extensions_vk_amd_shader_explicit_vertex_parameter, vk_amd_extensions_vk_amd_shader_fragment_mask, vk_amd_extensions_vk_amd_shader_image_load_store_lod, vk_amd_extensions_vk_amd_shader_info, vk_amd_extensions_vk_amd_shader_trinary_minmax, vk_amd_extensions_vk_amd_texture_gather_bias_lod [INFERRED 0.90]
- **Vulkan Command Form Conventions** — vk_api_structs_object_creation, vk_api_structs_pool_allocation, vk_api_structs_command_recording, vk_api_structs_query_commands [EXTRACTED 0.95]
- **Externally Synchronized Threading Model** — vk_api_structs_threading_behavior, vk_api_structs_externally_synchronized, vk_api_structs_implicit_external_sync, vk_api_structs_command_pool, vk_api_structs_command_buffer [EXTRACTED 0.90]

## Communities (124 total, 3 thin omitted)

### Community 0 - "hipDeviceProp_t"
Cohesion: 0.02
Nodes (110): hipDeviceArch_t, hipUUID, hipDeviceProp_t, accessPolicyMaxWindowSize, arch, asicRevision, asyncEngineCount, canMapHostMemory (+102 more)

### Community 1 - "__HOST_DEVICE__"
Cohesion: 0.05
Nodes (76): __HOST_DEVICE__, __float2half(), __float2half2_rn(), __float2half_rn(), __h2div(), __habs(), __habs2(), __hadd2() (+68 more)

### Community 2 - "amd_hip_fp16.h"
Cohesion: 0.06
Nodes (68): amd_mixed_dot(), atomicAdd(), __float2half_rd(), __float2half_ru(), __float2half_rz(), h2ceil(), h2cos(), h2exp() (+60 more)

### Community 3 - "VmaAllocator_T"
Cohesion: 0.04
Nodes (53): PoolList, VmaAlignDown(), VmaAllocator_T, CalculateGlobalMemoryTypeBits, FlushOrInvalidateAllocations, GetFlushOrInvalidateRange, ImportVulkanFunctions, ImportVulkanFunctions_Custom (+45 more)

### Community 4 - "vk_mem_alloc.h"
Cohesion: 0.06
Nodes (26): Free, CalculateGpuDefragmentationMemoryTypeBits, DestroyPool, GetGpuDefragmentationMemoryTypeBits, GetPoolStatistics, AddStatistics, Remove, VmaClearStatistics() (+18 more)

### Community 5 - "hip_runtime_api.h"
Cohesion: 0.04
Nodes (47): hipGraph_t, hipLaunchConfig_t, hipMipmappedArray_const_t, __host__, hipBindTexture(), hipBindTexture2D(), hipBindTextureToMipmappedArray(), hipChildGraphNodeParams (+39 more)

### Community 6 - "VmaAllocationCreateInfo"
Cohesion: 0.08
Nodes (36): BaseType, NewT, FindMemoryPreferences(), vmaAllocateDedicatedMemory(), vmaAllocateMemory(), vmaAllocateMemoryForBuffer(), vmaAllocateMemoryForImage(), vmaAllocateMemoryPages() (+28 more)

### Community 7 - "VmaDefragmentationContext_T"
Cohesion: 0.12
Nodes (37): CounterStatus, MoveAllocationData, StateBalanced, GetOffset, GetAllocationListBegin, GetNextAllocation, VmaDefragmentationContext_T, AllocInOtherBlock (+29 more)

### Community 8 - "VMA_CALL_PRE"
Cohesion: 0.09
Nodes (38): SwapBlockAllocation, VmaAllocationObjectAllocator::Allocate(), FreeMemory, vmaBindImageMemory(), vmaBindImageMemory2(), vmaCalculateStatistics(), vmaCalculateVirtualBlockStatistics(), vmaCheckCorruption() (+30 more)

### Community 9 - "VmaJsonWriter"
Cohesion: 0.15
Nodes (36): StackItem, VmaAllocation_T::PrintParameters(), VmaAllocator_T::PrintDetailedMap(), VmaBlockMetadata::PrintDetailedMap_Allocation(), VmaBlockMetadata::PrintDetailedMap_Begin(), VmaBlockMetadata::PrintDetailedMap_End(), VmaBlockMetadata::PrintDetailedMap_UnusedRange(), VmaBlockVector::PrintDetailedMap() (+28 more)

### Community 10 - "VmaAllocator"
Cohesion: 0.09
Nodes (39): vma_delete(), Destroy, FreeName, GetParentPool, SetName, CheckPoolCorruption, Init, vmaBeginDefragmentation() (+31 more)

### Community 11 - "VmaVulkanFunctions"
Cohesion: 0.05
Nodes (40): PFN_vkAllocateMemory, PFN_vkBindBufferMemory, PFN_vkBindImageMemory, PFN_vkCmdCopyBuffer, PFN_vkCreateBuffer, PFN_vkCreateImage, PFN_vkDestroyBuffer, PFN_vkDestroyImage (+32 more)

### Community 12 - "__half"
Cohesion: 0.06
Nodes (42): MaskT, __half(), __half2float(), __half2ll_rd(), __half2ll_rn(), __half2ll_ru(), __half2ll_rz(), __half2short_rd() (+34 more)

### Community 13 - "VmaBlockMetadata_TLSF"
Cohesion: 0.13
Nodes (32): Block, VmaBlockMetadata_TLSF, Alloc, CheckBlock, CreateAllocationRequest, FindFreeBlock, Free, GetListIndex (+24 more)

### Community 14 - "__CG_QUALIFIER__"
Cohesion: 0.07
Nodes (22): __CG_QUALIFIER__, CGTy, g, bit_and, bit_or, bit_xor, greater, __hip_uint32_t group_size() (+14 more)

### Community 15 - "thread_block"
Cohesion: 0.08
Nodes (10): arrival_token, __CG_STATIC_QUALIFIER__, cluster_group, arrival_token, map_shared_rank(), thread_block, arrival_token, __hip_uint32_t (+2 more)

### Community 16 - "T"
Cohesion: 0.12
Nodes (33): F, hipArray_const_t, hipError_t, hipLaunchParams, hipMemcpyKind, hipMemPool_t, hipStream_t, T() (+25 more)

### Community 17 - ".GetSize"
Cohesion: 0.11
Nodes (29): GetAllocHandle, GetMappedData, InitBlockAllocation, VmaAllocationRequest, algorithmData, allocHandle, customData, item (+21 more)

### Community 18 - "VmaBlockMetadata_Linear"
Cohesion: 0.08
Nodes (28): SECOND_VECTOR_MODE, SuballocationVectorType, VmaBlockMetadata_Linear, GetAllocationCount, GetAllocationInfo, GetAllocationListBegin, GetAllocationUserData, GetFreeRegionsCount (+20 more)

### Community 19 - "VmaAllocation_T"
Cohesion: 0.10
Nodes (23): ALLOCATION_TYPE, VmaAllocation_T, BlockAllocMap, BlockAllocUnmap, DedicatedAllocMap, DedicatedAllocUnmap, EnsureExtraData, InitDedicatedAllocation (+15 more)

### Community 20 - "Microcode Formats"
Cohesion: 0.13
Nodes (31): BUFFER_ATOMIC Family, Compute Shaders, Dot Product ALU Instructions (V_DOT*), Data Parallel Processing (DPP), EXEC Mask, FLAT/GLOBAL/SCRATCH Format, GLOBAL Instructions, GLOBAL_ATOMIC Family (+23 more)

### Community 21 - "RDNA4 Instruction Set Architecture"
Cohesion: 0.11
Nodes (31): FLAT Instructions, FLAT_ATOMIC Family, AMD Accelerated Parallel Processing OpenCL Programming Guide, Memory Aperture Registers (SH_MEM_BASES), Microsoft DirectX Reference, RDNA4 Instruction Set Architecture, DScnt Dependency Counter, Export Operations (+23 more)

### Community 22 - "Vector Memory Operations (VMEM)"
Cohesion: 0.09
Nodes (28): Buffer Resource (V#), Bounding Volume Hierarchy (BVH), GLC/DLC/SLC Coherency Bits, IMAGE_ATOMIC Family, IMAGE_BVH64_INTERSECT_RAY, IMAGE_BVH_INTERSECT_RAY, IMAGE_GATHER4 Family, Image Resource (T#) (+20 more)

### Community 23 - "Wave Matrix Multiply Accumulate (WMMA)"
Cohesion: 0.09
Nodes (28): Packed Math, 16-bit Math and VGPR Halves, 8-bit Math: FP8 and BF8, AMD Matrix Instruction Calculator, BF8 Data Format (E5M2), Data Conversion (CVT) Operations, DOT Product Instructions, Data Parallel Primitives (DPP) (+20 more)

### Community 24 - "VkDeviceSize"
Cohesion: 0.17
Nodes (23): RegionInfo, ValidationContext, VmaAlignUp(), CalcPreferredBlockSize, VmaBlockBufferImageGranularity, AllocPage, AllocPages, CheckConflictAndAlignUp (+15 more)

### Community 25 - "VmaBlockVector"
Cohesion: 0.08
Nodes (20): VmaBlockVector, IsEmpty, m_Algorithm, m_Blocks, m_BufferImageGranularity, m_ExplicitBlockSize, m_hAllocator, m_hParentPool (+12 more)

### Community 26 - "amd_hip_cooperative_groups.h"
Cohesion: 0.12
Nodes (24): GrandParentCGTy, parent, ParentCGTy, ParentSize, all(), any(), ballot(), build_mask() (+16 more)

### Community 27 - "VmaBlockMetadata"
Cohesion: 0.08
Nodes (25): VmaBlockMetadata, AddDetailedStatistics, AddStatistics, Alloc, CheckCorruption, Clear, CreateAllocationRequest, DebugLogAllAllocations (+17 more)

### Community 28 - "hipMemPoolProps"
Cohesion: 0.09
Nodes (25): hipMemAccessFlags, hipMemAllocationHandleType, hipMemAllocationType, hipMemLocation, hipMemAccessDesc, flags, location, hipMemAllocationProp (+17 more)

### Community 29 - "VkResult"
Cohesion: 0.14
Nodes (23): GetExternalMemoryHandleTypeFlags(), GetMemory, AllocateDedicatedMemory, AllocateDedicatedMemoryPage, AllocateVulkanMemory, CheckCorruption, CreatePool, FreeDedicatedMemory (+15 more)

### Community 30 - "Local Data Share (LDS)"
Cohesion: 0.14
Nodes (24): LDS Bank Conflict, Compute Unit (CU), LDS Modes: CU vs WGP, Device Memory, Data Share Instructions (DS_*), Flat Memory Instructions, Local Data Share (LDS), LDS Indexed and Atomic Access (+16 more)

### Community 31 - "Wave State"
Cohesion: 0.13
Nodes (24): Branching, Instruction Clauses, Mode Register, Program Counter (PC), Program Flow Control, S_ENDPGM, S_SUBVECTOR_LOOP_BEGIN/END, S_VERSION (+16 more)

### Community 32 - "VmaIntrusiveLinkedList"
Cohesion: 0.10
Nodes (23): VmaIntrusiveLinkedList, InsertAfter, InsertBefore, VmaIntrusiveLinkedList<ItemTypeTraits>::InsertAfter(), VmaIntrusiveLinkedList<ItemTypeTraits>::InsertBefore(), VmaIntrusiveLinkedList<ItemTypeTraits>::VmaIntrusiveLinkedList(), m_Back, m_Count (+15 more)

### Community 33 - "AMD Vulkan Extensions"
Cohesion: 0.09
Nodes (22): AMD Vulkan Extensions, VK_AMD_ANTI_LAG, VK_AMD_DEVICE_COHERENT_MEMORY, VK_AMD_DISPLAY_NATIVE_HDR, VK_AMD_GCN_SHADER, VK_AMD_GPU_SHADER_HALF_FLOAT, VK_AMD_GPU_SHADER_INT16, VK_AMD_MEMORY_OVERALLOCATION_BEHAVIOR (+14 more)

### Community 34 - "VmaVector"
Cohesion: 0.11
Nodes (15): AllocatorT, remove, VmaSmallVector<T, AllocatorT, N>::VmaSmallVector(), VmaVector, insert, m_Allocator, m_Capacity, m_Count (+7 more)

### Community 35 - "size"
Cohesion: 0.16
Nodes (17): size, CleanupAfterFree, Free, VmaBlockMetadata_Linear::PrintDetailedMap(), ShouldCompact1st, VmaPoolAllocator<T>::Alloc(), VmaPoolAllocator<T>::Free(), resize (+9 more)

### Community 36 - "Configuring Vulkan Layers"
Cohesion: 0.19
Nodes (21): Configuring Vulkan Layers, Khronos Profiles Layer, Khronos Validation Layer, Layer Manifest JSON, VK_ADD_LAYER_PATH, VK_EXT_layer_settings, VK_INSTANCE_LAYERS, VkLayer_override.json (deprecated) (+13 more)

### Community 37 - "HIP_LAUNCH_CONFIG_st"
Cohesion: 0.11
Nodes (19): hipLaunchAttribute, HIP_LAUNCH_CONFIG_st, attrs, blockDimX, blockDimY, blockDimZ, gridDimX, gridDimY (+11 more)

### Community 38 - "Externally Synchronized Parameters"
Cohesion: 0.16
Nodes (19): VK_AMD_BUFFER_MARKER, VK_AMD_DRAW_INDIRECT_COUNT, VK_AMD_GPA_INTERFACE, VkCommandBuffer (primary/secondary, render pass scope), VkCommandPool, Command Recording (vkCmd*), Externally Synchronized Parameters, Implicit Externally Synchronized Parameters (+11 more)

### Community 39 - "VmaDeviceMemoryBlock"
Cohesion: 0.11
Nodes (13): VmaDeviceMemoryBlock, m_Handle, m_hMemory, m_hParentPool, m_Id, m_IsMapped, m_MapAndBindMutex, m_MapCount (+5 more)

### Community 40 - "iterator"
Cohesion: 0.17
Nodes (6): const_iterator, iterator, VectorT, DebugLogAllAllocations, VmaSuballocationItemSizeLess, VmaVectorRemoveSorted()

### Community 41 - "VmaWin32Handle"
Cohesion: 0.22
Nodes (13): HANDLE, PFN_vkGetMemoryWin32HandleKHR, VmaAllocation_T::GetWin32Handle(), VmaDeviceMemoryBlock::CreateWin32Handle(), VmaWin32Handle, m_hHandle, m_IsNTHandle, m_Mutex (+5 more)

### Community 42 - "hipStreamMemOpWaitValueParams_t"
Cohesion: 0.12
Nodes (18): hipDeviceptr_t, hipStreamBatchMemOpType, hipStreamMemOpFlushRemoteWritesParams_t, flags, operation, hipStreamMemOpMemoryBarrierParams_t, flags, operation (+10 more)

### Community 43 - ".num_threads"
Cohesion: 0.12
Nodes (5): shfl(), tile_base, numThreads, tiled_group, tiled_group

### Community 44 - "hipLaunchParams_t"
Cohesion: 0.11
Nodes (18): dim3, x, y, z, hipKernelNodeParams, blockDim, extra, func (+10 more)

### Community 45 - "VmaDetailedStatistics"
Cohesion: 0.19
Nodes (17): VmaAddDetailedStatisticsAllocation(), VmaAddDetailedStatisticsUnusedRange(), CalculatePoolStatistics, AddDetailedStatistics, AddDetailedStatistics, VmaBlockMetadata_TLSF::PrintDetailedMap(), AddDetailedStatistics, VmaClearDetailedStatistics() (+9 more)

### Community 46 - "VmaRawList"
Cohesion: 0.12
Nodes (14): VmaRawList, Clear, InsertAfter, InsertBefore, m_Count, m_ItemAllocator, m_pAllocationCallbacks, m_pBack (+6 more)

### Community 47 - "VmaDedicatedAllocationList"
Cohesion: 0.15
Nodes (10): DedicatedAllocationLinkedList, CalculateStatistics, VmaDedicatedAllocationList, AddStatistics, IsEmpty, m_AllocationList, m_Mutex, m_UseMutex (+2 more)

### Community 48 - "hipArrayMapInfo"
Cohesion: 0.12
Nodes (16): hipArraySparseSubresourceType, hipMemHandleType, hipMemOperationType, hipResourceType, hipArrayMapInfo, deviceBitMask, flags, memHandle (+8 more)

### Community 49 - "ItemType"
Cohesion: 0.14
Nodes (8): ItemType, HasEmptyBlock, VmaDedicatedAllocationList::~VmaDedicatedAllocationList(), VmaDedicatedAllocationListItemTraits, VmaIntrusiveLinkedList<ItemTypeTraits>::PushBack(), VmaIntrusiveLinkedList<ItemTypeTraits>::PushFront(), VmaIntrusiveLinkedList<ItemTypeTraits>::Remove(), VmaIntrusiveLinkedList<ItemTypeTraits>::RemoveAll()

### Community 50 - "VmaAllocatorCreateInfo"
Cohesion: 0.15
Nodes (15): pDeviceMemoryCallbacks, physicalDevice, pVulkanFunctions, VmaAllocatorCreateInfo, flags, preferredLargeHeapBlockSize, VMA_NOT_NULL, VMA_NULLABLE (+7 more)

### Community 51 - "LDS & GDS Instructions"
Cohesion: 0.16
Nodes (16): DS_APPEND, DS_BPERMUTE_B32, DS_CONSUME, DS_GWS_BARRIER, DS_GWS_INIT, DS_GWS_SEMA_RELEASE_ALL, DS_GWS_SEMA_V, DS_ORDERED_COUNT (+8 more)

### Community 52 - "Vulkan Specification (v1.4)"
Cohesion: 0.12
Nodes (16): Appendix H: API Boilerplate, Cooperative Matrix Memory Access, Appendix D: Core Revisions, Appendix K: Credits, Document Conventions, Chapter 2: Introduction, Appendix I: Invariance, Appendix E: Layers & Extensions (+8 more)

### Community 53 - "VmaAllocationInfo"
Cohesion: 0.15
Nodes (24): deviceMemory, pMappedData, InitBufferUsage(), VmaAllocationInfo, memoryType, offset, size, VMA_NULLABLE (+16 more)

### Community 54 - "VmaRWMutex"
Cohesion: 0.16
Nodes (8): mutex, shared_mutex, SRWLOCK, VmaMutex, m_Mutex, VmaRWMutex, m_Lock, m_Mutex

### Community 55 - "grid_group"
Cohesion: 0.14
Nodes (5): __hip_uint32_t, grid_group, arrival_token, __hip_uint32_t, this_grid()

### Community 56 - "coalesced_group"
Cohesion: 0.16
Nodes (6): lane_mask, coalesced_group, coalesced_group, coalesced_group coalesced_threads(), coalesced_group tiled_partition(), tiled_group tiled_partition()

### Community 57 - "Vulkan API (Commands & Structures)"
Cohesion: 0.18
Nodes (14): pAllocator / Memory Allocation, Vk*CreateInfo Structures, Memory Barriers, Object Creation/Destruction Commands (vkCreate*/vkDestroy*), Pool Object Allocation (vkAllocate*/vkFree*), Protected Memory, Threading Behavior, vkBindBufferMemory (+6 more)

### Community 58 - "CreateImage"
Cohesion: 0.40
Nodes (11): InitImageUsage(), BindImageMemory, BindVulkanImage, CreateImage, vmaCreateAliasingImage(), vmaCreateAliasingImage2(), vmaCreateDedicatedImage(), vmaCreateImage() (+3 more)

### Community 59 - "Global Data Share (GDS)"
Cohesion: 0.18
Nodes (13): RDNA 2 Instruction Set Architecture, Data Share Operations (LDS/GDS), Data Dependency Resolution, DS (LDS/GDS) Format, EXP Format, EXPORT Instructions, Export Instructions (EXP), Global Data Share (GDS) (+5 more)

### Community 60 - "VmaStatistics"
Cohesion: 0.15
Nodes (13): VmaAddDetailedStatistics(), VmaAddStatistics(), GetHeapBudgets, AddStatistics, VmaBudget, budget, statistics, usage (+5 more)

### Community 61 - "hipFunctionLaunchParams_t"
Cohesion: 0.17
Nodes (12): hipFunction_t, hipFunctionLaunchParams_t, blockDimX, blockDimY, blockDimZ, function, gridDimX, gridDimY (+4 more)

### Community 62 - "VmaPoolCreateInfo"
Cohesion: 0.17
Nodes (12): pMemoryAllocateNext, VmaPoolCreateInfo, blockSize, flags, maxBlockCount, memoryTypeIndex, minAllocationAlignment, minBlockCount (+4 more)

### Community 63 - "thread_block_tile_type"
Cohesion: 0.17
Nodes (6): parent_group_info, thread_block_tile_type, numThreads, thread_block_tile_type<tileSize, void>, numThreads, tileSize

### Community 64 - "VmaSmallVector"
Cohesion: 0.17
Nodes (10): SrcAllocatorT, SrcN, SrcT, VmaSmallVector, clear, insert, m_Count, m_DynamicArray (+2 more)

### Community 65 - "thread_group"
Cohesion: 0.20
Nodes (9): group_type, __hip_uint64_t, thread_group, coalesced_info, __hip_uint32_t, _mask, _num_threads, thread_group tiled_partition() (+1 more)

### Community 66 - "hipExternalSemaphoreSignalNodeParams"
Cohesion: 0.18
Nodes (11): hipExternalSemaphore_t, hipExternalSemaphoreSignalParams, hipExternalSemaphoreWaitParams, hipExternalSemaphoreSignalNodeParams, extSemArray, numExtSems, paramsArray, hipExternalSemaphoreWaitNodeParams (+3 more)

### Community 67 - "hipFuncAttributes"
Cohesion: 0.18
Nodes (11): hipFuncAttributes, binaryVersion, cacheModeCA, constSizeBytes, localSizeBytes, maxDynamicSharedSizeBytes, maxThreadsPerBlock, numRegs (+3 more)

### Community 68 - "VkAllocationCallbacks"
Cohesion: 0.27
Nodes (8): vma_delete_array(), VmaAllocate(), VmaAllocateArray(), Destroy, VmaBlockMetadata_TLSF::VmaBlockMetadata_TLSF(), VmaMalloc(), VmaRawList<T>::VmaRawList(), VkAllocationCallbacks

### Community 69 - "VMA_CLASS_NO_COPY_NO_MOVE"
Cohesion: 0.25
Nodes (6): VmaMutexLockRead, m_pMutex, VmaMutexLockWrite, m_pMutex, VMA_CLASS_NO_COPY_NO_MOVE, VMA_RW_MUTEX

### Community 70 - "VmaDefragmentationMove"
Cohesion: 0.20
Nodes (10): dstTmpAllocation, srcAllocation, VmaDefragmentationMove, operation, VMA_NOT_NULL, VmaDefragmentationPassMoveInfo, moveCount, VMA_NULLABLE (+2 more)

### Community 71 - "VmaPoolAllocator"
Cohesion: 0.20
Nodes (9): ItemBlock, VmaAllocationObjectAllocator, m_Allocator, m_Mutex, VmaPoolAllocator, Free, m_FirstBlockCapacity, m_ItemBlocks (+1 more)

### Community 72 - "__clamp_01"
Cohesion: 0.20
Nodes (10): __clamp_01(), __hadd(), __hadd_sat(), __hfma2(), __hfma2_sat(), __hlt(), __hmul(), __hmul2() (+2 more)

### Community 73 - "T"
Cohesion: 0.27
Nodes (6): T, VmaDivideRoundingUp(), VmaRoundDiv(), VmaStlAllocator<T>::allocate(), VmaValidatePointerArray(), VmaVectorInsert()

### Community 74 - "Normative References"
Cohesion: 0.20
Nodes (10): AV1 Bitstream & Decoding Process Specification, ITU-T H.264 Advanced Video Coding, ITU-T H.265 High Efficiency Video Coding, IEEE Standard for Floating-Point Arithmetic (IEEE Std 754-2008), Khronos Data Format Specification v1.3, Architecture of the Vulkan Loader Interfaces, Normative References, SPIR-V Specification v1.6.7 (+2 more)

### Community 75 - "VmaVirtualBlock_T"
Cohesion: 0.20
Nodes (8): Clear, Clear, VmaPoolAllocator<T>::VmaPoolAllocator(), VmaVirtualBlock_T, GetAllocationCallbacks, m_AllocationCallbacks, m_AllocationCallbacksSpecified, m_Metadata

### Community 76 - "VmaMappingHysteresis"
Cohesion: 0.29
Nodes (5): VmaMappingHysteresis, COUNTER_MIN_EXTRA_MAPPING, m_ExtraMapping, m_MajorCounter, m_MinorCounter

### Community 77 - "VmaList"
Cohesion: 0.42
Nodes (4): const_reverse_iterator, reverse_iterator, VmaList, m_RawList

### Community 78 - "VmaDefragmentationInfo"
Cohesion: 0.22
Nodes (9): pBreakCallbackUserData, PFN_vmaCheckDefragmentationBreakFunction, pfnBreakCallback, VmaDefragmentationInfo, flags, maxAllocationsPerPass, maxBytesPerPass, VMA_NULLABLE (+1 more)

### Community 79 - "VmaPool_T"
Cohesion: 0.22
Nodes (7): VmaPool_T, m_BlockVector, m_DedicatedAllocations, m_Id, m_Name, m_NextPool, m_PrevPool

### Community 80 - "VmaSuballocation"
Cohesion: 0.22
Nodes (7): VmaSuballocation, offset, size, type, userData, VmaSuballocationOffsetGreater, VmaSuballocationOffsetLess

### Community 81 - "hipExternalMemoryMipmappedArrayDesc_st"
Cohesion: 0.25
Nodes (8): hipChannelFormatDesc, hipExtent, hipExternalMemoryMipmappedArrayDesc_st, extent, flags, formatDesc, numLevels, offset

### Community 82 - "Implicit Valid Usage"
Cohesion: 0.29
Nodes (8): Valid Usage for Enumerated Types, Implicit Valid Usage, Valid Usage for Object Handles, Valid Usage for Pointers & Strings, Valid Usage, Validation Layers, VkFlags (32-bit bitmask), VkFlags64 (64-bit bitmask)

### Community 83 - "hipAccessPolicyWindow"
Cohesion: 0.29
Nodes (7): hipAccessProperty, hipAccessPolicyWindow, base_ptr, hitProp, hitRatio, missProp, num_bytes

### Community 84 - "hipBatchMemOpNodeParams"
Cohesion: 0.29
Nodes (7): hipCtx_t, hipStreamBatchMemOpParams, hipBatchMemOpNodeParams, count, ctx, flags, paramArray

### Community 85 - "hipExternalMemoryHandleDesc_st"
Cohesion: 0.29
Nodes (7): hipExternalMemoryHandleType, hipExternalMemoryHandleDesc_st, flags, handle, reserved, size, type

### Community 86 - "hipGraphInstantiateParams"
Cohesion: 0.29
Nodes (7): hipGraphInstantiateResult, hipGraphNode_t, hipGraphInstantiateParams, errNode_out, flags, result_out, uploadStream

### Community 87 - "hipMemsetParams"
Cohesion: 0.29
Nodes (7): hipMemsetParams, dst, elementSize, height, pitch, value, width

### Community 89 - "VmaStlAllocator"
Cohesion: 0.33
Nodes (6): U, operator==(), VmaStlAllocator, allocate, deallocate, m_pCallbacks

### Community 90 - "VmaCurrentBudgetData"
Cohesion: 0.29
Nodes (7): VmaCurrentBudgetData, m_AllocationBytes, m_AllocationCount, m_BlockBytes, m_BlockCount, VMA_ATOMIC_UINT32, VMA_ATOMIC_UINT64

### Community 91 - "AtomicTransactionalIncrement"
Cohesion: 0.40
Nodes (3): AtomicT, AtomicTransactionalIncrement, m_Atomic

### Community 92 - "VmaVectorInsertSorted"
Cohesion: 0.53
Nodes (6): CmpLess, IterT, KeyT, VmaBinaryFindFirstNotLess(), VmaBinaryFindSorted(), VmaVectorInsertSorted()

### Community 93 - "hipExternalSemaphoreHandleDesc_st"
Cohesion: 0.33
Nodes (6): hipExternalSemaphoreHandleType, hipExternalSemaphoreHandleDesc_st, flags, handle, reserved, type

### Community 94 - "VmaVirtualBlockCreateInfo"
Cohesion: 0.33
Nodes (6): pAllocationCallbacks, VmaVirtualBlockCreateInfo, flags, size, VMA_NULLABLE, VmaVirtualBlockCreateFlags

### Community 95 - "RDNA 2 Instruction Set Architecture Reference Guide"
Cohesion: 0.33
Nodes (6): Microsoft DirectX Reference Website, RDNA 2 Instruction Set Architecture Reference Guide, Intermediate Language (IL) Reference Manual, AMD Accelerated Parallel Processing OpenCL Programming Guide, The OpenCL Specification, OpenGL Programming Guide

### Community 96 - "VmaPnextChainFind"
Cohesion: 0.50
Nodes (4): FindT, MainT, VmaPnextChainFind(), VkStructureType

### Community 97 - "hipPointerAttribute_t"
Cohesion: 0.33
Nodes (6): hipPointerAttribute_t, allocationFlags, devicePointer, hostPointer, isManaged, type

### Community 98 - "rocfft.h"
Cohesion: 0.33
Nodes (5): rocfft_brick_t, rocfft_execution_info_t, rocfft_field_t, rocfft_plan_description_t, rocfft_plan_t

### Community 99 - "VmaAllocationInfo2"
Cohesion: 0.33
Nodes (6): VmaAllocationInfo2, allocationInfo, blockSize, dedicatedMemory, GetAllocationInfo2, vmaGetAllocationInfo2()

### Community 100 - "VmaDefragmentationStats"
Cohesion: 0.33
Nodes (5): VmaDefragmentationStats, allocationsMoved, bytesFreed, bytesMoved, deviceMemoryBlocksFreed

### Community 101 - "VmaVirtualAllocationCreateInfo"
Cohesion: 0.33
Nodes (6): VmaVirtualAllocationCreateInfo, alignment, flags, size, VMA_NULLABLE, VmaVirtualAllocationCreateFlags

### Community 102 - "hipEventRecordNodeParams"
Cohesion: 0.40
Nodes (5): hipEvent_t, hipEventRecordNodeParams, event, hipEventWaitNodeParams, event

### Community 103 - "hipGraphNodeParams"
Cohesion: 0.40
Nodes (5): hipGraphNodeType, hipGraphNodeParams, reserved0, reserved2, type

### Community 104 - "VmaDeviceMemoryCallbacks"
Cohesion: 0.40
Nodes (5): PFN_vmaAllocateDeviceMemoryFunction, PFN_vmaFreeDeviceMemoryFunction, pfnAllocate, VmaDeviceMemoryCallbacks, VMA_NULLABLE

### Community 105 - "__shfl"
Cohesion: 0.40
Nodes (5): shfl_down(), shfl_up(), __shfl(), __shfl_down(), __shfl_up()

### Community 106 - "hipExternalMemoryBufferDesc_st"
Cohesion: 0.40
Nodes (5): hipExternalMemoryBufferDesc_st, flags, offset, reserved, size

### Community 107 - "hipGraphEdgeData"
Cohesion: 0.40
Nodes (5): hipGraphEdgeData, from_port, reserved, to_port, type

### Community 108 - "Appendix C: Compressed Image Formats"
Cohesion: 0.40
Nodes (5): ASTC Compressed Image Formats, Block-Compressed Image Formats, Appendix C: Compressed Image Formats, ETC Compressed Image Formats, PVRTC Compressed Image Formats

### Community 109 - "hipHostNodeParams"
Cohesion: 0.50
Nodes (4): hipHostFn_t, hipHostNodeParams, fn, userData

### Community 110 - "hipLaunchAttribute_st"
Cohesion: 0.50
Nodes (4): hipLaunchAttributeID, hipLaunchAttribute_st, id, pad

### Community 111 - "hipExternalSemaphoreSignalParams_st"
Cohesion: 0.50
Nodes (4): hipExternalSemaphoreSignalParams_st, flags, params, reserved

### Community 112 - "hipExternalSemaphoreWaitParams_st"
Cohesion: 0.50
Nodes (4): hipExternalSemaphoreWaitParams_st, flags, params, reserved

### Community 113 - "Query Commands (vkGet*/vkEnumerate*)"
Cohesion: 0.67
Nodes (4): Array Results Query Pattern, Opaque Binary Data Results, Query Commands (vkGet*/vkEnumerate*), VkResult Return Codes (VK_INCOMPLETE, VK_ERROR_NOT_ENOUGH_SPACE_KHR)

### Community 115 - "VmaMutexLock"
Cohesion: 0.67
Nodes (3): VmaMutexLock, m_pMutex, VMA_MUTEX

### Community 116 - "VmaTotalStatistics"
Cohesion: 0.50
Nodes (4): VmaTotalStatistics, memoryHeap, memoryType, total

### Community 117 - "__float22half2_rn"
Cohesion: 0.67
Nodes (3): float2, __float22half2_rn(), __floats2half2_rn()

### Community 118 - "hipArrayMemoryRequirements"
Cohesion: 0.67
Nodes (3): hipArrayMemoryRequirements, alignment, size

### Community 119 - "VmaAllocationExtraData"
Cohesion: 0.67
Nodes (3): VmaAllocationExtraData, m_Handle, m_pMappedData

## Ambiguous Edges - Review These
- `SOPP Format` → `Branching`  [AMBIGUOUS]
  rdna2_isa.txt · relation: references
- `SOPP Format` → `S_WAITCNT`  [AMBIGUOUS]
  rdna2_isa.txt · relation: references
- `SOPP Format` → `Trap and Exception System`  [AMBIGUOUS]
  rdna2_isa.txt · relation: references
- `SOPP Format` → `Workgroups`  [AMBIGUOUS]
  rdna2_isa.txt · relation: references
- `VOP3P Format` → `VOP3A & VOP3B Instructions`  [AMBIGUOUS]
  rdna2_isa.txt · relation: conceptually_related_to
- `GWS Synchronization Instructions` → `DS_ORDERED_COUNT`  [AMBIGUOUS]
  rdna2_isa.txt · relation: conceptually_related_to
- `VK_AMD_BUFFER_MARKER` → `vkCmdWriteMarkerToMemoryAMD`  [AMBIGUOUS]
  vk_api_structs.txt · relation: references

## Knowledge Gaps
- **722 isolated node(s):** `VMA_NULLABLE`, `VMA_NULLABLE`, `flags`, `VMA_NOT_NULL`, `preferredLargeHeapBlockSize` (+717 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `SOPP Format` and `Branching`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._
- **What is the exact relationship between `SOPP Format` and `S_WAITCNT`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._
- **What is the exact relationship between `SOPP Format` and `Trap and Exception System`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._
- **What is the exact relationship between `SOPP Format` and `Workgroups`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._
- **What is the exact relationship between `VOP3P Format` and `VOP3A & VOP3B Instructions`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `GWS Synchronization Instructions` and `DS_ORDERED_COUNT`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `VK_AMD_BUFFER_MARKER` and `vkCmdWriteMarkerToMemoryAMD`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._