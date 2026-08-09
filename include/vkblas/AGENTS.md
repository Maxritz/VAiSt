# VKBLAS Include — Local Contract

Child of root `AGENTS.md`. Governs the public API surface under `include/vkblas/`.

## API Design Principles

### Vulkan-native handles everywhere
- `VkBLASContext` wraps a `VkDevice`, `VkPipelineCache`, and a descriptor-set allocator
- All buffer arguments are `VkBuffer` handles (no CUDA/hip pointers)
- Users manage memory via `VkAllocationCallbacks` or VMA; VKBLAS never allocates device memory
- Command recording happens into a user-supplied `VkCommandBuffer` (never a vkQueue)

### rocBLAS mirror naming
- Prefix `vkblas_` (analogous to `rocblas_`, `hipblas_`)
- Type-precision suffixes: `s`=f32, `d`=f64, `h`=f16, `bf`=bf16, `c`=complex-f32, `z`=complex-f64, `i8`=int8
- `gemm`, `gemm_strided_batched`, `gemm_batched`, `gemm_ex` — exact parameter order mirrors `hipblasSgemm`
- `VkBLASOperation_t` mirrors `hipblasOperation_t` (NONE=0, T=1, C=2)
- Fused quantized GEMM: `vkblas_qgemm_q8_0_f32`, `vkblas_qgemm_q4k_f32`,
  `vkblas_qgemm_q4_0_f32`, `vkblas_qgemm_q5k_f32`, `vkblas_qgemm_q6k_f32`,
  `vkblas_qgemm_q3k_f32`, `vkblas_qgemm_iq4xs_f32`
  (`y = alpha*(dequant(W)*x) + beta*y`, in place). The weight layout contract
  (Q8_0: 36 B/block of 32 elems; Q4_0: 20 B/block of 32 elems; Q4_K: ggml
  144 B/block of 256 elems; Q5_K/Q6_K/Q3_K/IQ4_XS: ggml 176/210/110/136
  B/block of 256 elems; row r at byte offset `r*ldw`) is documented on each
  function in `vkblas.h`.

### No heap allocation in hot paths
- All dispatch functions take pointers, never allocate
- Pipeline objects are created lazily and cached inside `VkBLASContext`
- Descriptor sets are allocated from a pool owned by the context

## Files

| File | Purpose |
|------|---------|
| `vkblas.h` | Public API: context mgmt, GEMM family, types, constants |
| `vkblas_matmul.h` | [future] MatMul fusion layer (blockwise attention) |

## Thread Safety

- `VkBLASContext` is **not thread-safe**. Caller serializes concurrent `vkblas_*()` calls that touch the same context.
- Recording into different `VkCommandBuffer`s from different threads is fine, but each must use a distinct context or be serialized.
