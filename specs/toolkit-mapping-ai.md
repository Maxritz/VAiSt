# AI/ML Toolkit -> Vulkan SDK Component Mapping

> If you already know **CUDA / ROCm / OpenVINO**, this maps those components to
> this Vulkan SDK so you can use it by analogy. Grounded in: NVIDIA CUDA docs
> (docs.nvidia.com/cuda), AMD ROCm docs (rocmdocs.amd.com), OpenVINO (Wikipedia
> + docs.openvino.ai). Verified against this repo's actual public headers and
> `CMakeLists.txt`.

## Verified module inventory (this SDK)

| Module     | Library | Entry point(s) | Role |
|------------|---------|----------------|------|
| VKBLAS     | vkblas  | `vkblas_sgemm`/`dgemm`/`hgemm`/`bgemm` + `*_strided_batched`/`*_batched`, `vkblas_gemm_ex*` , `vkblas_qgemm_<fmt>_f32` | BLAS Level-3 (GEMM), tiled shared-mem + on-the-fly dequant for q4_0/q4k/q5k/q6k/q3k/q8_0/iq4xs |
| VKBLAS-L1L2| vkblas_l1l2 | `vkblas_l1_axpy/scal/dot/nrm2/asum/amax`, `vkblas_l2_gemv` | BLAS Level-1 / Level-2 (vectors, matrix-vector) |
| VKFFT      | vkfft   | `vkfft_create_plan*`, `vkfft_execute{_inverse}_{f32,f16}`, `vkfft_execute_2d*` | Radix-2 FFT forward+inverse (f16/f32, 2D) |
| VKRAND     | vkrand  | `vkrand_uniform_f32`, `vkrand_threefry_uniform_f32`, `vkrand_normal_f32`, `vkrand_uniform_uint32` | PRNG (Threefry) + uniform/normal sampling |
| VKQuant    | vkquant | `vkquant_dequant_<fmt>_f32` (23 fmt), `vkquant_quantize_<fmt>_f32` (22 fmt) | Block-quant dequant + forward (encode) quantize; q4_0..tq2_0, iq1s..iq4xs |
| VKMath     | vkmath  | `vkmath_relu/silu/gelu/tanh/sigmoid{_f16}`, `vkmath_add/mul/add_mul/scale{_f16}`, `vkmath_{max|sum}_reduce_dim_f32`, `vkmath_softmax/rms_norm/layernorm_f32`, `vkmath_argmax/argmin_f32`, `vkmath_cumsum_f32`, `vkmath_clip/abs/sign/exp/log/sqrt/rsqrt/pow_f32`, `vkmath_cast_f32_to_bf16`/`vkmath_cast_bf16_to_f32` | Elementwise unary/binary, reductions, norms, activations, bf16 casts |
| VKKV       | vkkv    | `vkkv_create_transfer`, `vkkv_fit_cpu`, `vkkv_apply` | LLM KV-cache: ridge-fit new keys into cache basis (host double Gauss-Jordan), GPU apply |
| VKDIST     | vkdist  | `vkdist_server_run`, `vkdist_server_serve_many`, `vkdist_register_buffer`, `vkdist_upload`, `vkdist_sgemm*`, `vkdist_readback`, `vkdist_close` | Distributed inference transport (TCP, u32-LE length+opcode framing, <=1 MiB frames) |
| VKModel    | vkmodel | `vkmodel_load_gguf`, `vkmodel_load_safetensors` -> `VkModelTensor[]` | Model/weight loading (GGUF + safetensors) |
| VKRuntime  | vkruntime | (caps detection, push descriptors) | Vulkan device/pipeline/descriptor/memory/queue layer |

## Capability mapping

### Math / compute primitives (the porting heart)
| Capability | CUDA | ROCm | OpenVINO | This SDK | Status |
|---|---|---|---|---|---|
| BLAS L3 (GEMM/MatMul, batched, strided, mixed) | cuBLAS | rocBLAS | (oneAPI MKL inside) | VKBLAS + L1L2 | exists; qgemm & f16/bf16/f64 GEMM baseline-only (no subgroup/coopmatrix tier) |
| BLAS L1/L2 (axpy, scal, dot, nrm2, asum, amax, gemv) | cuBLAS | rocBLAS | — | VKBLAS-L1L2 | exists |
| FFT | cuFFT | rocFFT | OV runtime FFT ops | VKFFT | exists (radix-2 f16/f32, 2D) |
| RNG / distribution sampling | cuRAND | rocRAND / rocPRIM | (OV runtime RNG) | VKRAND | exists (Threefry + uniform/normal) |
| Elementwise / activations / reductions / softmax | (kernels/CUB) | rocPRIM + MIOpen | ngraph ops | VKMath | exists; **bf16 elementwise compute ops integrated** (`add/mul/add_mul/scale`, uint16_t bf16 I/O, f32 compute) — no bf16 activations/reductions; f16 add/mul/scale public APIs lack blobs → `VK_ERROR_FEATURE_NOT_PRESENT` |
| Weight quantization (block formats) | cuTensor/cuQuant (via) | MIOpen | NNCF | VKQuant | exists (dequant 23 + forward-quant 22 formats) |
| Sparse BLAS | cuSPARSE | rocSPARSE | — | (none) | gap |
| Signal/image processing | NPP | rocAL | — | (none) | gap |

### Compiler / JIT
| CUDA nvcc/NVRTC (runtime compile) | ROCm hipcc/hipRTC | OpenVINO Model Optimizer | This SDK |
|---|---|---|---|
| JIT compile of kernels | hipcc compile; hipRTC JIT | convert model -> IR (.xml/.bin) | `compile_shaders.ps1` -> `shaders_spv.h` embedded SPIR-V |
Status: **offline static compile only**, no runtime JIT (NVRTC/hipRTC-style). Shader blobs are compile-time embedded arrays; "add a kernel" = drop a `.comp` + regenerate.

### Distributed / transport
| NCCL (GPU-aware collectives) | rccl | — | VKDIST |
VKKDIST: u32-LE length+opcode framing, OS-agnostic sockets, <=1 MiB frames, `vkdist_sgemm`/`sgemm_partitioned` partitioning. Note: a transport+framing layer, **not** full NCCL/rccl collective semantics (no GPUDirect RDMA, no ring/tree allreduce).

### Model weights (what a `.bin`/`safetensors`/IR maps to)
| Framework format | Toolkit | This SDK |
|---|---|---|
| HuggingFace `.safetensors` | (HF) | `vkmodel_load_safetensors` -> `VkModelTensor[]` |
| GGUF / ggml | llama.cpp | `vkmodel_load_gguf` -> `VkModelTensor[]` |
| Torch `.bin` (pickle) | PyTorch | NOT supported (unsafe pickle) -> use safetensors |
| **OpenVINO IR** `.xml`+`.bin` | OpenVINO | NOT supported yet (gap) |

### LLM domain
| Concept | Toolkit equiv | This SDK |
|---|---|---|
| KV-cache adaptation for new context | (llama.cpp fit/compress) | VKKV (`vkkv_fit_cpu` + `vkkv_apply`) |
| Single command buffer per pass (avoid per-op submit) | (NVTX/annotation grouping) | project ideology — VKMath/VKBLAS record into caller's `VkCommandBuffer` |
| Push descriptors everywhere | (driver descriptors) | VK_KHR_push_descriptor in runtime + math |

### Profiling / perf / arch selection
| CUDA Nsight Compute/Systems | ROCm rocprof/rocgdb | OpenVINO Benchmark | This SDK |
|---|---|---|---|
| Profiler | rocprof-sys, rocgdb | `benchmark_app` | VKRuntime `vkr_detect_capability` -> `active_tier` (baseline/subgroup/coopmatrix) + Vulkan timestamps + `test_vk*` harnesses |

## Function-name quick-map (verified headers)
- `cublas_gemm` / `hipblasGemm` -> `vkblas_sgemm` (f32) / `vkblas_dgemm` (f64) / `vkblas_hgemm` (f16) / `vkblas_bgemm` (bf16) / `vkblas_gemm_ex` (mixed)
- `cublasGemmStridedBatched` / `hipblasGemmStridedBatched` -> `vkblas_*_strided_batched`
- `cublasGemmBatched` -> `vkblas_*_batched`
- quantized Gemm (ggml-style) -> `vkblas_qgemm_<q4_0|q4k|q5k|q6k|q3k|q8_0|iq4xs>_f32`
- `cublasSaxpy/dot/nrm2/asum/amax` -> `vkblas_l1_axpy/dot/nrm2/asum/amax`
- `cublasSgemv` -> `vkblas_l2_gemv`
- `cufftPlanMany` -> `vkfft_create_plan`; `cufftExecC2C` -> `vkfft_execute_f32` (+ inverse / 2d / f16)
- `curandGenerateUniform/Normal` -> `vkrand_uniform_f32 / vkrand_normal_f32` (default); `vkrand_threefry_uniform_f32` for reproducible streams
- `nnorm/activation` kernels -> `vkmath_relu/silu/gelu/tanh/sigmoid{_f16}`, `vkmath_{max|sum}_reduce_dim_f32`, `vkmath_softmax/rms_norm/layernorm_f32`, `vkmath_cumsum_f32`, `vkmath_abs/sqrt/exp/log/rsqrt/pow/scale/add/mul/add_mul/clip/sign_f32`
- `ggml_quantize_*` / `dequant` -> `vkquant_quantize_<fmt>_f32` / `vkquant_dequant_<fmt>_f32`
- `model.layers.*.weight` load -> `vkmodel_load_gguf` / `vkmodel_load_safetensors` -> `VkModelTensor[]`

## Verified gaps (priority order)
1. **qgemm + f16/bf16/f64 GEMM: baseline shared-mem tiled only** — no subgroup/coopmatrix dequant+MAC tier. qgemm is the decode hot path (>80 tok/s goal); RX 9070 XT (coopmatrix) can host this tier. **Highest perf priority.**
2. **qgemm outputs f32 only** — no fp16 output storage path (all writes f32).
3. **VKMath: bf16 elementwise ops integrated** — `vkmath_add_bf16`/`vkmath_mul_bf16`/`vkmath_add_mul_bf16`/`vkmath_scale_bf16` (`baseline/{add,mul,add_mul,scale}_bf16.comp`, uint16_t bf16 I/O, f32 compute, truncate) PASS on RX 9070 XT; bf16 casts already integrated. Remaining VKMath: bf16 activations/reductions; f16 add/mul/scale public APIs lack `(KERNEL, DTYPE_F16)` table entries → `VK_ERROR_FEATURE_NOT_PRESENT`.
4. No cuSPARSE-equivalent (sparse BLAS) / no NPP-equivalent (signal/image).
5. No runtime JIT (NVRTC/hipRTC) — offline shader compile only.

## Next action queue
1. (STARTING) **qgemm subgroup/coopmatrix dequant+MAC tier** (RX 9070 XT), then f16/bf16/f64 GEMM tier — the >80 tok/s path.
2. (optional) OpenVINO IR loader; sparse BLAS; runtime JIT — lower priority.
