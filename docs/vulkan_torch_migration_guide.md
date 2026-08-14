# vulkan_torch_migration_guide.md

This guide maps the most commonly used PyTorch/TorchDynamo functions to their equivalent VAiT/Vulkan runtime calls so you can port ML code to Vulkan compute without a compatibility shim layer.

## 1. Tensor creation & movement

### CPU → GPU upload
```python
# PyTorch
import torch
x = torch.randn(1024, 1024, device='cpu')
d = x.cuda()
```

```c
// VAiT — allocate device buffer & upload
#include "vkruntime/vkruntime.h"

VkBuffer buf;
VkDeviceMemory mem;
vkr_malloc(vkr, size_bytes(x), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &buf, &mem);
vkr_upload(vkr, cmd, queue, x.data_ptr(), buf, 0, size_bytes(x));
```

### Zero-copy interop (VAIT_HIP_WMMA=1)
```c
// Export VRAM allocation to HIP
void* hip_ptr = vkExportExternalMemory(vkr, buf, mem);
// HIP kernel reads directly
```

## 2. Math ops

| PyTorch op        | VAiT/VkBLAS equivalent                     |
|------------------|--------------------------------------------|
| `torch.matmul`   | `vkblas_sgemm_f32(...)`                    |
| `torch.relu`     | `vkmath_relu_f32(cmd, ...)`                |
| `torch.gelu`     | `vkmath_gelu_f32(cmd, ...)`                |
| `torch.softmax`  | `vkmath_softmax_f32(cmd, ...)`             |
| `torch.sqrt`     | `vkmath_sqrt_f32(cmd, ...)`                |

### Example: GELU activation
```python
y = torch.nn.functional.gelu(x)
```

```c
// VAiT
vkmath_gelu_f32(cmd, x_buf, y_buf, n);
```

## 3. FFT

```python
# PyTorch
from torch.fft import fft2
Y = fft2(x)
```

```c
// VAiT — radix-2 FFT via VkFFT
vkfft_plan_t plan;
vkfft_create_plan(...);
vkfft_execute_f32(&plan, x_buf, y_buf, ...);
```

## 4. Sampling / RNG

```python
# PyTorch
torch.manual_seed(42)
z = torch.randn(100)
```

```c
// VAiT
vkrand_context_t* ctx;
vkrand_create_context(pdevice, device, queue, &ctx);
vkrand_normal_f32(ctx, cmd, seed, count, out_buf);
```

## 5. Model load/upload

```python
# PyTorch
model = torch.load('model.safetensors')
for k,v in model.items():
    model[k] = v.cuda()
```

```c
// VAiT — GGUF/Safetensors loader
vkmodel_t* m;
vkmodel_load_safetensors(vkr, "model.safetensors", &m);
vkmodel_upload_buffers(m, upload_ctx, cmd, queue);
```

## 6. Command buffer orchestration

PyTorch’s graph execution model maps to a single Vulkan command buffer per pass:

```python
# PyTorch — implicit graph / lazy compilation
result = torch.matmul(a, b) + torch.relu(c)
```

```c
// VAiT — explicit single cmd buffer
vkBeginCommandBuffer(cmd, ...);

vkblas_sgemm_f32(vkblas_ctx, cmd, ...);
vkCmdPipelineBarrier(cmd, ..., VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, 1, &buf_bar);

vkmath_relu_f32(vkmath_ctx, cmd, c_buf, out_buf, n);
vkCmdPipelineBarrier(cmd, ...);

vkEndCommandBuffer(cmd);
vkQueueSubmit(queue, 1, &submit, fence);
vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
```

## 7. Key differences from PyTorch

| Feature              | PyTorch               | VAiT/Vulkan                                 |
|---------------------|-----------------------|---------------------------------------------|
| Tensors             | Reference counted     | Manual `vkr_malloc` / `vkr_free`            |
| Device sync         | Implicit              | Explicit `vkQueueWaitIdle` / fences         |
| Streams             | `torch.cuda.Stream()` | `vkstream_create()` (see vkstream.h)         |
| Backward pass       | Autograd              | Manual or external (JAX autodiff bridge)    |
| Kernel dispatch     | Lazy                  | Explicit `vkCmdDispatch`                    |

## 8. Stubbed / unsupported mappings

We deliberately skip shim implementations for these. Replace with direct VAiT calls:

| Unsupported | Replace with |
|-------------|-------------|
| `torch.jit.script()`          | Pre-compile with `compile_shaders.ps1`           |
| `torch.autograd.backward()`   | Manual gradients via external diff frameworks     |
| `torch.nn.Module.__call__()`  | Explicit `vkCmdDispatch` chains                 |
| `torch.cuda.ipc_collect()`    | Not needed for single-GPU setups                 |
