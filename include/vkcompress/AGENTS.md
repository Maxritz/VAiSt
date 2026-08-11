# VKCompress Include — Local Contract

Child of root `AGENTS.md`. Governs the public API surface under `include/vkcompress/`.

## API Design Principles

### Vulkan-native
- `VkCompressContext` wraps `VkDevice`, pipeline layout, descriptor set layout
- All buffer arguments are `VkBuffer`/`vkcomp_buffer_id_t` handles
- No heap allocation in hot paths; staged buffers allocated in context
- Caller supplies `VkCommandBuffer`

### Agnostic tagging
- `vkcomp_register_buffer(size, tag, level)` — `tag` is an opaque string
  (e.g. "model:abc123:tensor:0", "cache:kv:layer_5")
- Engine owns tag semantics; vkcompress manages compression + mapping
- Compression level 0-9: 0 = fastest, 9 = highest ratio

### Files
| File | Purpose |
|------|---------|
| `vkcompress.h` | Public API: context, register_buffer, write/read, catalog I/O |
