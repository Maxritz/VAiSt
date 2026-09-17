# VAiSt

Full multi-platform C99/C++/Python source baseline following the locked VAiSt architecture.

- Vulkan target: 1.4.357
- Windows + Linux
- C99 public ABI
- C++17 convenience/RAII layer
- Python ctypes layer
- CPU fallback
- 12 production public libraries
- cooperative matrix isolated under `vulkan/DO_NOT_USE/cooperative_matrix/`

The code is intentionally buildable without requiring the Vulkan SDK for the baseline CPU path. The runtime detects the platform Vulkan loader dynamically and records API availability; Vulkan kernel implementations belong behind the runtime/compute boundaries and must be capability-selected.

The supplied source is a complete baseline implementation for the exposed APIs. The user's full Windows/Vulkan hardware build is the next integration validation step.
