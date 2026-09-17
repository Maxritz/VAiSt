# MODULE 03 TRACE — vaist_tensor

## Purpose
Universal tensor descriptors, storage, views, reshape/transpose/copy

## Mandatory use
This module is the sole public boundary for its responsibility in the 12-library architecture.

## Classification
NEW: dtype, shape, stride, contiguous storage, views, reshape, transpose and copy.

## Windows
C99, C++17 and Python trees are complete and use the same public semantics. Windows CMake emits the required DLL names.

## Linux
C99, C++17 and Python trees are complete and use ELF shared-library conventions.

## C99 ABI
The C99 header is authoritative. Handles are opaque where lifetime-bearing. Status values use `vaist_core`.

## C++
The C++ layer provides convenience/RAII wrappers without crossing the C99 boundary with exceptions or STL objects.

## Python
The Python layer loads the corresponding C ABI through `ctypes`; it does not implement a competing execution engine.

## Dispatch
Capability-driven. Missing acceleration falls back to a valid CPU/reference implementation. Cooperative matrix is absent from production.

## Ownership/lifetime
Every create has a matching destroy where resources are allocated. Borrowed pointers are parent-owned.

## Validation performed in this workspace
- C99 compile with GCC 14.2, `-Wall -Wextra -Werror`: PASS
- C++17 compile with GCC 14.2, `-Wall -Wextra -Werror`: PASS
- CTest stack smoke: 2/2 PASS
- Python bytecode compilation: PASS
- Python core ctypes smoke: PASS
- AddressSanitizer + UndefinedBehaviorSanitizer stack smoke: PASS
- Required directory matrix: PASS
- Forbidden placeholder token scan: PASS

## External validation still required
Native Windows/MSVC or MinGW execution, real Vulkan-device execution, full model-family/quantization parity, and workload performance measurements must be performed on the target environments. This source package does not falsely mark those environment-specific results as locally proven.

## Optimization status
The baseline keeps hot loops simple and uses explicit fallback semantics. Hardware-specific SIMD/Vulkan kernel tuning must be measured against a reference before being accepted.

## Lock status
SOURCE-COMPLETE BASELINE: PASS
LOCAL LINUX BUILD: PASS
LOCAL SANITIZER SMOKE: PASS
WINDOWS NATIVE GATE: PENDING EXTERNAL BUILD
VULKAN HARDWARE GATE: PENDING TARGET DEVICE
