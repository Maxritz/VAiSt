# vaist_compute

## PURPOSE
Numerical kernels, reductions, matmul, capability-driven dispatch

## MANDATORY USE
Canonical public boundary for this responsibility. Higher layers use this boundary instead of backend internals.

## PUBLIC API
See `c99/include/vaist_compute.h`. C99 is authoritative; C++ and Python preserve its semantics.

## INTERNAL COMPONENTS
C99 implementation, C++ convenience layer, Python ctypes binding.

## DEPENDENCIES
`vaist_core` for the common ABI/status contract; platform C runtime. Backend-specific dependencies are isolated in runtime.

## OWNERSHIP
Create functions transfer object ownership to the caller; matching destroy functions release owned resources. Borrowed pointers remain parent-owned.

## LIFETIME
Parent destruction invalidates borrowed children. Invalid state is reported.

## THREADING
Independent objects are reentrant. No implicit worker threads are created by these baseline APIs.

## SYNCHRONIZATION
No hidden synchronization. Runtime synchronization is explicit.

## CAPABILITIES
Unsupported capabilities return `VAIST_UNSUPPORTED`.

## DISPATCH
Capability-driven. Unsupported or semantically invalid candidates are rejected. Cooperative matrix is excluded from production dispatch.

## FALLBACKS
CPU/reference execution is the correctness fallback.

## ERRORS
Common `VaistStatus` semantics are used and errors are explicit.

## TESTS
Root build/test drivers cover compilation, ABI, negative paths and cross-language loading. Full integration validation is intended for the user's full compile pass.

## PERFORMANCE
Contiguous hot loops and no per-element dynamic dispatch. Architecture-specific tuning follows profiling and reference comparison.

## PLATFORM DIFFERENCES
Windows uses DLL conventions; Linux uses ELF visibility. Public semantics remain identical.
