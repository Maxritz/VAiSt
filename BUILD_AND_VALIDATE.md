# VAiSt Full Build and Validation

Target: Vulkan 1.4.357. Public architecture: 12 libraries. Platforms: Windows and Linux. Public ABI: C99. CPU fallback is mandatory. Cooperative matrix is excluded from production dispatch and exists only under `vulkan/DO_NOT_USE/cooperative_matrix/`.

## Build

Linux: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j2 && ctest --test-dir build --output-on-failure`

Windows with Visual Studio: `cmake -S . -B build -G "Visual Studio 17 2022" -A x64 && cmake --build build --config Release && ctest --test-dir build -C Release --output-on-failure`

Windows with MinGW: configure with a working MinGW C/C++ toolchain and Ninja or MinGW Makefiles. The Windows target removes the GNU shared-library `lib` prefix so the C99 core artifact is `vaist_core.dll`.

## Python

Each module contains a complete ctypes loader under its platform Python tree. Point `load(path)` at the corresponding built library when the library is not on the loader path.

## Validation order

Source completeness, compile, API/ABI, unit, negative/error, ownership/lifetime, Windows, Linux, C99, C++, Python, reference/correctness, stress, performance, optimization, regression. The full external Windows matrix is to be run by the user after checkout; this environment can validate the Linux side only.

## Architectural rule

The production graph contains exactly the 12 public libraries. BLAS, FFT, RNG, LAPACK, sparse, reductions, KV, tokenizer, sampling, RoPE, attention, transformer and MoE remain internal responsibilities of those libraries. No cooperative-matrix code is linked into production.
