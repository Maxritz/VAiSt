# VAiSt Component Issue Closure — Pass 2

Assessment target: `VAiSt_full_source_continued.zip` baseline plus remediation changes.
Assessment date: 27 August 2026.

## Scope

This pass addresses the independent component-library buildability issues identified in `VAiSt_Component_Build_and_Issue_List.md`.

## Closed issues

### ISSUE-001 — C++ individual library targets

Closed. The root CMake file now declares twelve independent static C++ targets:

`vaist_cpp_core`, `vaist_cpp_runtime`, `vaist_cpp_tensor`, `vaist_cpp_compute`, `vaist_cpp_quant`, `vaist_cpp_graph`, `vaist_cpp_model`, `vaist_cpp_nn`, `vaist_cpp_llm`, `vaist_cpp_engine`, `vaist_cpp_ai`, `vaist_cpp_distributed`.

`vaist_cpp_wrappers` remains only as an optional INTERFACE aggregate convenience target.

Each C++ target links its corresponding C99 component and has its own include contract.

### ISSUE-002 — Python loaders/package model

Closed at the package-module level. Each component now has a Python package directory containing `__init__.py` and `vaist_<component>.py`. CMake installs all twelve Python component directories. The test suite imports every component package and loads every corresponding native library.

The Python surface remains a `ctypes` package rather than a CPython extension ABI. This is intentional and documented.

### ISSUE-003 — Windows Python DLL naming

Closed in the build contract. All Windows C99 shared targets receive `PREFIX ""`, producing `vaist_<component>.dll`. Python `library_name()` returns the same Windows names. Linux continues to use `libvaist_<component>.so`.

### ISSUE-006 — Build directory inconsistency

Closed. Helper scripts now use `build/linux` and `build/windows`. CMake-generated tests use `${CMAKE_BINARY_DIR}` rather than hardcoded directory names.

### ISSUE-007 — C++ coverage only four components

Closed. Twelve individual C++ component smoke executables are registered with CTest. Each test includes and links its own component target. The aggregate C++ smoke remains as an additional integration test.

### ISSUE-008 — Python not registered with CTest

Closed. Python component/package loading is registered as a CTest test when Python 3 is available. All twelve packages and native libraries are exercised.

### ISSUE-009 — Native dependency packaging

Closed for the defined install model. CMake installs all C99 component libraries, all C++ libraries, headers, CMake export metadata and Python packages. Linux shared libraries receive `$ORIGIN` install RPATH so installed component libraries locate installed `vaist_core`. Windows uses co-located runtime DLL output/install behavior. A clean-staging load test validates the installed native library set on Linux.

### ISSUE-010 — Ambiguous 12-library terminology

Closed. `ARTIFACT_MATRIX.md` explicitly defines the delivery matrix as 12 components × 2 platforms × 3 language surfaces = 72 component/surface/platform deliverables, excluding tests and the isolated cooperative-matrix reference tree.

### ISSUE-012 — Per-component ABI/symbol verification

Closed for Linux C99. CTest now runs a header-derived exported-symbol check over all twelve ELF libraries. Native Windows PE symbol verification remains an external Windows gate.

### ISSUE-013 — Clean-staging load test

Closed for Linux. CTest performs a clean install into a fresh staging directory and loads all twelve installed C99 libraries. A separate manual staged Python test also loaded all twelve Python packages against the staged libraries.

## Existing issue retained

### ISSUE-004 — Windows MinGW TLS

Already fixed in the baseline and retained. The Windows GNU branch uses a MinGW-compatible TLS declaration.

### ISSUE-005 / ISSUE-011 — native Windows execution

Open as an environment-dependent validation gate. Cross-compilation is not substituted for native Windows execution. Native Windows CTest must be run on Windows.

## Current Linux validation

- C99 libraries: 12/12 build PASS
- C++ component libraries: 12/12 build PASS
- Individual C++ smoke tests: 12/12 PASS
- Aggregate C99 stack smoke: PASS
- Aggregate C++ smoke: PASS
- Python package/native load: 12/12 PASS
- Clean staging native load: 12/12 PASS
- Linux C99 exported-symbol verification: PASS
- Total CTest tests: 17/17 PASS
- Release configuration: PASS
- Strict C/C++ warning flags: PASS

## Remaining external gates

1. Native Windows/MSVC configure/build/CTest.
2. Native Windows MinGW runtime/loader test.
3. Real Vulkan 1.4.357 device initialization and feature/capability validation.
4. Real workload/model tests.
5. Performance benchmarking and dispatch tuning on target hardware.

These are not claimed as passed by this Linux validation environment.
