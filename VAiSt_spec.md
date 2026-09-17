# VAiSt Master Engineering Specification
## Locked Architecture, Module Contracts, Dispatch Rules, Validation Gates, and Implementation Plan

**Status:** LOCKED FOR IMPLEMENTATION  
**Target Vulkan:** Vulkan 1.4.357  
**Platforms:** Windows + Linux  
**Languages:** C99 + C++ + Python  
**Public ABI:** C99  
**Primary backend:** Vulkan  
**Required fallback:** CPU  
**Production cooperative-matrix path:** NONE  
**Final cooperative-matrix module:** `DO_NOT_USE` only

---

# 1. Authority and Compliance

This document consolidates the architectural decisions and engineering contracts established for the VAiSt implementation.

It must be used together with:

- `rules.md`
- `VAiSt_12_DLL_Consolidation_Reuse_Plan-1.md`

`rules.md` is absolute for generated source. No placeholders, ellipses, omitted functions, incomplete files, pseudo-code, or abbreviated implementation blocks are permitted.

The consolidation plan establishes reuse of the existing VAiSt low-level implementation, consolidation into 12 public libraries, Windows + Linux support, C99 as the public ABI, Vulkan as the primary backend, and CPU as the required fallback.

This specification adds the missing engineering contracts required to make that architecture implementable without uncontrolled ABI, lifecycle, dispatch, synchronization, or platform drift.

---

# 2. Absolute Global Rules

These rules apply to every module and every public or internal function.

## 2.1 Complete implementation

Every generated source file must be complete.

Forbidden:

- TODO
- FIXME placeholders
- `...`
- omitted functions
- "implementation omitted"
- pseudo-code
- fake success returns
- incomplete imports
- incomplete build files
- intentionally unwired APIs

If a module is too large for one output unit, split it into complete files/modules. Never truncate a file.

## 2.2 Module gate

A module must pass its complete validation gate before the next module begins.

```text
SPECIFY
  ↓
DATA MODEL
  ↓
API CONTRACT
  ↓
IMPLEMENT
  ↓
TRACE
  ↓
BUILD
  ↓
UNIT TEST
  ↓
NEGATIVE TEST
  ↓
ABI TEST
  ↓
WINDOWS TEST
  ↓
LINUX TEST
  ↓
C99 TEST
  ↓
C++ TEST
  ↓
PYTHON TEST
  ↓
REFERENCE/CORRECTNESS TEST
  ↓
RESOURCE/LIFETIME TEST
  ↓
PERFORMANCE TEST
  ↓
OPTIMIZE
  ↓
RETEST
  ↓
MODULE LOCK
```

No module proceeds after a failed gate.

## 2.3 Fastest valid implementation

Every operation may have multiple implementations.

The system selects the fastest implementation that is:

1. supported;
2. semantically correct;
3. numerically valid within its defined tolerance;
4. resource-safe;
5. ABI-compatible;
6. compatible with dtype, layout, alignment and shape;
7. valid for the current workload and runtime state.

Correctness always outranks performance.

## 2.4 No mandatory optimization

No hardware-specific optimization is required for correctness.

A missing optimization must fall back to another valid implementation.

## 2.5 Cooperative matrix

Cooperative matrix is completely excluded from production execution.

It is not:

- a required backend;
- a fallback;
- a dispatch candidate;
- an autotuning candidate;
- a correctness dependency;
- a performance dependency.

A complete reference/experimental implementation may exist only at the end of the implementation sequence under:

```text
vulkan/DO_NOT_USE/cooperative_matrix/
```

It must never be linked into or discovered by the production dispatch registry.

## 2.6 No hidden copies

No operation may silently introduce CPU↔GPU or other expensive data movement unless that movement is explicitly part of its contract and visible to the planner/runtime.

## 2.7 No hidden synchronization

An asynchronous operation must not secretly wait for completion unless its documented contract explicitly defines it as synchronous.

## 2.8 Capability-driven execution

API version, device feature, extension, subgroup property, shader capability, and instruction capability are separate concepts.

Never infer availability of an optimization from the Vulkan target version alone.

## 2.9 Reuse before rewrite

Existing working VAiSt implementations are reused, moved behind the correct public boundary, extended, rewired, or replaced only when necessary.

The architectural transformation is:

```text
KEEP
MERGE
EXTEND
REWIRE
REPLACE
NEW
```

not a rewrite from zero.

---

# 3. Public Architecture

Exactly 12 production public libraries:

```text
01 vaist_core
02 vaist_runtime
03 vaist_tensor
04 vaist_compute
05 vaist_quant
06 vaist_graph
07 vaist_model
08 vaist_nn
09 vaist_llm
10 vaist_engine
11 vaist_ai
12 vaist_distributed
```

BLAS, FFT, RNG, LAPACK, Sparse, Reductions, KV, Tokenizer, Sampling, RoPE, Attention, Transformer, MoE, GGUF, SafeTensors, ONNX, OpenVINO, Memory, Streams, Events, Schedulers and similar components remain internal modules of the 12 boundaries and do not become additional public DLLs.

---

# 4. Required Source Layout

The requested implementation tree is:

```text
vulkan/
├── windows/
│   ├── c99/
│   │   ├── core/
│   │   ├── runtime/
│   │   ├── tensor/
│   │   ├── compute/
│   │   ├── quant/
│   │   ├── graph/
│   │   ├── model/
│   │   ├── nn/
│   │   ├── llm/
│   │   ├── engine/
│   │   ├── ai/
│   │   └── distributed/
│   ├── c++/
│   │   ├── core/
│   │   ├── runtime/
│   │   ├── tensor/
│   │   ├── compute/
│   │   ├── quant/
│   │   ├── graph/
│   │   ├── model/
│   │   ├── nn/
│   │   ├── llm/
│   │   ├── engine/
│   │   ├── ai/
│   │   └── distributed/
│   └── python/
│       ├── core/
│       ├── runtime/
│       ├── tensor/
│       ├── compute/
│       ├── quant/
│       ├── graph/
│       ├── model/
│       ├── nn/
│       ├── llm/
│       ├── engine/
│       ├── ai/
│       └── distributed/
└── linux/
    ├── c99/
    │   ├── core/
    │   ├── runtime/
    │   ├── tensor/
    │   ├── compute/
    │   ├── quant/
    │   ├── graph/
    │   ├── model/
    │   ├── nn/
    │   ├── llm/
    │   ├── engine/
    │   ├── ai/
    │   └── distributed/
    ├── c++/
    │   ├── core/
    │   ├── runtime/
    │   ├── tensor/
    │   ├── compute/
    │   ├── quant/
    │   ├── graph/
    │   ├── model/
    │   ├── nn/
    │   ├── llm/
    │   ├── engine/
    │   ├── ai/
    │   └── distributed/
    └── python/
        ├── core/
        ├── runtime/
        ├── tensor/
        ├── compute/
        ├── quant/
        ├── graph/
        ├── model/
        ├── nn/
        ├── llm/
        ├── engine/
        ├── ai/
        └── distributed/
```

The final isolated module is:

```text
vulkan/
└── DO_NOT_USE/
    └── cooperative_matrix/
        ├── windows/
        │   ├── c99/
        │   ├── c++/
        │   └── python/
        └── linux/
            ├── c99/
            ├── c++/
            └── python/
```

The `DO_NOT_USE` tree is outside the production dependency graph.

---

# 5. Language Contract

## 5.1 C99

C99 is the canonical public ABI.

All public types, status values, handles, function signatures, ownership semantics and versioning originate in C99.

## 5.2 C++

C++ provides:

- RAII;
- type-safe wrappers;
- convenience classes;
- resource management;
- C++-appropriate error translation.

C++ must preserve the C99 semantics and ABI.

C++ exceptions must never cross the C99 boundary.

## 5.3 Python

Python provides:

- native bindings;
- Pythonic object lifetime;
- buffer interoperability;
- Python exception translation;
- access to the same runtime/backend/dispatch semantics.

Python must not create a competing execution model.

---

# 6. ABI Contract

Public ABI must be versioned.

Required concepts:

```text
VAIST_ABI_MAJOR
VAIST_ABI_MINOR
VAIST_ABI_PATCH
```

Required discovery functions:

```text
vaist_get_version()
vaist_get_abi_version()
vaist_get_build_info()
```

Public structures must use explicit size/version handling where appropriate.

Rules:

- no ABI-dependent C++ objects in C99 structures;
- no STL types in C99 ABI;
- no exceptions across C99;
- fixed-width integer types;
- explicit ownership;
- explicit alignment requirements;
- explicit lifetime;
- reserved fields where forward compatibility requires them.

---

# 7. Global Status/Error Contract

A common `VaistStatus` model is mandatory.

Required semantic categories include:

```text
VAIST_OK
VAIST_INVALID_ARGUMENT
VAIST_INVALID_STATE
VAIST_UNSUPPORTED
VAIST_OUT_OF_MEMORY
VAIST_DEVICE_ERROR
VAIST_DEVICE_LOST
VAIST_TIMEOUT
VAIST_CANCELLED
VAIST_IO_ERROR
VAIST_MODEL_ERROR
VAIST_COMPILE_ERROR
VAIST_RUNTIME_ERROR
VAIST_INTERNAL_ERROR
```

Exact numeric assignments must be frozen in `vaist_core`.

Errors must preserve enough context to diagnose:

- operation;
- module;
- backend;
- resource;
- platform;
- underlying Vulkan/native error where applicable.

---

# 8. Ownership and Lifetime Contract

Every resource must have explicit ownership.

Canonical lifecycle:

```text
CREATE
  ↓
OWN / BORROW
  ↓
RETAIN
  ↓
USE
  ↓
RELEASE
  ↓
DESTROY
```

Every handle must define:

- creator;
- owner;
- borrower;
- retain behavior;
- release behavior;
- destruction behavior;
- thread-safety;
- valid states;
- parent dependencies.

A module may not destroy a resource owned by another module without an explicit ownership transfer.

---

# 9. Threading Contract

Every public API must be classified as:

```text
THREAD_SAFE
THREAD_COMPATIBLE_WITH_EXTERNAL_SYNC
SINGLE_THREAD_ONLY
```

This classification must be documented and tested.

The contract applies to:

- runtime;
- devices;
- streams;
- events;
- tensors;
- graphs;
- models;
- KV;
- engine;
- Python objects;
- distributed resources.

---

# 10. Synchronization Contract

The runtime must explicitly distinguish:

```text
submitted
running
completed
failed
cancelled
```

Required semantics:

- synchronous execution;
- asynchronous execution;
- event dependency;
- wait;
- poll;
- timeout;
- cancellation;
- error propagation.

No hidden wait is permitted.

---

# 11. Capability Model

Capability discovery is centralized.

The capability model must distinguish:

```text
API_VERSION
DEVICE_FEATURE
DEVICE_EXTENSION
SUBGROUP_CAPABILITY
SHADER_CAPABILITY
INSTRUCTION_CAPABILITY
DATATYPE_SUPPORT
LAYOUT_SUPPORT
KERNEL_SUPPORT
```

Higher layers consume the authoritative capability model.

They must not independently query Vulkan features and develop divergent decisions.

---

# 12. Dispatch Architecture

Dispatch is shared infrastructure.

Every implementation registers metadata sufficient to answer:

```text
What operation?
What backend?
What dtype?
What layout?
What shape?
What alignment?
What capabilities?
What workspace?
What numerical contract?
What performance profile?
What fallback?
```

Selection:

```text
request
  ↓
capability filter
  ↓
semantic validity
  ↓
numerical validity
  ↓
resource feasibility
  ↓
workload suitability
  ↓
cached performance / cost model
  ↓
fastest valid implementation
```

Runtime failures may trigger recovery/fallback only when the failure is safely recoverable.

Examples:

```text
unsupported → fallback
workspace OOM → alternate workspace/kernel or fallback
recoverable allocation pressure → retry/replan
invalid argument → return error
fatal device loss → device-loss policy
```

Do not retry invalid input indefinitely.

---

# 13. Autotuning Contract

Autotuning must not run on every invocation.

Use:

```text
default/cost model
  ↓
cached result
  ↓
controlled autotuning
  ↓
cache result
  ↓
reuse
```

Tuning identity must include all relevant factors, such as:

- device identity;
- driver identity;
- Vulkan/runtime capability identity;
- kernel implementation version;
- operation;
- dtype;
- shape;
- layout;
- alignment;
- relevant workload characteristics.

Invalidate cached results when the relevant environment or implementation changes.

---

# 14. Numerical Correctness Contract

Every optimized implementation must have a trusted reference implementation.

Validation model:

```text
reference implementation
        ↕
optimized implementation
```

Correctness must use operation-specific rules.

Examples:

```text
integer → exact
FP32 → defined numerical tolerance
FP16/BF16 → defined numerical tolerance
quantized → format-specific tolerance
stochastic → deterministic test mode plus statistical validation
```

Reference implementations prioritize trustworthiness over speed.

---

# 15. Memory Contract

Centralize:

- host allocation;
- device allocation;
- mapped allocation;
- staging allocation;
- workspace allocation;
- temporary allocation;
- persistent allocation;
- arena allocation;
- pooled allocation.

Runtime must expose:

- memory pressure;
- allocator statistics;
- allocation failures;
- resource accounting.

Tensor/graph/kernel layers must not independently invent incompatible ownership models.

---

# 16. Platform Contract

Windows and Linux must expose equivalent public semantics.

Platform-specific code is permitted only where OS or driver APIs require it.

Conceptually:

```text
common contract
     ↑       ↑
Windows   Linux
```

not two independent engines.

Windows platform integration and Linux platform integration must both be tested independently.

---

# 17. Build Contract

The implementation must provide a reproducible build matrix for:

```text
Windows:
    MSVC
    compatible Clang/LLVM path where supported

Linux:
    GCC
    Clang
```

Required configurations:

```text
Debug
Release
RelWithDebInfo
```

Testing configurations must support appropriate sanitizers and diagnostics where the toolchain permits.

Python bindings must be built and tested against the supported Python versions defined by the implementation release.

---

# 18. Dependency Contract

Dependencies must be classified:

```text
required build dependency
optional build dependency
required runtime dependency
optional runtime dependency
platform dependency
test-only dependency
```

The full Vulkan SDK must not become an accidental runtime requirement merely because it was used for development.

Runtime deployment requirements must be documented separately from development requirements.

---

# 19. Input and Model Security Contract

External model/data inputs must be validated before use.

Validation includes:

- file bounds;
- offsets;
- sizes;
- integer overflow;
- tensor dimensions;
- metadata;
- allocation sizes;
- graph structure;
- tokenizer data;
- quantization metadata;
- unsupported combinations.

Malformed external data must fail cleanly without unsafe allocation or execution.

---

# 20. Module Contracts

Every module below has a mandatory **purpose** and **use**. These fields are not optional documentation. They define the boundary and must be included in the module's implementation trace.

---

## Module 01 — `vaist_core`

### Purpose

Provide the platform-neutral foundation and canonical C99 ABI.

### Mandatory use

All other public modules use `vaist_core` for:

- ABI types;
- status/error definitions;
- versioning;
- common handles;
- flags;
- configuration contracts;
- logging/diagnostics contracts;
- portable utility contracts.

### Must not contain

- Vulkan device implementation;
- graph logic;
- NN operations;
- model parsers;
- engine orchestration.

### Required validation

- C99 ABI compilation;
- C++ ABI consumption;
- Python binding consumption;
- Windows;
- Linux;
- status/error behavior;
- version queries;
- structure size/version behavior;
- ABI stability tests.

---

## Module 02 — `vaist_runtime`

### Purpose

Provide the execution/runtime abstraction over Vulkan and CPU backends.

### Mandatory use

All execution layers use it for:

- backend selection;
- device creation;
- physical-device selection;
- capabilities;
- Vulkan context;
- CPU backend;
- memory;
- queues;
- streams;
- events;
- synchronization;
- descriptor resources;
- pipeline infrastructure;
- upload/download;
- native Vulkan interop;
- memory pressure;
- allocator statistics.

### Existing reuse

Reuse the existing runtime/stream foundation where valid.

### Must not contain

- model-family logic;
- high-level engine orchestration;
- graph semantics.

---

## Module 03 — `vaist_tensor`

### Purpose

Provide the universal tensor/storage abstraction.

### Mandatory use

All compute, graph, NN, model execution and relevant LLM code use it for:

- dtype;
- shape;
- stride;
- layout;
- storage;
- device;
- views;
- slices;
- reshape;
- transpose;
- broadcast;
- cast;
- copy.

### Mandatory tensor invariants

Define and validate:

- contiguous layout;
- strided layout;
- byte offsets;
- alignment;
- aliasing;
- broadcast views;
- overlapping views;
- storage ownership.

---

## Module 04 — `vaist_compute`

### Purpose

Provide general numerical computation and kernel execution.

### Mandatory use

Contains:

- math;
- elementwise operations;
- activations;
- broadcast;
- reductions;
- BLAS;
- BLAS L1/L2;
- LAPACK;
- FFT;
- sparse;
- random;
- normalization primitives;
- general kernels;
- kernel dispatch.

### Existing reuse

Reuse:

- `vkmath`;
- `vkblas`;
- `vkblas_l1l2`;
- `vkfft`;
- `vkrand`.

### Mandatory kernel classes

Potential implementations include:

- scalar;
- tiled;
- subgroup;
- integer-dot;
- packed integer-dot;
- CPU SIMD;
- other validated non-cooperative implementations.

The selector chooses the fastest valid implementation.

### Cooperative matrix

Not a production candidate.

---

## Module 05 — `vaist_quant`

### Purpose

Provide unified tensor/model quantization and dequantization.

### Mandatory use

Contains:

- quantization formats;
- quantization;
- dequantization;
- calibration;
- scale management;
- mixed precision;
- per-tensor;
- per-channel;
- per-group;
- format validation.

### Existing formats to retain

The consolidation plan identifies existing support including:

```text
Q2
Q3
Q4
Q5
Q6
Q8
IQ
TQ
```

### Mandatory format semantics

Each format must define:

- block/group size;
- scale;
- zero point if applicable;
- rounding;
- saturation;
- packing;
- alignment;
- accumulation precision;
- dequantization behavior.

---

## Module 06 — `vaist_graph`

### Purpose

Provide graph IR, compilation/planning, validation and execution-plan generation.

### Mandatory use

Contains:

- graph IR;
- operator registry;
- shape inference;
- graph validation;
- constant folding;
- fusion;
- layout propagation;
- precision propagation;
- kernel registry;
- kernel selection;
- autotuning;
- memory planning;
- execution plans;
- execution-plan serialization.

### Mandatory architecture

```text
model config
   ↓
generic operators
   ↓
graph IR
   ↓
optimization/planning
   ↓
kernel selection
   ↓
execution plan
```

Do not create monolithic model-family execution switches.

---

## Module 07 — `vaist_model`

### Purpose

Provide a common model-loading and model-representation layer.

### Mandatory use

Supports the model formats specified by the consolidation plan:

- GGUF;
- SafeTensors;
- OpenVINO;
- ONNX.

Also provides:

- common model representation;
- metadata;
- weight streaming;
- lazy loading;
- mmap;
- sharded weights;
- checkpoint loading/saving;
- model validation.

### Mandatory architecture

```text
format adapter
   ↓
common model representation
   ↓
graph
   ↓
NN/runtime
```

Model formats must not create separate execution architectures.

---

## Module 08 — `vaist_nn`

### Purpose

Provide generic neural-network operators and Transformer primitives.

### Mandatory use

### Basic operators

- Add
- Sub
- Mul
- Div
- MatMul
- Linear
- Embedding
- Conv
- Pool
- Upsample

### Activations

- ReLU
- GELU
- SiLU
- Tanh
- Sigmoid
- Softmax

### Normalization

- LayerNorm
- RMSNorm
- BatchNorm
- GroupNorm

### Attention

- MHA
- MQA
- GQA
- cross attention
- causal attention
- sliding-window attention
- local attention

### Position encoding

- RoPE
- ALiBi
- scaling variants

### Transformer

- encoder;
- decoder;
- decoder-only;
- FFN;
- SwiGLU;
- GeGLU;
- residual;
- LM head.

### MoE

- router;
- top-k;
- expert dispatch;
- expert combine;
- load balancing.

All operations use `vaist_tensor`, `vaist_compute`, `vaist_quant`, `vaist_runtime` and graph infrastructure through defined boundaries.

---

## Module 09 — `vaist_llm`

### Purpose

Provide autoregressive LLM-specific runtime functionality.

### Mandatory use

Contains:

- tokenizer;
- chat templates;
- KV cache;
- sampling;
- decoding;
- grammar;
- structured output;
- speculative decoding;
- generation state.

### Tokenizers

Support the specified families:

- BPE;
- SentencePiece;
- Unigram;
- WordPiece;
- byte-level BPE.

### Sampling

Support:

- greedy;
- temperature;
- top-k;
- top-p;
- min-p;
- typical;
- Mirostat;
- beam;
- contrastive;
- repetition penalty;
- frequency penalty;
- presence penalty;
- logit bias.

### KV

Provide:

- contiguous KV;
- paged KV;
- allocation;
- eviction;
- prefix sharing;
- copy-on-write;
- KV quantization;
- KV compression;
- KV serialization;
- transactional KV;
- commit;
- rollback.

### Speculative decoding

Provide:

```text
draft model
candidate generation
verification
KV transaction
commit
rollback
```

---

## Module 10 — `vaist_engine`

### Purpose

Provide the application-facing orchestration layer.

### Mandatory use

Provides:

- engine lifecycle;
- model lifecycle;
- session lifecycle;
- graph execution orchestration;
- generation lifecycle;
- scheduling;
- batching;
- cancellation;
- streaming.

Representative application flow:

```text
engine create
   ↓
model load
   ↓
session create
   ↓
generate
```

### Must not own

- kernels;
- Vulkan implementation;
- quantization algorithms;
- model format parser internals;
- tokenizer internals.

It orchestrates other libraries.

---

## Module 11 — `vaist_ai`

### Purpose

Provide the optional high-level AI layer.

### Mandatory use

Source modules include:

```text
vision/
audio/
retrieval/
rag/
training/
autograd/
agents/
```

Functionality includes:

- vision;
- image;
- audio;
- speech;
- multimodal;
- embeddings;
- vector search;
- RAG;
- reranking;
- training;
- autograd;
- optimizers;
- losses;
- agents;
- tools;
- memory.

It remains one optional public library.

---

## Module 12 — `vaist_distributed`

### Purpose

Provide multi-process/multi-device/networked execution.

### Mandatory use

Reuse existing distributed infrastructure where valid:

- transport;
- master/worker;
- capability discovery;
- remote buffers;
- distributed GEMM.

Add:

- all-reduce;
- all-gather;
- reduce-scatter;
- broadcast;
- data parallelism;
- tensor parallelism;
- pipeline parallelism;
- expert parallelism;
- distributed KV;
- distributed scheduling.

### Mandatory failure contract

Later implementation must explicitly define:

- worker failure;
- timeout;
- reconnect;
- rank failure;
- collective cancellation;
- partial failure;
- version mismatch;
- network backpressure.

---

# 21. `DO_NOT_USE` Cooperative Matrix Module

## Purpose

Provide a complete isolated reference/experimental implementation only.

## Mandatory use

**Do not use in production.**

It exists solely for:

- reference;
- experimental comparison;
- future investigation;
- isolated testing.

## Location

```text
vulkan/DO_NOT_USE/cooperative_matrix/
```

## Mandatory implementation

It must still obey all `rules.md` requirements:

- complete C99;
- complete C++;
- complete Python;
- Windows;
- Linux;
- build files;
- tests;
- error handling;
- capability detection;
- documentation.

## Hard isolation

Production code must have no dependency edge to this module.

It must not appear in:

- production kernel registry;
- production dispatch;
- production autotuning;
- fallback selection;
- default builds;
- required capability checks.

---

# 22. Dependency Direction

Primary dependency direction:

```text
core
  ↓
runtime
  ↓
tensor
  ↓
compute / quant
  ↓
graph
  ↓
nn / model
  ↓
llm
  ↓
engine
  ↓
ai / distributed
```

Controlled relationships:

```text
model → graph
nn → graph
llm → nn + model
engine → graph + model + llm
```

Forbidden architectural direction includes:

```text
core → runtime
runtime → graph
runtime → nn
runtime → engine
tensor → engine
```

Lower layers must not depend on high-level orchestration.

---

# 23. Module Trace Requirement

After every module is implemented, provide a complete trace containing:

```text
MODULE
VERSION
VULKAN TARGET
WINDOWS STATUS
LINUX STATUS
C99 STATUS
C++ STATUS
PYTHON STATUS

PURPOSE
MANDATORY USE

FILES CREATED
FILES REUSED
FILES MODIFIED
FILES REWIRED
FILES REPLACED

PUBLIC API
INTERNAL API

DEPENDENCIES
DEPENDENCY DIRECTION

DATA MODEL
CONTROL FLOW
DATA FLOW

OWNERSHIP
LIFETIME
THREADING
SYNCHRONIZATION

CAPABILITY REQUIREMENTS
DISPATCH CANDIDATES
FALLBACK CHAIN

REFERENCE IMPLEMENTATION
NUMERICAL CONTRACT

WINDOWS BUILD
LINUX BUILD

C99 TESTS
C++ TESTS
PYTHON TESTS
ABI TESTS
NEGATIVE TESTS
RESOURCE TESTS
CROSS-PLATFORM TESTS
PERFORMANCE TESTS
REGRESSION TESTS

OPTIMIZATIONS
OPTIMIZATION EFFECT

FAILURES FOUND
FIXES APPLIED
RETEST RESULTS

FINAL STATUS
PASS / FAIL

MODULE LOCK
YES / NO
```

---

# 24. Validation Levels

Every module must pass:

```text
L0  source completeness
L1  compile
L2  API/ABI
L3  unit tests
L4  negative/error tests
L5  ownership/lifetime
L6  Windows
L7  Linux
L8  C99
L9  C++
L10 Python
L11 reference-vs-optimized
L12 stress
L13 performance
L14 regression
```

A module is locked only when all applicable levels pass.

---

# 25. Production Dispatch Example

For a numerical operation:

```text
operation request
       ↓
capability discovery
       ↓
candidate implementations
       ↓
remove unsupported
       ↓
remove semantically invalid
       ↓
remove numerically invalid
       ↓
remove resource-infeasible
       ↓
workload ranking
       ↓
cached/autotuned performance
       ↓
fastest valid implementation
       ↓
execute
```

Potential candidate classes:

```text
CPU scalar
CPU SIMD
Vulkan scalar
Vulkan tiled
Vulkan subgroup
Vulkan integer-dot
Vulkan packed-dot
other validated specialized implementations
```

Cooperative matrix is absent.

---

# 26. Runtime Failure Policy

Failures are classified.

## Non-recoverable

Return immediately:

- invalid argument;
- invalid graph;
- malformed model;
- ABI mismatch;
- impossible shape;
- unsupported semantic request.

## Potentially recoverable

Use the fallback/recovery policy:

- temporary workspace exhaustion;
- implementation-specific resource exhaustion;
- supported-but-unavailable optimization;
- transient scheduling/resource condition.

## Device-loss class

Enter explicit device-loss handling.

Do not silently pretend the operation succeeded.

---

# 27. No Hidden Work Contract

Every operation must make significant work visible to the planner/runtime.

This includes:

- allocations;
- transfers;
- synchronization;
- workspace requirements;
- kernel dispatch;
- backend transitions.

The graph/runtime layers must be able to reason about these costs.

---

# 28. Model Execution Contract

All model formats must converge toward common execution:

```text
GGUF / SafeTensors / OpenVINO / ONNX
                ↓
        common model representation
                ↓
             graph IR
                ↓
           NN operators
                ↓
          kernel selection
                ↓
             runtime
                ↓
       Vulkan or CPU backend
```

No format-specific hidden inference engine.

---

# 29. Reuse Matrix

The existing consolidation plan identifies the following valuable implementation assets:

```text
vkruntime
vkstream
vkmath
vkblas
vkblas_l1l2
vkfft
vkrand
vkquant
vkmodel
vkkv
vkdist
```

They should be reused behind the appropriate new public boundaries rather than rewritten without technical reason.

Target mapping:

```text
vkruntime      → vaist_runtime
vkstream       → vaist_runtime
vkmath         → vaist_compute
vkblas         → vaist_compute
vkblas_l1l2    → vaist_compute
vkfft          → vaist_compute
vkrand         → vaist_compute
vkquant        → vaist_quant
vkmodel        → vaist_model
vkkv           → vaist_llm
vkdist         → vaist_distributed
```

---

# 30. Implementation Order

Strict order:

```text
01 vaist_core
02 vaist_runtime
03 vaist_tensor
04 vaist_compute
05 vaist_quant
06 vaist_graph
07 vaist_model
08 vaist_nn
09 vaist_llm
10 vaist_engine
11 vaist_ai
12 vaist_distributed

FINAL:
DO_NOT_USE/cooperative_matrix
```

The order may not be bypassed merely because a later module is easier to implement.

Dependencies must be stabilized first.

---

# 31. Per-Module Optimization Rule

Optimization starts only after correctness.

```text
correct
  ↓
cross-platform equivalent
  ↓
profile
  ↓
identify bottleneck
  ↓
optimize
  ↓
reference comparison
  ↓
resource validation
  ↓
performance validation
  ↓
regression
```

An optimization that violates correctness, ABI, ownership, or platform semantics is rejected.

---

# 32. Required Documentation Inside Every Module

Each module directory must contain documentation describing:

```text
PURPOSE
MANDATORY USE
PUBLIC API
INTERNAL COMPONENTS
DEPENDENCIES
OWNERSHIP
LIFETIME
THREADING
SYNCHRONIZATION
CAPABILITIES
DISPATCH
FALLBACKS
ERRORS
TESTS
PERFORMANCE
PLATFORM DIFFERENCES
```

These fields are mandatory.

---

# 33. Definition of Done

A module is complete only when:

```text
[ ] Purpose documented
[ ] Mandatory use documented
[ ] API complete
[ ] Implementation complete
[ ] Windows C99 complete
[ ] Windows C++ complete
[ ] Windows Python complete
[ ] Linux C99 complete
[ ] Linux C++ complete
[ ] Linux Python complete
[ ] ABI verified
[ ] Ownership verified
[ ] Lifetime verified
[ ] Threading verified
[ ] Synchronization verified
[ ] Error paths verified
[ ] Capability detection verified
[ ] Dispatch verified
[ ] Fallback verified
[ ] Reference implementation verified
[ ] Negative tests pass
[ ] Cross-platform tests pass
[ ] Performance tested
[ ] Optimization retested
[ ] Regression tests pass
[ ] No hidden copies
[ ] No hidden synchronization
[ ] No placeholders
[ ] No omitted code
[ ] No unauthorized dependency
[ ] Module trace complete
[ ] Module locked
```

---

# 34. Final Architectural State

The resulting system is:

```text
                         VAiSt
                           │
                    vaist_core
                           │
                    vaist_runtime
                    /             \
               Vulkan             CPU
                    \             /
                     vaist_tensor
                           │
                 ┌─────────┴─────────┐
                 │                   │
           vaist_compute        vaist_quant
                 │                   │
                 └─────────┬─────────┘
                           │
                     vaist_graph
                           │
                  ┌────────┴────────┐
                  │                 │
              vaist_nn         vaist_model
                  │                 │
                  └────────┬────────┘
                           │
                       vaist_llm
                           │
                     vaist_engine
                       /       \
                vaist_ai    vaist_distributed
```

Cross-cutting:

```text
ABI
Errors
Logging
Diagnostics
Capabilities
Dispatch
Autotuning
Memory
Ownership
Threading
Synchronization
Testing
Profiling
Serialization
```

Production acceleration:

```text
fastest valid supported implementation
```

not a fixed hardware instruction.

Cooperative matrix:

```text
vulkan/DO_NOT_USE/cooperative_matrix/
```

and nowhere in the production dependency or dispatch graph.

---

# 35. Locked Execution Directive

Implementation must now proceed one module at a time.

For each module:

1. inspect existing code relevant to the module;
2. classify existing code as KEEP / MERGE / EXTEND / REWIRE / REPLACE / NEW;
3. implement the complete Windows C99 version;
4. implement the complete Windows C++ layer;
5. implement the complete Windows Python layer;
6. implement the complete Linux C99 version;
7. implement the complete Linux C++ layer;
8. implement the complete Linux Python layer;
9. produce the module trace;
10. compile;
11. test;
12. validate ABI;
13. validate runtime behavior;
14. validate errors and resources;
15. validate Windows and Linux;
16. validate C99/C++/Python;
17. compare optimized implementations against the reference;
18. optimize;
19. retest;
20. lock the module;
21. only then begin the next module.

**No module may be declared complete by inspection alone.**

**No later module may compensate for an incomplete earlier module.**

**No cooperative-matrix implementation may enter production.**

**No implementation may violate `rules.md`.**

---

# 36. Source Basis

The canonical architecture, reuse mapping, 12-library boundaries, module purposes, migration order, and existing implementation assets are derived from the VAiSt consolidation plan.

The complete-file/no-placeholder/no-omission requirements are derived from `rules.md`.

This document records the additional engineering contracts and explicit decisions required to execute that architecture safely and consistently.
