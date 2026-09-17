# VAiSt Full Implementation Continuation

This tree is the from-scratch baseline continuation. It preserves the 12 public boundaries and the Windows/Linux C99/C++/Python matrix.

## Hard decisions

- Vulkan target: 1.4.357.
- Production cooperative-matrix path: excluded.
- CPU fallback: mandatory.
- Fastest-valid implementation: mandatory selection rule.
- C99: canonical ABI.
- No placeholder source is accepted.
- `DO_NOT_USE/cooperative_matrix` is isolated from the production graph.

## Concrete hardening completed in this pass

1. Tensor views retain their storage owner until the view is destroyed.
2. Tensor reshape is rejected for non-contiguous views.
3. Tensor copy walks strides for non-contiguous tensors instead of blindly memcpy-ing.
4. Tensor element-count and byte-size overflow checks are explicit.
5. Integer-dot accumulation detects signed 32-bit overflow.
6. Distributed all-gather size multiplication is checked before arithmetic.
7. Model inspection validates basic file readability and format magic/version instead of reporting only a header heuristic.

## External validation still required

- Native Windows/MSVC or MinGW complete build/test.
- Real Vulkan 1.4.357 device discovery and execution.
- Real model fixtures for GGUF/SafeTensors/ONNX/OpenVINO.
- Performance measurements on target hardware.

Those are not fabricated as PASS.


## Component buildability remediation

The component matrix has been remediated: twelve independent C++ targets, twelve Python package modules, consistent Windows DLL naming, CTest coverage for all C++ and Python components, install/export rules, Linux clean-staging load validation, and Linux C99 exported-symbol checks are now present. See `COMPONENT_ISSUE_CLOSURE.md`.
