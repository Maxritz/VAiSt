# DO_NOT_USE Cooperative Matrix Trace

Production status: disabled. No production CMake target, registry, dispatch candidate, fallback, autotune candidate or correctness dependency references this tree.

Reason: VAiSt uses the fastest valid supported non-cooperative implementation. Integer-dot/packed-dot/subgroup/tiled/scalar candidates are selected through capability-driven dispatch.
