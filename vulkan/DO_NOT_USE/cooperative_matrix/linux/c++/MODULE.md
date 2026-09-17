# DO NOT USE — Cooperative Matrix

PURPOSE
: Isolated archival/experimental boundary for cooperative-matrix experimentation.

MANDATORY USE
: None. This module must not be used by production code.

PUBLIC API
: None.

PRODUCTION STATUS
: Forbidden from production dependency, dispatch, autotuning and correctness paths.

FALLBACK
: Production code uses capability-driven scalar/tiled, subgroup, integer-dot and packed-dot implementations where valid.

VULKAN TARGET
: Vulkan 1.4.357.
