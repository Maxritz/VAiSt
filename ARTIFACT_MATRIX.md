# VAiSt Artifact Matrix

The project has 12 architectural components. The delivery matrix is separate from the phrase “12 public libraries”.

| Platform | C99 | C++ | Python |
|---|---:|---:|---:|
| Windows | 12 shared DLLs | 12 independent static libraries | 12 package modules/loaders |
| Linux | 12 shared objects | 12 independent static libraries | 12 package modules/loaders |

Total component/surface/platform deliverables: 72, excluding tests and the isolated `DO_NOT_USE/cooperative_matrix` reference tree.

C99 is the canonical ABI. C++ wrappers link to their corresponding C99 component. Python modules load the corresponding native C99 component and expose a package module.

Windows native shared-library naming is `vaist_<component>.dll`. Linux naming is `libvaist_<component>.so`.

The C++ aggregate `vaist_cpp_wrappers` is convenience-only and does not replace the twelve independent C++ targets.
