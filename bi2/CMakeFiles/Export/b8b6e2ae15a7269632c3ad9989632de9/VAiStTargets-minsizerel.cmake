#----------------------------------------------------------------
# Generated CMake target import file for configuration "MinSizeRel".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "VAiSt::vaist_core" for configuration "MinSizeRel"
set_property(TARGET VAiSt::vaist_core APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(VAiSt::vaist_core PROPERTIES
  IMPORTED_IMPLIB_MINSIZEREL "${_IMPORT_PREFIX}/lib/vaist_core.lib"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/bin/vaist_core.dll"
  )

list(APPEND _cmake_import_check_targets VAiSt::vaist_core )
list(APPEND _cmake_import_check_files_for_VAiSt::vaist_core "${_IMPORT_PREFIX}/lib/vaist_core.lib" "${_IMPORT_PREFIX}/bin/vaist_core.dll" )

# Import target "VAiSt::vaist_runtime" for configuration "MinSizeRel"
set_property(TARGET VAiSt::vaist_runtime APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(VAiSt::vaist_runtime PROPERTIES
  IMPORTED_IMPLIB_MINSIZEREL "${_IMPORT_PREFIX}/lib/vaist_runtime.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_MINSIZEREL "VAiSt::vaist_core"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/bin/vaist_runtime.dll"
  )

list(APPEND _cmake_import_check_targets VAiSt::vaist_runtime )
list(APPEND _cmake_import_check_files_for_VAiSt::vaist_runtime "${_IMPORT_PREFIX}/lib/vaist_runtime.lib" "${_IMPORT_PREFIX}/bin/vaist_runtime.dll" )

# Import target "VAiSt::vaist_tensor" for configuration "MinSizeRel"
set_property(TARGET VAiSt::vaist_tensor APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(VAiSt::vaist_tensor PROPERTIES
  IMPORTED_IMPLIB_MINSIZEREL "${_IMPORT_PREFIX}/lib/vaist_tensor.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_MINSIZEREL "VAiSt::vaist_core"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/bin/vaist_tensor.dll"
  )

list(APPEND _cmake_import_check_targets VAiSt::vaist_tensor )
list(APPEND _cmake_import_check_files_for_VAiSt::vaist_tensor "${_IMPORT_PREFIX}/lib/vaist_tensor.lib" "${_IMPORT_PREFIX}/bin/vaist_tensor.dll" )

# Import target "VAiSt::vaist_compute" for configuration "MinSizeRel"
set_property(TARGET VAiSt::vaist_compute APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(VAiSt::vaist_compute PROPERTIES
  IMPORTED_IMPLIB_MINSIZEREL "${_IMPORT_PREFIX}/lib/vaist_compute.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_MINSIZEREL "VAiSt::vaist_core;VAiSt::vaist_runtime"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/bin/vaist_compute.dll"
  )

list(APPEND _cmake_import_check_targets VAiSt::vaist_compute )
list(APPEND _cmake_import_check_files_for_VAiSt::vaist_compute "${_IMPORT_PREFIX}/lib/vaist_compute.lib" "${_IMPORT_PREFIX}/bin/vaist_compute.dll" )

# Import target "VAiSt::vaist_quant" for configuration "MinSizeRel"
set_property(TARGET VAiSt::vaist_quant APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(VAiSt::vaist_quant PROPERTIES
  IMPORTED_IMPLIB_MINSIZEREL "${_IMPORT_PREFIX}/lib/vaist_quant.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_MINSIZEREL "VAiSt::vaist_core"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/bin/vaist_quant.dll"
  )

list(APPEND _cmake_import_check_targets VAiSt::vaist_quant )
list(APPEND _cmake_import_check_files_for_VAiSt::vaist_quant "${_IMPORT_PREFIX}/lib/vaist_quant.lib" "${_IMPORT_PREFIX}/bin/vaist_quant.dll" )

# Import target "VAiSt::vaist_graph" for configuration "MinSizeRel"
set_property(TARGET VAiSt::vaist_graph APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(VAiSt::vaist_graph PROPERTIES
  IMPORTED_IMPLIB_MINSIZEREL "${_IMPORT_PREFIX}/lib/vaist_graph.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_MINSIZEREL "VAiSt::vaist_core"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/bin/vaist_graph.dll"
  )

list(APPEND _cmake_import_check_targets VAiSt::vaist_graph )
list(APPEND _cmake_import_check_files_for_VAiSt::vaist_graph "${_IMPORT_PREFIX}/lib/vaist_graph.lib" "${_IMPORT_PREFIX}/bin/vaist_graph.dll" )

# Import target "VAiSt::vaist_model" for configuration "MinSizeRel"
set_property(TARGET VAiSt::vaist_model APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(VAiSt::vaist_model PROPERTIES
  IMPORTED_IMPLIB_MINSIZEREL "${_IMPORT_PREFIX}/lib/vaist_model.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_MINSIZEREL "VAiSt::vaist_core"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/bin/vaist_model.dll"
  )

list(APPEND _cmake_import_check_targets VAiSt::vaist_model )
list(APPEND _cmake_import_check_files_for_VAiSt::vaist_model "${_IMPORT_PREFIX}/lib/vaist_model.lib" "${_IMPORT_PREFIX}/bin/vaist_model.dll" )

# Import target "VAiSt::vaist_nn" for configuration "MinSizeRel"
set_property(TARGET VAiSt::vaist_nn APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(VAiSt::vaist_nn PROPERTIES
  IMPORTED_IMPLIB_MINSIZEREL "${_IMPORT_PREFIX}/lib/vaist_nn.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_MINSIZEREL "VAiSt::vaist_core"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/bin/vaist_nn.dll"
  )

list(APPEND _cmake_import_check_targets VAiSt::vaist_nn )
list(APPEND _cmake_import_check_files_for_VAiSt::vaist_nn "${_IMPORT_PREFIX}/lib/vaist_nn.lib" "${_IMPORT_PREFIX}/bin/vaist_nn.dll" )

# Import target "VAiSt::vaist_llm" for configuration "MinSizeRel"
set_property(TARGET VAiSt::vaist_llm APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(VAiSt::vaist_llm PROPERTIES
  IMPORTED_IMPLIB_MINSIZEREL "${_IMPORT_PREFIX}/lib/vaist_llm.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_MINSIZEREL "VAiSt::vaist_core"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/bin/vaist_llm.dll"
  )

list(APPEND _cmake_import_check_targets VAiSt::vaist_llm )
list(APPEND _cmake_import_check_files_for_VAiSt::vaist_llm "${_IMPORT_PREFIX}/lib/vaist_llm.lib" "${_IMPORT_PREFIX}/bin/vaist_llm.dll" )

# Import target "VAiSt::vaist_engine" for configuration "MinSizeRel"
set_property(TARGET VAiSt::vaist_engine APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(VAiSt::vaist_engine PROPERTIES
  IMPORTED_IMPLIB_MINSIZEREL "${_IMPORT_PREFIX}/lib/vaist_engine.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_MINSIZEREL "VAiSt::vaist_core"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/bin/vaist_engine.dll"
  )

list(APPEND _cmake_import_check_targets VAiSt::vaist_engine )
list(APPEND _cmake_import_check_files_for_VAiSt::vaist_engine "${_IMPORT_PREFIX}/lib/vaist_engine.lib" "${_IMPORT_PREFIX}/bin/vaist_engine.dll" )

# Import target "VAiSt::vaist_ai" for configuration "MinSizeRel"
set_property(TARGET VAiSt::vaist_ai APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(VAiSt::vaist_ai PROPERTIES
  IMPORTED_IMPLIB_MINSIZEREL "${_IMPORT_PREFIX}/lib/vaist_ai.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_MINSIZEREL "VAiSt::vaist_core"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/bin/vaist_ai.dll"
  )

list(APPEND _cmake_import_check_targets VAiSt::vaist_ai )
list(APPEND _cmake_import_check_files_for_VAiSt::vaist_ai "${_IMPORT_PREFIX}/lib/vaist_ai.lib" "${_IMPORT_PREFIX}/bin/vaist_ai.dll" )

# Import target "VAiSt::vaist_distributed" for configuration "MinSizeRel"
set_property(TARGET VAiSt::vaist_distributed APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(VAiSt::vaist_distributed PROPERTIES
  IMPORTED_IMPLIB_MINSIZEREL "${_IMPORT_PREFIX}/lib/vaist_distributed.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_MINSIZEREL "VAiSt::vaist_core"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/bin/vaist_distributed.dll"
  )

list(APPEND _cmake_import_check_targets VAiSt::vaist_distributed )
list(APPEND _cmake_import_check_files_for_VAiSt::vaist_distributed "${_IMPORT_PREFIX}/lib/vaist_distributed.lib" "${_IMPORT_PREFIX}/bin/vaist_distributed.dll" )

# Import target "VAiSt::vaist_blas" for configuration "MinSizeRel"
set_property(TARGET VAiSt::vaist_blas APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(VAiSt::vaist_blas PROPERTIES
  IMPORTED_IMPLIB_MINSIZEREL "${_IMPORT_PREFIX}/lib/vaist_blas.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_MINSIZEREL "VAiSt::vaist_core;VAiSt::vaist_runtime;VAiSt::vaist_compute;VAiSt::vaist_quant"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/bin/vaist_blas.dll"
  )

list(APPEND _cmake_import_check_targets VAiSt::vaist_blas )
list(APPEND _cmake_import_check_files_for_VAiSt::vaist_blas "${_IMPORT_PREFIX}/lib/vaist_blas.lib" "${_IMPORT_PREFIX}/bin/vaist_blas.dll" )

# Import target "VAiSt::vaist_linalg" for configuration "MinSizeRel"
set_property(TARGET VAiSt::vaist_linalg APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(VAiSt::vaist_linalg PROPERTIES
  IMPORTED_IMPLIB_MINSIZEREL "${_IMPORT_PREFIX}/lib/vaist_linalg.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_MINSIZEREL "VAiSt::vaist_core;VAiSt::vaist_blas;VAiSt::vaist_runtime"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/bin/vaist_linalg.dll"
  )

list(APPEND _cmake_import_check_targets VAiSt::vaist_linalg )
list(APPEND _cmake_import_check_files_for_VAiSt::vaist_linalg "${_IMPORT_PREFIX}/lib/vaist_linalg.lib" "${_IMPORT_PREFIX}/bin/vaist_linalg.dll" )

# Import target "VAiSt::vaist_attn" for configuration "MinSizeRel"
set_property(TARGET VAiSt::vaist_attn APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(VAiSt::vaist_attn PROPERTIES
  IMPORTED_IMPLIB_MINSIZEREL "${_IMPORT_PREFIX}/lib/vaist_attn.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_MINSIZEREL "VAiSt::vaist_core;VAiSt::vaist_runtime;VAiSt::vaist_compute;VAiSt::vaist_quant"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/bin/vaist_attn.dll"
  )

list(APPEND _cmake_import_check_targets VAiSt::vaist_attn )
list(APPEND _cmake_import_check_files_for_VAiSt::vaist_attn "${_IMPORT_PREFIX}/lib/vaist_attn.lib" "${_IMPORT_PREFIX}/bin/vaist_attn.dll" )

# Import target "VAiSt::vaist_cpp_core" for configuration "MinSizeRel"
set_property(TARGET VAiSt::vaist_cpp_core APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(VAiSt::vaist_cpp_core PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_MINSIZEREL "CXX"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/vaist_cpp_core.lib"
  )

list(APPEND _cmake_import_check_targets VAiSt::vaist_cpp_core )
list(APPEND _cmake_import_check_files_for_VAiSt::vaist_cpp_core "${_IMPORT_PREFIX}/lib/vaist_cpp_core.lib" )

# Import target "VAiSt::vaist_cpp_runtime" for configuration "MinSizeRel"
set_property(TARGET VAiSt::vaist_cpp_runtime APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(VAiSt::vaist_cpp_runtime PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_MINSIZEREL "CXX"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/vaist_cpp_runtime.lib"
  )

list(APPEND _cmake_import_check_targets VAiSt::vaist_cpp_runtime )
list(APPEND _cmake_import_check_files_for_VAiSt::vaist_cpp_runtime "${_IMPORT_PREFIX}/lib/vaist_cpp_runtime.lib" )

# Import target "VAiSt::vaist_cpp_tensor" for configuration "MinSizeRel"
set_property(TARGET VAiSt::vaist_cpp_tensor APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(VAiSt::vaist_cpp_tensor PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_MINSIZEREL "CXX"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/vaist_cpp_tensor.lib"
  )

list(APPEND _cmake_import_check_targets VAiSt::vaist_cpp_tensor )
list(APPEND _cmake_import_check_files_for_VAiSt::vaist_cpp_tensor "${_IMPORT_PREFIX}/lib/vaist_cpp_tensor.lib" )

# Import target "VAiSt::vaist_cpp_compute" for configuration "MinSizeRel"
set_property(TARGET VAiSt::vaist_cpp_compute APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(VAiSt::vaist_cpp_compute PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_MINSIZEREL "CXX"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/vaist_cpp_compute.lib"
  )

list(APPEND _cmake_import_check_targets VAiSt::vaist_cpp_compute )
list(APPEND _cmake_import_check_files_for_VAiSt::vaist_cpp_compute "${_IMPORT_PREFIX}/lib/vaist_cpp_compute.lib" )

# Import target "VAiSt::vaist_cpp_quant" for configuration "MinSizeRel"
set_property(TARGET VAiSt::vaist_cpp_quant APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(VAiSt::vaist_cpp_quant PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_MINSIZEREL "CXX"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/vaist_cpp_quant.lib"
  )

list(APPEND _cmake_import_check_targets VAiSt::vaist_cpp_quant )
list(APPEND _cmake_import_check_files_for_VAiSt::vaist_cpp_quant "${_IMPORT_PREFIX}/lib/vaist_cpp_quant.lib" )

# Import target "VAiSt::vaist_cpp_graph" for configuration "MinSizeRel"
set_property(TARGET VAiSt::vaist_cpp_graph APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(VAiSt::vaist_cpp_graph PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_MINSIZEREL "CXX"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/vaist_cpp_graph.lib"
  )

list(APPEND _cmake_import_check_targets VAiSt::vaist_cpp_graph )
list(APPEND _cmake_import_check_files_for_VAiSt::vaist_cpp_graph "${_IMPORT_PREFIX}/lib/vaist_cpp_graph.lib" )

# Import target "VAiSt::vaist_cpp_model" for configuration "MinSizeRel"
set_property(TARGET VAiSt::vaist_cpp_model APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(VAiSt::vaist_cpp_model PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_MINSIZEREL "CXX"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/vaist_cpp_model.lib"
  )

list(APPEND _cmake_import_check_targets VAiSt::vaist_cpp_model )
list(APPEND _cmake_import_check_files_for_VAiSt::vaist_cpp_model "${_IMPORT_PREFIX}/lib/vaist_cpp_model.lib" )

# Import target "VAiSt::vaist_cpp_nn" for configuration "MinSizeRel"
set_property(TARGET VAiSt::vaist_cpp_nn APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(VAiSt::vaist_cpp_nn PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_MINSIZEREL "CXX"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/vaist_cpp_nn.lib"
  )

list(APPEND _cmake_import_check_targets VAiSt::vaist_cpp_nn )
list(APPEND _cmake_import_check_files_for_VAiSt::vaist_cpp_nn "${_IMPORT_PREFIX}/lib/vaist_cpp_nn.lib" )

# Import target "VAiSt::vaist_cpp_llm" for configuration "MinSizeRel"
set_property(TARGET VAiSt::vaist_cpp_llm APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(VAiSt::vaist_cpp_llm PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_MINSIZEREL "CXX"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/vaist_cpp_llm.lib"
  )

list(APPEND _cmake_import_check_targets VAiSt::vaist_cpp_llm )
list(APPEND _cmake_import_check_files_for_VAiSt::vaist_cpp_llm "${_IMPORT_PREFIX}/lib/vaist_cpp_llm.lib" )

# Import target "VAiSt::vaist_cpp_engine" for configuration "MinSizeRel"
set_property(TARGET VAiSt::vaist_cpp_engine APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(VAiSt::vaist_cpp_engine PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_MINSIZEREL "CXX"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/vaist_cpp_engine.lib"
  )

list(APPEND _cmake_import_check_targets VAiSt::vaist_cpp_engine )
list(APPEND _cmake_import_check_files_for_VAiSt::vaist_cpp_engine "${_IMPORT_PREFIX}/lib/vaist_cpp_engine.lib" )

# Import target "VAiSt::vaist_cpp_ai" for configuration "MinSizeRel"
set_property(TARGET VAiSt::vaist_cpp_ai APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(VAiSt::vaist_cpp_ai PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_MINSIZEREL "CXX"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/vaist_cpp_ai.lib"
  )

list(APPEND _cmake_import_check_targets VAiSt::vaist_cpp_ai )
list(APPEND _cmake_import_check_files_for_VAiSt::vaist_cpp_ai "${_IMPORT_PREFIX}/lib/vaist_cpp_ai.lib" )

# Import target "VAiSt::vaist_cpp_distributed" for configuration "MinSizeRel"
set_property(TARGET VAiSt::vaist_cpp_distributed APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(VAiSt::vaist_cpp_distributed PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_MINSIZEREL "CXX"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/vaist_cpp_distributed.lib"
  )

list(APPEND _cmake_import_check_targets VAiSt::vaist_cpp_distributed )
list(APPEND _cmake_import_check_files_for_VAiSt::vaist_cpp_distributed "${_IMPORT_PREFIX}/lib/vaist_cpp_distributed.lib" )

# Import target "VAiSt::vaist_cpp_blas" for configuration "MinSizeRel"
set_property(TARGET VAiSt::vaist_cpp_blas APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(VAiSt::vaist_cpp_blas PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_MINSIZEREL "CXX"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/vaist_cpp_blas.lib"
  )

list(APPEND _cmake_import_check_targets VAiSt::vaist_cpp_blas )
list(APPEND _cmake_import_check_files_for_VAiSt::vaist_cpp_blas "${_IMPORT_PREFIX}/lib/vaist_cpp_blas.lib" )

# Import target "VAiSt::vaist_cpp_linalg" for configuration "MinSizeRel"
set_property(TARGET VAiSt::vaist_cpp_linalg APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(VAiSt::vaist_cpp_linalg PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_MINSIZEREL "CXX"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/vaist_cpp_linalg.lib"
  )

list(APPEND _cmake_import_check_targets VAiSt::vaist_cpp_linalg )
list(APPEND _cmake_import_check_files_for_VAiSt::vaist_cpp_linalg "${_IMPORT_PREFIX}/lib/vaist_cpp_linalg.lib" )

# Import target "VAiSt::vaist_cpp_attn" for configuration "MinSizeRel"
set_property(TARGET VAiSt::vaist_cpp_attn APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(VAiSt::vaist_cpp_attn PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_MINSIZEREL "CXX"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/vaist_cpp_attn.lib"
  )

list(APPEND _cmake_import_check_targets VAiSt::vaist_cpp_attn )
list(APPEND _cmake_import_check_files_for_VAiSt::vaist_cpp_attn "${_IMPORT_PREFIX}/lib/vaist_cpp_attn.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
