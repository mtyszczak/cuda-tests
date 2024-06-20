add_executable( test0 main.cu )

target_compile_features( test0 PUBLIC cxx_std_23 )

set_target_properties( test0 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property( TARGET test0 PROPERTY CUDA_SEPARABLE_COMPILATION ON )

if(APPLE)
  # We need to add the path to the driver (libcuda.dylib) as an rpath,
  # so that the static cuda runtime can find it at runtime.
  set_property( TARGET test0 PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} )
endif()

install( TARGETS
  test0

  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
