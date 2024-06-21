add_executable( test1 main.cu )

set_target_properties( test1 PROPERTIES CUDA_STANDARD 20 )

set_target_properties( test1 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property( TARGET test1 PROPERTY CUDA_SEPARABLE_COMPILATION ON )

# I wanted to use double in the example for atomicAdd but it seems that it is not supported on my GPU
set_target_properties( test1 PROPERTIES CUDA_ARCHITECTURES 60 )

if(APPLE)
  # We need to add the path to the driver (libcuda.dylib) as an rpath,
  # so that the static cuda runtime can find it at runtime.
  set_property( TARGET test1 PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} )
endif()

install( TARGETS
  test1

  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
