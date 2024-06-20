cmake_minimum_required( VERSION 3.20 )
# https://developer.nvidia.com/blog/building-cuda-applications-cmake/
project( CudaTest VERSION 0.1.0 DESCRIPTION "Testing CUDA in C++" LANGUAGES CXX CUDA )

add_subdirectory( programs )
