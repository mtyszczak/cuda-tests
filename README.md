# CUDA tests

CUDA tests

## Building

### Building on Ubuntu 22.04

Install [nvcc](https://developer.nvidia.com/cuda-downloads)

Remember to use the compatible `g++` version

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -GNinja ..
ninja -j$(nproc) all
```

This shall generate the binary executable in the build directory that can be run to verify that the make file indeed compiles the project

## License

See [LICENSE.md](LICENSE.md)
