/**
 * This program is a simple test program that adds two vectors of floating point numbers.
 * The program is divided into two parts: the device code and the host code.
 * The device code is a CUDA kernel that adds two vectors of floating point numbers.
 * The host code is a function that adds two vectors of floating point numbers.
 * The program measures the time it takes to add two vectors of floating point numbers using the device code and the host code.
 *
 * Example output:
 * ```
 * Generating random numbers...
 * Random numbers generated.
 * Device time: 1.30158 ms
 * Host time: 47.7095 ms
 * First operation: 0.931856 + 0.188839 = 1.1207
 * Last operation: 0.685969 + 0.383569 = 1.06954
 * ```
 */

#include <random>
#include <iostream>
#include <chrono>

#define THREADS_PER_BLOCK 256

#define precision_t double

// Device code
__global__ void DotProductKernel(precision_t* A, precision_t* B, precision_t* C, int N)
{
  __shared__ precision_t temp[THREADS_PER_BLOCK];
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  if (index < N)
    temp[threadIdx.x] = A[index] * B[index];
  else
    temp[threadIdx.x] = 0;

  // Synchronize all threads in the block
  __syncthreads();

  // Perform reduction only on first thread of the block
  if (threadIdx.x == 0)
  {
    precision_t sum = 0;
    for (int i = 0; i < blockDim.x; ++i)
      sum += temp[i];

    atomicAdd(C, sum);
  }
}

// Host code
void H_DotProductKernel(const precision_t* A, const precision_t* B, precision_t* C, int N)
{
  for(int i = 0; i < N; ++i)
    *C += A[i] * B[i];
}

int main()
{
  int N = 100'000'000;
  size_t size = N * sizeof(precision_t);

  // Allocate input vectors h_A and h_B in host memory
  precision_t* h_A = (precision_t*)malloc(size);
  precision_t* h_B = (precision_t*)malloc(size);
  precision_t* h_C = (precision_t*)malloc(sizeof(precision_t));

  // Initialize input vectors
  // Create a random device and a random number generator
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0); // Floating point numbers between 0.0 and 1.0

  std::cout << "Generating random numbers..." << std::endl;
  for (int i = 0; i < N; ++i)
  {
    h_A[i] = static_cast<precision_t>(dis(gen));
    h_B[i] = static_cast<precision_t>(dis(gen));
  }
  std::cout << "Random numbers generated." << std::endl;

  // Allocate vectors in device memory
  precision_t* d_A;
  cudaMalloc(&d_A, size);
  precision_t* d_B;
  cudaMalloc(&d_B, size);
  precision_t* d_C;
  cudaMalloc(&d_C, sizeof(precision_t));

  // Copy vectors from host memory to device memory
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
  cudaMemset(d_C, 0, sizeof(precision_t));

  auto start = std::chrono::high_resolution_clock::now();
  // Invoke kernel
  int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  DotProductKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, N);

  std::cout << "Device time: " << std::chrono::duration<double, std::milli>{ std::chrono::high_resolution_clock::now() - start }.count() << " ms" << std::endl;

  start = std::chrono::high_resolution_clock::now();
  // Perform same action, but in host environment, measuring time
  H_DotProductKernel(h_A, h_B, h_C, N);

  std::cout << "Host time: " << std::chrono::duration<double, std::milli>{ std::chrono::high_resolution_clock::now() - start }.count() << " ms" << std::endl;

  // Copy result from device memory to host memory
  // h_C contains the result in host memory
  cudaMemcpy(h_C, d_C, sizeof(precision_t), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // Display first and last operation
  std::cout << "Dot notation result: " << std::fixed << *h_C << std::endl;

  // Free host memory
  free(h_A);
  free(h_B);
  free(h_C);
}
