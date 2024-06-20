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

// Device code
__global__ void VecAdd(float* A, float* B, float* C, int N)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N)
    C[i] = A[i] + B[i];
}

// Host code
void H_VecAdd(const float* A, const float* B, float* C, int N)
{
  for (int i = 0; i < N; ++i)
    C[i] = A[i] + B[i];
}

int main()
{
  int N = 100'000'000;
  size_t size = N * sizeof(float);

  // Allocate input vectors h_A and h_B in host memory
  float* h_A = (float*)malloc(size);
  float* h_B = (float*)malloc(size);
  float* h_C = (float*)malloc(size);

  // Initialize input vectors
  // Create a random device and a random number generator
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0); // Floating point numbers between 0.0 and 1.0

  std::cout << "Generating random numbers..." << std::endl;
  for (int i = 0; i < N; ++i)
  {
    h_A[i] = static_cast<float>(dis(gen));
    h_B[i] = static_cast<float>(dis(gen));
    h_C[i] = 0.0f;
  }
  std::cout << "Random numbers generated." << std::endl;

  // Allocate vectors in device memory
  float* d_A;
  cudaMalloc(&d_A, size);
  float* d_B;
  cudaMalloc(&d_B, size);
  float* d_C;
  cudaMalloc(&d_C, size);

  // Copy vectors from host memory to device memory
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  auto start = std::chrono::high_resolution_clock::now();
  // Invoke kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

  std::cout << "Device time: " << std::chrono::duration<double, std::milli>{ std::chrono::high_resolution_clock::now() - start }.count() << " ms" << std::endl;

  start = std::chrono::high_resolution_clock::now();
  // Perform same action, but in host environment, measuring time
  H_VecAdd(h_A, h_B, h_C, N);

  std::cout << "Host time: " << std::chrono::duration<double, std::milli>{ std::chrono::high_resolution_clock::now() - start }.count() << " ms" << std::endl;

  // Copy result from device memory to host memory
  // h_C contains the result in host memory
  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // Display first and last operation
  std::cout << "First operation: " << h_A[0] << " + " << h_B[0] << " = " << h_C[0] << std::endl;
  std::cout << "Last operation: " << h_A[N - 1] << " + " << h_B[N - 1] << " = " << h_C[N - 1] << std::endl;

  // Free host memory
  free(h_A);
  free(h_B);
  free(h_C);
}
