#ifndef NOVA_CUDA_H
#define NOVA_CUDA_H

#include <cuda_runtime.h>

#define CUDA_CHECK(v) checkCuda(v, #v, __FILE__, __LINE__)

void checkCuda(cudaError_t err, const char *func, const char *file, const int line) {
  if (err == cudaSuccess) return;

  std::cerr << "cuda error: " << err << " at " << file << ":" << line << "'" << func << "'" << std::endl;

  cudaDeviceReset();

  exit(1);
}

#endif