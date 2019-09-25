#ifndef NOVA_KERNEL_CU
#define NOVA_KERNEL_CU

#include <cstddef>

#include <device_launch_parameters.h>

namespace nova {
  __global__ void computeFrame(unsigned char *frame, int maxX, int maxY) {
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= maxX || j >= maxY) return;

    unsigned int idx = i * 4 + j * 4 * maxX;

    frame[idx] = i % 255;
    frame[idx + 1] = j % 255;
    frame[idx + 2] = (i * j) % 255;
    frame[idx + 3] = 255;
  }

  template<std::size_t Blocks, std::size_t Threads>
  void computeFrame(unsigned char *frame, unsigned int maxX, unsigned int maxY) {
    computeFrame << < Blocks, Threads >> > (frame, maxX, maxY);
  }
}

#endif