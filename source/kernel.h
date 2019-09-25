#ifndef NOVA_KERNEL_H
#define NOVA_KERNEL_H

#include <cstddef>

namespace nova {
  template<std::size_t Blocks, std::size_t Threads>
  void computeFrame(unsigned char *frame, unsigned int maxX, unsigned int maxY);
}

#endif