#ifndef NOVA_FRAMEBUFFER_H
#define NOVA_FRAMEBUFFER_H

#include "kernel.h"

namespace nova {
  class Framebuffer final {
  public:
    explicit Framebuffer(unsigned int width, unsigned int height) :
        mWidth(width),
        mHeight(height) {
      CUDA_CHECK(cudaMallocManaged(reinterpret_cast<void **>(&mData), width * height * 4));
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

      glGenTextures(1, &mTex);
      glBindTexture(GL_TEXTURE_2D, mTex);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, mData);
      glBindTexture(GL_TEXTURE_2D, 0);

      glGenFramebuffers(1, &mFbo);
      glBindFramebuffer(GL_READ_FRAMEBUFFER, mFbo);
      glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, mTex, 0);
      glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    }

    virtual ~Framebuffer() {
      glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
      glDeleteFramebuffers(1, &mFbo);

      glBindTexture(GL_TEXTURE_2D, 0);
      glDeleteTextures(1, &mTex);

      CUDA_CHECK(cudaFree(mData));
    }

    void Compute() {
      //computeFrame<128, 128>(mData, mWidth, mHeight);
    }

    void Bind() {
      glBindFramebuffer(GL_READ_FRAMEBUFFER, mFbo);
    }

    unsigned int Width() const { return mWidth; }
    unsigned int Height() const { return mHeight; }

  private:
    unsigned int mWidth;
    unsigned int mHeight;

    unsigned int mTex = 0;
    unsigned int mFbo = 0;

    unsigned char *mData = nullptr;
  };
}

#endif