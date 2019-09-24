#include <glad/glad.h>

#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <iostream>
#include <stdexcept>

#include "ntime.h"
#include "ndebug.h"

#define CUDA_CHECK(v) check_cuda(v, #v, __FILE__, __LINE__)

void check_cuda(cudaError_t err, const char *func, const char *file, const int line) {
  if (!err) return;

  std::cerr << "cuda error: " << static_cast<unsigned int>(err) << " at " << file << ":" << line << "'" << func << "'"
            << std::endl;

  cudaDeviceReset();

  exit(1);
}

__global__ void render(unsigned char *frameBuffer, int maxX, int maxY) {
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i >= maxX || j >= maxY) return;

  unsigned int idx = i * 4 + j * 4 * maxX;

  frameBuffer[idx] = i % 255;
  frameBuffer[idx + 1] = j % 255;
  frameBuffer[idx + 2] = (i * j) % 255;
  frameBuffer[idx + 3] = 255;
}

int main() {
  const char *winTitle = "Trace";
  const char *glslVersion = "#version 430";

  const int winWidth = 800;
  const int winHeight = 800;

  const int threadX = 8;
  const int threadY = 8;

  const int fbWidth = 256;
  const int fbHeight = 256;

  const int fbSize = fbWidth * fbHeight * 4;

  if (!glfwInit()) throw std::runtime_error("Failed to initialize glfw");

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

  GLFWwindow *window = glfwCreateWindow(winWidth, winHeight, winTitle, nullptr, nullptr);

  if (!window) throw std::runtime_error("Failed to create window");

  glfwMakeContextCurrent(window);

  if (!gladLoadGL()) throw std::runtime_error("Failed to initialize glad");

  IMGUI_CHECKVERSION();

  ImGui::CreateContext();
  ImGui::StyleColorsDark();

  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init(glslVersion);

  unsigned char *fb = nullptr;

  CUDA_CHECK(cudaMallocManaged(reinterpret_cast<void **>(&fb), fbSize));

  dim3 blocks(fbWidth / threadX, fbHeight / threadY);
  dim3 threads(threadX, threadY);

  render<<<blocks, threads>>>(fb, fbWidth, fbHeight);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  GLuint tex;
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, fbWidth, fbHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, fb);
  glBindTexture(GL_TEXTURE_2D, 0);

  GLuint fbo;
  glGenFramebuffers(1, &fbo);
  glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
  glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);

  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

  nova::Debug debug(60);

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    glClearColor(0.f, 0.f, 0.f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);

    glBlitFramebuffer(0, 0, fbWidth, fbHeight, 0, 0, winWidth, winHeight, GL_COLOR_BUFFER_BIT, GL_NEAREST);

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();

    debug.FPS();

    ImGui::Render();

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);
  }

  glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
  glDeleteFramebuffers(1, &fbo);

  CUDA_CHECK(cudaFree(fb));

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();

  ImGui::DestroyContext();

  glfwDestroyWindow(window);
  glfwTerminate();
}