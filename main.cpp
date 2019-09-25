#include <glad/glad.h>

#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>

#include <iostream>
#include <stdexcept>

#include "application.h"
#include "cuda.h"
#include "kernel.h"
#include "framebuffer.h"
#include "debug.h"

int main() {
  int maxFPS = 60;
  double fpsLimit = 1.0 / maxFPS;

  const int winWidth = 800;
  const int winHeight = 800;

  const int threadX = 8;
  const int threadY = 8;

  const int fbWidth = 256;
  const int fbHeight = 256;

  nova::Application app(winWidth, winHeight, "Nova");
  nova::Framebuffer fb(fbWidth, fbHeight);
  nova::Debug debug(3);

  double lastUpdateTime = 0;
  double lastFrameTime = 0;

  while (app.Running()) {
    double now = glfwGetTime();
    double deltaTime = now - lastUpdateTime;

    glfwPollEvents();

    fb.Compute();
    fb.Bind();

    if ((now - lastFrameTime) >= fpsLimit) {
      glClearColor(0.f, 0.f, 0.f, 1.f);
      glClear(GL_COLOR_BUFFER_BIT);

      glBlitFramebuffer(0, 0, fb.Width(), fb.Height(), 0, 0, app.Width(), app.Height(), GL_COLOR_BUFFER_BIT,
                        GL_NEAREST);

      ImGui_ImplOpenGL3_NewFrame();
      ImGui_ImplGlfw_NewFrame();

      {
        ImGui::NewFrame();

        if (ImGui::SliderInt("FPS Limit", &maxFPS, 1, 200, nullptr))
          fpsLimit = 1.0 / maxFPS;

        debug.History("Delta", 0, static_cast<float>(deltaTime));
        debug.History("Update", 1, static_cast<float>(lastUpdateTime));
        debug.History("Frame", 2, static_cast<float>(lastFrameTime));

        ImGui::EndFrame();
      }

      ImGui::Render();

      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

      glfwSwapBuffers(&app.Window());

      lastFrameTime = now;
    }

    lastUpdateTime = now;
  }
}