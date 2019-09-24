#ifndef NDEBUG_H
#define NDEBUG_H

#include <imgui.h>

#include <vector>

namespace nova {
  class Debug final {
  public:
    explicit Debug(unsigned int maxFPS) : mMaxFPS(maxFPS) {}

    void FPS() {
      int fps = glfwGetTime();

      if (mFrames.size() > 100) {
        for (std::size_t i = 1; i < mFrames.size(); i++)
          mFrames[i - 1] = mFrames[i];
        mFrames[mFrames.size() - 1] = fps;
      } else
        mFrames.push_back(fps);

      ImGui::NewFrame();

      ImGui::PlotHistogram("Framerate", &mFrames[0], mFrames.size(), 0, nullptr, 0.f, 100.f, ImVec2(300, 100));

      if (ImGui::SliderInt("Max FPS", &mMaxFPS, 0, 200, nullptr)) {

      }

      ImGui::EndFrame();
    }

  private:
    int mMaxFPS;

    std::vector<float> mFrames;
  };
}

#endif