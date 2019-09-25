#ifndef NOVA_DEBUG_H
#define NOVA_DEBUG_H

#include <imgui.h>

#include <vector>

namespace nova {
  class Debug final {
  public:
    explicit Debug(unsigned int size) {
      mHistories.resize(size);
    }

    void History(const char *label, int idx, float time) {
      std::vector<float> &frames = mHistories[idx];

      if (frames.size() > 100) {
        for (std::size_t i = 1; i < frames.size(); i++)
          frames[i - 1] = frames[i];
        frames[frames.size() - 1] = time;
      } else
        frames.push_back(time);

      ImGui::PlotHistogram(label, &frames[0], frames.size(), 0, nullptr, 0.f, 1.f, ImVec2(300, 100));
    }

  private:
    std::vector<std::vector<float>> mHistories;
  };
}

#endif