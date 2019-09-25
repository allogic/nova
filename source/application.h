#ifndef NOVA_APPLICATION_H
#define NOVA_APPLICATION_H

namespace nova {
  class Application final {
  public:
    explicit Application(unsigned int width, unsigned int height, const char *title) :
        mWidth(width),
        mHeight(height) {
      if (!glfwInit()) throw std::runtime_error("Failed to initialize glfw");

      glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
      glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
      glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
      glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

      mWindow = glfwCreateWindow(static_cast<int>(width), static_cast<int>(height), title, nullptr, nullptr);

      if (!mWindow) throw std::runtime_error("Failed to create window");

      glfwMakeContextCurrent(mWindow);

      if (!gladLoadGL()) throw std::runtime_error("Failed to initialize glad");

      IMGUI_CHECKVERSION();

      ImGui::CreateContext();
      ImGui::StyleColorsDark();

      ImGui_ImplGlfw_InitForOpenGL(mWindow, true);
      ImGui_ImplOpenGL3_Init("#version 430");
    }

    virtual ~Application() {
      ImGui_ImplOpenGL3_Shutdown();
      ImGui_ImplGlfw_Shutdown();

      ImGui::DestroyContext();

      glfwDestroyWindow(mWindow);
      glfwTerminate();
    }

    GLFWwindow &Window() const { return *mWindow; };

    unsigned int Width() const { return mWidth; }
    unsigned int Height() const { return mHeight; }

    bool Running() const { return !glfwWindowShouldClose(mWindow); }

  private:
    unsigned int mWidth;
    unsigned int mHeight;

    GLFWwindow *mWindow;
  };
}

#endif