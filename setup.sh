#!/usr/bin/env bash

vendor=$(pwd)/vendor
mkdir -p $vendor

# GLAD
git clone https://github.com/Dav1dde/glad $vendor/glad
cd $vendor/glad
python -m glad --generator=c --out-path=gl

# GLFW
git clone https://github.com/glfw/glfw $vendor/glfw
cd $vendor/glfw
cmake .

# GLM
git clone https://github.com/g-truc/glm $vendor/glm

# IMGUI
git clone https://github.com/ocornut/imgui $vendor/imgui