cmake_minimum_required(VERSION 3.13)

project(nova)

enable_language(CUDA)

# SETTINGS
set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(SOURCE_DIR ${ROOT_DIR}/source)
set(VENDOR_DIR ${ROOT_DIR}/vendor)

# GLM
set(GLM_DIR ${VENDOR_DIR}/glm)

include_directories(${GLM_DIR})

# IMGUI
set(IMGUI_DIR ${VENDOR_DIR}/imgui)

file(GLOB IMGUI_HEADER
        ${IMGUI_DIR}/*.h
        ${IMGUI_DIR}/examples/imgui_impl_glfw.h
        ${IMGUI_DIR}/examples/imgui_impl_opengl3.h)

file(GLOB IMGUI_SOURCE
        ${IMGUI_DIR}/*.cpp
        ${IMGUI_DIR}/examples/imgui_impl_glfw.cpp
        ${IMGUI_DIR}/examples/imgui_impl_opengl3.cpp)

include_directories(${IMGUI_DIR} ${IMGUI_DIR}/examples)

# TRACE
include_directories(${SOURCE_DIR})

file(GLOB NOVA_SOURCES ${SOURCE_DIR}/*.*)

add_executable(${PROJECT_NAME} ${ROOT_DIR}/main.cpp
        ${NOVA_SOURCES}
        ${IMGUI_HEADER}
        ${IMGUI_SOURCE})

target_compile_definitions(${PROJECT_NAME} PRIVATE IMGUI_IMPL_OPENGL_LOADER_GLAD)

# GLAD
set(GLAD_DIR ${VENDOR_DIR}/glad)

add_library(glad ${GLAD_DIR}/gl/src/glad.c)

target_include_directories(glad PRIVATE ${GLAD_DIR}/gl/include)
target_include_directories(${PROJECT_NAME} PUBLIC ${GLAD_DIR}/gl/include)

target_link_libraries(${PROJECT_NAME} glad ${CMAKE_DL_LIBS})

# GLFW
set(GLFW_DIR ${VENDOR_DIR}/glfw)

set(GLFW_BUILD_EXAMPLES OFF CACHE INTERNAL "Build the GLFW examples")
set(GLFW_BUILD_TESTS OFF CACHE INTERNAL "Build the GLFW tests")
set(GLFW_BUILD_DOCS OFF CACHE INTERNAL "Build the GLFW documentation")
set(GLFW_INSTALL OFF CACHE INTERNAL "Generate GLFW installation target")

add_subdirectory(${GLFW_DIR})

target_include_directories(${PROJECT_NAME} PUBLIC ${GLFW_DIR}/include)

target_link_libraries(${PROJECT_NAME} glfw ${GLFW_LIBRARIES})