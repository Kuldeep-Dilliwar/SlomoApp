cmake_minimum_required(VERSION 3.22.1)
project("slomoapp")

# Set up OpenCV
set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/opencv/sdk/native/jni)
find_package(OpenCV REQUIRED)

add_library(${CMAKE_PROJECT_NAME} SHARED slomo.cpp)

target_link_libraries(${CMAKE_PROJECT_NAME}
        ${OpenCV_LIBS}
        GLESv3
        EGL
        android
        log)

target_link_options(${CMAKE_PROJECT_NAME} PRIVATE "-Wl,-z,max-page-size=16384")