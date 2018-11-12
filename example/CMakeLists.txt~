project(MiniCaffe-Examples)
cmake_minimum_required(VERSION 2.8)

set(OpenCV_DIR /usr/local/opencv341/share/OpenCV)
find_package(OpenCV REQUIRED)
include(../mini-caffe.cmake)

add_executable(ssd ssd/main.cpp)
target_link_libraries(ssd caffe ${OpenCV_LIBS})
