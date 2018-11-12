project(ubt-caffe C CXX)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS "-w -Wno-unused-result -Wno-narrowing -Wsign-compare -Wunused-variable -fPIC -O3")

include(ubt-caffe.cmake)

