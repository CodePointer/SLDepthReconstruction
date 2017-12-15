cmake_minimum_required(VERSION 3.9)
project(SLDepthReconstruction)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

include_directories(
        global
        ${OPENCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRS}
)

add_executable(SLDepthReconstruction
        main.cpp
        reconstructor.cpp
        reconstructor.h
        global/static_para.h global/global_fun.h vertex_set.cpp vertex_set.h)

target_link_libraries(SLDepthReconstruction
        ceres
        ${OpenCV_LIBS})