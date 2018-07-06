cmake_minimum_required(VERSION 3.9)
project(SLDepthReconstruction)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

include_directories(
        global
        optimizer
        DataStructure
        ${OPENCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRS}
)

add_executable(SLDepthReconstruction
        main.cpp
        reconstructor.cpp
        reconstructor.h
        global/static_para.h
        global/global_fun.h
        vertex_set.cpp
        vertex_set.h
#        optimizer/deform_cost_functor.h
#        optimizer/deform_cost_functor.cpp
#        optimizer/regular_cost_functor.h
#        optimizer/regular_cost_functor.cpp
        global/global_fun.cpp
        DataStructure/intensity_slot.cpp
        DataStructure/intensity_slot.h
        optimizer/shade_data_constraint.cpp
        optimizer/shade_data_constraint.h
        optimizer/shade_reg_constraint.cpp
        optimizer/shade_reg_constraint.h
        DataStructure/node_set.cpp
        DataStructure/node_set.h
        optimizer/depth_reg_constraint.cpp
        optimizer/depth_reg_constraint.h optimizer/depth_inter_constraint.cpp optimizer/depth_inter_constraint.h optimizer/depth_spacial_constraint.cpp optimizer/depth_spacial_constraint.h optimizer/depth_temporal_constraint.cpp optimizer/depth_temporal_constraint.h optimizer/depth_consist_constraint.cpp optimizer/depth_consist_constraint.h)

target_link_libraries(SLDepthReconstruction
        ceres
        ${OpenCV_LIBS})