cmake_minimum_required(VERSION 3.10)
project(imu_to_velodyne)

find_package(Eigen3 REQUIRED)

add_executable(imu_to_velodyne main.cpp)
target_link_libraries(imu_to_velodyne Eigen3::Eigen)

