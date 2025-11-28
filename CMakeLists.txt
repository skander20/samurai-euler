cmake_minimum_required(VERSION 3.16)
project(euler_tutorial)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 20)
find_package(samurai REQUIRED)

add_executable(euler_1d main_1d.cpp)
target_link_libraries(euler_1d PRIVATE samurai::samurai)

add_executable(euler_user_pred_1d main_user_pred_1d.cpp)
target_link_libraries(euler_user_pred_1d PRIVATE samurai::samurai)

add_executable(euler_2d main_2d.cpp)
target_link_libraries(euler_2d PRIVATE samurai::samurai)