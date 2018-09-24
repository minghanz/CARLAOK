cmake_minimum_required(VERSION 3.3)
project(CarlaLidar)

find_package(PCL REQUIRED COMPONENTS common io visualization filters surface)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(CarlaLidar entry.cxx)
target_link_libraries(CarlaLidar ${PCL_LIBRARIES})

add_definitions(-std=c++11)
