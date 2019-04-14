cmake_minimum_required(VERSION 3.5.0)
set(CMAKE_CXX_STANDARD 11)

project(carla_detector)

find_package(PythonExtensions REQUIRED)
find_package(PythonInterp REQUIRED)
find_package(Cython REQUIRED)
find_package(PCL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

execute_process(COMMAND "${PYTHON_EXECUTABLE}"
    -c "import pcl; print(pcl.get_include())"
    OUTPUT_VARIABLE _pcl_include_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
message(STATUS ${_pcl_include_dir})
include_directories(${_pcl_include_dir})

# Add at last to make sure using the right python library
include_directories(BEFORE ${PYTHON_INCLUDE_DIR})

# support cython/c api
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_LIST_DIR})

add_cython_target(detector CXX)
add_library(detector MODULE ${detector} detector_src.cxx)
target_link_libraries(detector ${PCL_LIBRARIES})
python_extension_module(detector)
install(TARGETS detector LIBRARY DESTINATION detector)

