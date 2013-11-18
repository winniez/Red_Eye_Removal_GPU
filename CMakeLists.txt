cmake_minimum_required(VERSION 2.8)
set( PROJECT_NAME "red_eye_removal" )
project( ${PROJECT_NAME} C CXX )
set( OpenCL_LIBS "/usr/lib64/OpenCL/vendors/nvidia/libOpenCL.so" )
set( OpenCL_INC "/usr/include" )
set( CMAKE_C_FLAGS "-Wall -Wno-comment -Os -std=gnu99 -Wno-unused-result
-Wno-write-strings" )
include_directories( "${OpenCL_INC}" )
find_package( OpenCV REQUIRED )
add_executable( ${PROJECT_NAME} ${PROJECT_NAME} )
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} ${CUDA_LIBS}
${OpenCL_LIBS} )
