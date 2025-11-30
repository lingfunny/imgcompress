cmake_minimum_required(VERSION 3.10)
project(ImageCompress)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(include)

add_executable(imgcompress src/main.cpp src/ImageLoader.cpp)

target_link_libraries(imgcompress ${OpenCV_LIBS})
