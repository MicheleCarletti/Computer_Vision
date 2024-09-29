cmake_minimum_required(VERSION 2.8)
project(Histogram)

find_package(OpenCV REQUIRED)

add_definitions("-std=c++11 -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -g")

add_executable(Histogram main.cpp)
target_link_libraries(Histogram ${OpenCV_LIBS})
install(TARGETS Histogram RUNTIME DESTINATION bin)


