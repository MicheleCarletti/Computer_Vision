cmake_minimum_required(VERSION 2.8)
project(Hough_Transform)

find_package(OpenCV REQUIRED)

add_definitions("-std=c++11 -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -g")

add_executable(Hough main.cpp)
target_link_libraries(Hough ${OpenCV_LIBS})
install(TARGETS Hough RUNTIME DESTINATION bin)


