cmake_minimum_required(VERSION 2.8)
project(sobel)

find_package(OpenCV REQUIRED)

add_definitions("-std=c++11 -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -g")

add_executable(mm mm.cpp)
target_link_libraries(mm ${OpenCV_LIBS})
install(TARGETS mm RUNTIME DESTINATION bin)


