cmake_minimum_required(VERSION 2.8)
project(FeatureExtraction)

find_package(OpenCV REQUIRED)

add_definitions("-std=c++11 -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -g")

add_executable(Harris main.cpp)
target_link_libraries(Harris ${OpenCV_LIBS})
install(TARGETS Harris RUNTIME DESTINATION bin)


