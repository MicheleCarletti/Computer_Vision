cmake_minimum_required(VERSION 2.8)
project(Integral_Image)

find_package(OpenCV REQUIRED)

add_definitions("-std=c++11 -g")
#add_definitions("-DNATIVE_OPTS")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -g")

add_executable(integral main.cpp)
target_link_libraries(integral ${OpenCV_LIBS})
install(TARGETS integral RUNTIME DESTINATION bin)


#add_executable(test test.cpp)
#target_link_libraries(test ${OpenCV_LIBS})
#install(TARGETS test RUNTIME DESTINATION bin)
