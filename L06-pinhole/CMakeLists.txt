cmake_minimum_required(VERSION 2.8)
project(pinhole_camera)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(pinhole_camera main.cpp)

target_link_libraries( pinhole_camera ${OpenCV_LIBS} )
install(TARGETS pinhole_camera RUNTIME DESTINATION bin)
