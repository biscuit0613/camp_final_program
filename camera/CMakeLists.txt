cmake_minimum_required(VERSION 3.10)

project(camera)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(nlohmann_json REQUIRED)
include_directories(${NLOHMANN_JSON_INCLUDE_DIRS})

add_executable(camera camera.cpp)
target_link_libraries( camera ${OpenCV_LIBS} )
target_link_libraries(camera ${NLOHMANN_JSON_LIBRARIES})