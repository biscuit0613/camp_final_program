cmake_minimum_required(VERSION 3.8)
project(ball_coord_sub)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

add_executable(subscriber src/subscriber.cpp ../kmfilter/KF.cpp)
find_package(nlohmann_json REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${NLOHMANN_JSON_INCLUDE_DIRS})
ament_target_dependencies(subscriber rclcpp geometry_msgs std_msgs nlohmann_json OpenCV)
install(TARGETS subscriber DESTINATION lib/${PROJECT_NAME})

ament_package()
target_link_libraries(subscriber Eigen3::Eigen
)