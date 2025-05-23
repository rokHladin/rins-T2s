cmake_minimum_required(VERSION 3.5)
project(dis_tutorial7)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)

find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)

install(DIRECTORY
  launch
  meshes
  urdf
  config
  DESTINATION share/${PROJECT_NAME}/
)

install(PROGRAMS
  scripts/arm_mover_actions.py
  scripts/bridge_navigation.py
  DESTINATION lib/${PROJECT_NAME})

ament_package()
