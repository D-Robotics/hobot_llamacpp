# Copyright (c) 2025，D-Robotics.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.5)
project(hobot_llamacpp)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Release)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(ai_msgs REQUIRED)
find_package(dnn_node REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV 4 REQUIRED)

option(SHARED_MEM "using shared mem" ON)
if (${SHARED_MEM})
  message("using shared mem")
  # rclcpp definition
  find_package(hbm_img_msgs REQUIRED)
  add_definitions(-DSHARED_MEM_ENABLED)
endif ()

# x5
set(PREFIX_PATH x5)
set(SYS_ROOT ${CMAKE_SYSROOT})

if(PLATFORM_X5)
  message("build platform X5")
  add_definitions(-DPLATFORM_X5)
  set(PREFIX_PATH x5)
else()
  if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
    message("invalid platform, build platform X5 default")
    add_definitions(-PLATFORM_X5)
    set(PREFIX_PATH x5)
  endif()
endif()

message("PREFIX_PATH is " ${PREFIX_PATH})
message("SYS_ROOT is " ${SYS_ROOT})

include_directories(include
  ${PROJECT_SOURCE_DIR}
)

include_directories(
  ${SYS_ROOT}/usr/include/opencv/
  ${SYS_ROOT}/usr/include/
  ${SYS_ROOT}/usr/hobot/include/
  ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/
  ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/common/
  ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include/
  ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/ggml/include/
  # ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/spm-headers/
  ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/examples/llava/
)

link_directories(
  ${SYS_ROOT}/usr/lib/hbbpu/
  ${SYS_ROOT}/usr/lib/hbmedia/
  ${SYS_ROOT}/usr/lib/
  ${SYS_ROOT}/usr/hobot/lib/
  ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/build/bin
  ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/build/common
)

add_subdirectory(llama.cpp)

add_executable(${PROJECT_NAME}
  src/main.cpp
  src/image_utils.cpp
  src/cli.cpp
  src/llamacpp_node.cpp
  src/post_process/llama_cpp_parser.cpp
)

target_link_libraries(${PROJECT_NAME} 
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/build/common/libcommon.a 
    ggml 
    llama
    ${CMAKE_THREAD_LIBS_INIT})

ament_target_dependencies(
  ${PROJECT_NAME}
  rclcpp
  dnn_node
  sensor_msgs
  ai_msgs
  cv_bridge
)

if (${SHARED_MEM})
  ament_target_dependencies(
    ${PROJECT_NAME}
    hbm_img_msgs
  )
endif ()

# Install executables
install(
  TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  ${PROJECT_SOURCE_DIR}/config/
  DESTINATION lib/${PROJECT_NAME}/config)

install(DIRECTORY
  ${PROJECT_SOURCE_DIR}/launch/
  DESTINATION share/${PROJECT_NAME}/launch)

ament_package()
