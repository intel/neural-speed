cmake_minimum_required(VERSION 2.8.12)
set (CMAKE_CXX_STANDARD 20)
project(tuner)

link_directories(${PROJECT_SOURCE_DIR}/resource_file/yaml_parser/)

add_subdirectory(selector)
add_subdirectory(explorer)
add_subdirectory(codegen)
add_subdirectory(runner)
add_subdirectory(tests)
add_subdirectory(util)

include_directories("${PROJECT_SOURCE_DIR}/resource_file/yaml_parser/include")
include_directories("${PROJECT_SOURCE_DIR}/selector")
include_directories("${PROJECT_SOURCE_DIR}/explorer")
include_directories("${PROJECT_SOURCE_DIR}/codegen")
include_directories("${PROJECT_SOURCE_DIR}/runner")
include_directories("${PROJECT_SOURCE_DIR}/util")
include_directories("${PROJECT_SOURCE_DIR}/util/include")

add_executable(${PROJECT_NAME} main.cpp tuner.cpp)

target_link_libraries(${PROJECT_NAME} Selector Explorer Codegen Runner Util yaml-cpp.a pthread)
