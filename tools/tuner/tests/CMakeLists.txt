cmake_minimum_required(VERSION 2.8.12)
set (CMAKE_CXX_STANDARD 20)
project(test_tuner)


include_directories("${PROJECT_SOURCE_DIR}/../codegen")
include_directories("${PROJECT_SOURCE_DIR}/../runner")


include_directories("${PROJECT_SOURCE_DIR}/../resource_file/yaml_parser/include")
include_directories("${PROJECT_SOURCE_DIR}/../selector")
include_directories("${PROJECT_SOURCE_DIR}/../explorer")
include_directories("${PROJECT_SOURCE_DIR}/../codegen")
include_directories("${PROJECT_SOURCE_DIR}/../runner")
include_directories("${PROJECT_SOURCE_DIR}/../util/include")
include_directories("${PROJECT_SOURCE_DIR}/../")


aux_source_directory ("${PROJECT_SOURCE_DIR}/../codegen" code_gen_src_list)
aux_source_directory ("${PROJECT_SOURCE_DIR}/../runner" runner_src_list)
aux_source_directory("${PROJECT_SOURCE_DIR}/../selector" selector_src_list)
aux_source_directory("${PROJECT_SOURCE_DIR}/../explorer" explorer_src_list)
aux_source_directory("${PROJECT_SOURCE_DIR}/../util" util_src_list)

link_directories(${PROJECT_SOURCE_DIR}/../resource_file/yaml_parser/)
add_executable(${PROJECT_NAME} test_tuner.cpp ${PROJECT_SOURCE_DIR}/../tuner.cpp ${code_gen_src_list} ${runner_src_list} ${selector_src_list} ${explorer_src_list} ${util_src_list})
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/../resource_file/yaml_parser/yaml-cpp/include)
target_link_libraries(${PROJECT_NAME} yaml-cpp.a)
