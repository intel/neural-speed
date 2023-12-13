aux_source_directory(. DIR_LIB_SRCS)

#include_directories(../resource_file/yaml_parser/yaml-cpp/include)
include_directories(../util/include)

add_library (Codegen ${DIR_LIB_SRCS})