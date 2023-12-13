aux_source_directory(. DIR_LIB_SRCS)

#include_directories(../resource_file/yaml_parser/yaml-cpp/include)
include_directories(../util/include)
include_directories(../codegen)
include_directories(../runner)

add_library (Explorer ${DIR_LIB_SRCS})