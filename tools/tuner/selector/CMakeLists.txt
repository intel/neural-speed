aux_source_directory(. DIR_LIB_SRCS)

include_directories(../resource_file/yaml_parser/include)
include_directories(../util/include)

add_library (Selector ${DIR_LIB_SRCS})