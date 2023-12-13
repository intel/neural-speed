aux_source_directory(. DIR_LIB_SRCS)

include_directories(../util/include)

add_library (Runner ${DIR_LIB_SRCS})