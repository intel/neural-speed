get_filename_component(ProjectId ${CMAKE_CURRENT_SOURCE_DIR} NAME)
string(REPLACE " " "_" ProjectId ${ProjectId})
string(PREPEND ProjectId "gemm_")

FILE(GLOB src main.cpp)
add_integration_test(${ProjectId} ${src})
add_integration_test(${ProjectId}_dg2 main_dg2.cpp)
