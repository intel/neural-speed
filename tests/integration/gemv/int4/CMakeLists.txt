get_filename_component(ProjectId ${CMAKE_CURRENT_SOURCE_DIR} NAME)
string(REPLACE " " "_" ProjectId ${ProjectId})
set(ProjectIdClient ${ProjectId})
set(ProjectIdXe ${ProjectId})
string(PREPEND ProjectIdClient "gemv_")

FILE(GLOB src main.cpp)
add_integration_test(${ProjectIdClient} ${src})

