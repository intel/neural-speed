set(TARGET fmha)

#build test
# add_integration_test(${ProjectId} ${src})  # TODO: make it as a integration_test target
add_executable(${TARGET} fmha.cpp)
