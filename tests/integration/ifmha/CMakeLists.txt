set(TARGET ifmha)

#build test
# add_integration_test(${ProjectId} ${src})  # TODO: make it as a integration_test target
add_executable(${TARGET} ifmha_forward.cpp main.cpp)
