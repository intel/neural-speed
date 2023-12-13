cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 20)

project(XeTLA_tune)

set(XETLA_DIR $ENV{XETLA_SRC_DIR})
include_directories(${XETLA_DIR}/include)
include_directories(${XETLA_DIR}/tests)
include_directories(${XETLA_DIR})

set(XETLA_INCLUDE_DIR ${XETLA_DIR}/include)
include_directories($ENV{DPCPP_ROOT}/include/sycl;$ENV{DPCPP_ROOT}/include/)

set(TARGET tune_kernel)


set(USE_BLAS "openblas" CACHE STRING "Specified vendor of BLAS library")
set(NEW_XMAIN "off" CACHE STRING "set to use new xmain")
set(EMBARGO "off" CACHE STRING "set to use Rialto features")
if (${USE_BLAS} STREQUAL "openblas")
    add_definitions(-DUSE_OPENBLAS)
    set(BLA_VENDOR OpenBLAS)
    #set(LIB_BLAS $<IF:$<PLATFORM_ID:Windows>,-llibopenblas,-lblas>)
elseif (${USE_BLAS} STREQUAL "mkl")
    set(BLA_VENDOR Intel10_64lp)
endif ()

add_link_options(-fsycl -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm)

# build test
add_executable(${TARGET} main.cpp)

target_link_libraries(${TARGET} PUBLIC OpenCL)
target_link_libraries(${TARGET} PUBLIC gtest gtest_main)
target_link_options(${TARGET} PUBLIC -fsycl ${LIB_BLAS})

target_compile_options(${TARGET} PUBLIC -fsycl)
target_compile_definitions(${TARGET} PRIVATE XETLA_INCLUDE_DIR=\"${XETLA_INCLUDE_DIR}\")
