cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
if(UNIX)
else() # Windows
    # Force CMake to use icx-cl rather than the default C++ compiler/linker
    # (needed on Windows only)
    # include (CMakeForceCompiler)
    # CMAKE_FORCE_CXX_COMPILER (icx-cl IntelDPCPP)
    set(CMAKE_CXX_COMPILER icx-cl)
    include (Platform/Windows-Clang)
    include(cmake/GTestExternal.cmake)
endif()

project(XeTLA)

include(CTest)
enable_testing()

if(UNIX)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/tools/cmake")
endif()
find_package(MKL CONFIG REQUIRED)
message(STATUS "MKL_VERSION=${MKL_VERSION}")
message(STATUS "MKL_IMPORTED_TARGETS=${MKL_IMPORTED_TARGETS}")

# debug option
message(STATUS "'DEBUG' is set to " ${DEBUG})
if (${DEBUG})
add_compile_options(-debug=minimal -Rno-debug-disables-optimization -DDEBUG=${DEBUG})
endif ()

# log message print
set(LOG "off" CACHE STRING "on/off log message")
message(STATUS "log message is " ${LOG})
if (${LOG} STREQUAL "on")
    add_definitions(-DLOG_PRINT)
endif ()

add_compile_options(-fsycl)
add_link_options(-fsycl)
if(UNIX)
    add_compile_options(-fp-model=precise -Wall -Wextra -ftemplate-backtrace-limit=0)
    add_link_options(-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm)
    link_libraries(-lgtest -lgtest_main)
else() # Windows
    add_compile_options(/EHsc)
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(/MDd)
    endif()
    add_link_options(/Qmkl-ilp64=parallel /Qmkl=parallel)
    link_libraries(MKL::MKL)
    link_libraries(MKL::mkl_intel_ilp64)
    link_libraries(MKL::mkl_core)
    link_libraries(GTest GTestMain)

    # set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
    # set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
    # set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

    add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX)
    # supress warnings
    add_compile_options(/wd4244 /wd4267)  # possible loss of data
    add_compile_options(/wd4305)  # truncation from 'double' to 'float'
    add_compile_options(/wd4018)  # '>': signed/unsigned mismatch
    add_compile_options(/wd4334)  # '<<': result of 32-bit shift implicitly converted to 64 bits

    # Microsoft renamed some POSIX and Microsoft-specific library functions in the CRT to conform with C99 and C++03
    # constraints on reserved and global implementation-defined names. If you need to use the existing function names
    # for portability reasons, you can turn off these warnings. The functions are still available in the library under
    # their original names.
    add_compile_definitions(_CRT_NONSTDC_NO_WARNINGS)
endif()
add_subdirectory(tests)
add_subdirectory(examples)
