#  Copyright (c) 2023 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

cmake_minimum_required(VERSION 3.12) # Don't bump this version for no reason
project("ne_graph" C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

#
# Option list
#

# general
option(NS_STATIC                 "neural_speed: static link libraries"                          OFF)
option(NS_NATIVE                 "neural_speed: enable -march=native flag"                      OFF)
option(NS_LTO                    "neural_speed: enable link time optimization"                  OFF)
option(NS_BUILD_APPLICATIONS     "neural_speed: build applications"                             ON)

# GPU
option(NS_GPU                    "neural_speed: enable GPU inference"                           OFF)

# debug
option(NS_ALL_WARNINGS           "neural_speed: enable all compiler warnings"                   ON)
option(NS_ALL_WARNINGS_3RD_PARTY "neural_speed: enable all compiler warnings in 3rd party libs" OFF)
option(NS_GPROF                  "neural_speed: enable gprof"                                   OFF)

# tensor parallelism
option(NS_TP                     "neural_speed: enable tensor parallelism"                        OFF)
if (NS_TP)
  add_compile_definitions(NS_TP_MODEL)
endif()

# sanitizers
option(NS_SANITIZE_THREAD        "neural_speed: enable thread sanitizer"                        OFF)
option(NS_SANITIZE_ADDRESS       "neural_speed: enable address sanitizer"                       OFF)
option(NS_SANITIZE_UNDEFINED     "neural_speed: enable undefined sanitizer"                     OFF)

# instruction set specific
option(NS_AVX                    "neural_speed: enable AVX"                                     ON)
option(NS_AVX2                   "neural_speed: enable AVX2"                                    ON)
option(NS_F16C                   "neural_speed: enable F16C"                                    ON)
option(NS_AVX512                 "neural_speed: enable AVX512"                                  OFF)
option(NS_AVX512_VBMI            "neural_speed: enable AVX512-VBMI"                             OFF)
option(NS_AVX512_VNNI            "neural_speed: enable AVX512-VNNI"                             OFF)
option(NS_FMA                    "neural_speed: enable FMA"                                     ON)
option(NS_AMX                    "neural_speed: enable AMX"                                     OFF)
option(NS_USE_OMP                "neural_speed: use OpenMP thread pool."                        ON)

option(NS_BUILD_TESTS            "neural_speed: build tests"                       ${NS_STANDALONE})
option(NS_BUILD_EXAMPLES         "neural_speed: build examples"                    ${NS_STANDALONE})
option(NS_USE_CLANG_TIDY         "neural_speed: clang-tidy check"                               OFF)


if(NS_BUILD_TESTS)
  add_compile_definitions(NS_BUILD_TESTS)
endif()
option(NS_PROFILING              "neural_speed: use Profiling"                                  OFF)
if (NS_PROFILING)
    add_compile_definitions(NS_PERF)
endif()
option(NS_BEAM_SEARCH_VERBOSE    "neural_speed: print beam search processing log"               OFF)
if (NS_BEAM_SEARCH_VERBOSE)
    add_compile_definitions(NS_BEAM_SEARCH_VERBOSE_ON)
endif()
option(NS_GELU_VEC               "neural_speed: enable vec in gelu"                             ON)
if (NS_GELU_VEC)
    add_compile_definitions(NS_GELU_USE_VEC)
endif()
option(NS_PYTHON_API             "neural_speed: use python api"                                 OFF)
option(NS_SIMD_VEC_DOT_F16       "neural_speed: enable vec_dot_fp16 SIMD optimization"          ON)
option(BUILD_SHARED_LIBS         "If build as shared libs"                                       ON)

if (NS_SIMD_VEC_DOT_F16)
    add_compile_definitions(NS_SIMD_VEC_DOT_F16)
endif()

if(NS_BUILD_TESTS)
    enable_testing()
endif()

if (MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX)
    if (BUILD_SHARED_LIBS)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
endif()

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)  # default to false so that pybind11 will not try to use IPO
if (NS_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT result OUTPUT output)
    if (result)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(WARNING "IPO is not supported: ${output}")
    endif()
endif()

if (NOT MSVC)
    if (NS_STATIC)
        add_link_options(-static)
        if (MINGW)
            add_link_options(-static-libgcc -static-libstdc++)
        endif()
    endif()
    if (NS_GPROF)
        add_compile_options(-pg)
    endif()
    if (NS_NATIVE)
        add_compile_options(-march=native)
    endif()
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if (NS_PYTHON_API)
  add_subdirectory(third_party/pybind11)
endif()

if(NS_USE_OMP)
  include(FindOpenMP)
  # compile BesTLA's OMPTheading class, then it can be used in ne_layers
  set(BTLA_ENABLE_OPENMP ON CACHE BOOL "BesTLA enable compiling OpenMP threading")
  add_compile_definitions(NS_USE_OMP)
endif()

add_subdirectory(bestla)

add_subdirectory(neural_speed)
