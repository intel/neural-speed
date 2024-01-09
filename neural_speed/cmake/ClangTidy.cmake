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

if (NS_USE_CLANG_TIDY MATCHES "(CHECK|FIX)" AND ${CMAKE_VERSION} VERSION_LESS "3.6.0")
    message(FATAL_ERROR "Using clang-tidy requires CMake 3.6.0 or newer")
elseif(NS_USE_CLANG_TIDY MATCHES "(CHECK|FIX)")
    find_program(CLANG_TIDY NAMES clang-tidy)
    if(NOT CLANG_TIDY)
        message(FATAL_ERROR "Clang-tidy not found")
    else()
        add_compile_definitions(CLANGTIDY)
        if(NS_USE_CLANG_TIDY STREQUAL "CHECK")
            set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY})
            message(STATUS "Using clang-tidy to run checks")
        elseif(NS_USE_CLANG_TIDY STREQUAL "FIX")
            set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY} -fix)
            message(STATUS "Using clang-tidy to run checks and fix found issues")
        endif()
    endif()
endif()
