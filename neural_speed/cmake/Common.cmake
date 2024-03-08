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

function(warning_check TARGET)
    # TODO(hengyu): add warning check
    if (MSVC)
    #     target_compile_definitions(${TARGET} PUBLIC -DPLATFORM_WINDOWS -DNOGDI -DNOMINMAX -D_USE_MATH_DEFINES -D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)
        target_compile_options(${TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:DPCPP>:SHELL:--compiler-options /utf-8>" "$<$<NOT:$<COMPILE_LANGUAGE:DPCPP>>:/utf-8>")
    #     target_compile_options(${TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:DPCPP>:SHELL:--compiler-options /sdl>" "$<$<NOT:$<COMPILE_LANGUAGE:DPCPP>>:/sdl>")

        # Use public to affect pybind targets
        target_compile_options(${TARGET} PUBLIC /wd4244 /wd4267)  # possible loss of data
        target_compile_options(${TARGET} PUBLIC /wd4305)  # truncation from 'double' to 'float'
        target_compile_options(${TARGET} PUBLIC /wd4018)  # '>': signed/unsigned mismatch
        target_compile_options(${TARGET} PUBLIC /wd4334)  # '<<': result of 32-bit shift implicitly converted to 64 bits

        # 'std::codecvt_utf8<wchar_t,1114111,(std::codecvt_mode)0>': warning STL4017: std::wbuffer_convert,
        # std::wstring_convert, and the <codecvt> header (containing std::codecvt_mode, std::codecvt_utf8,
        # std::codecvt_utf16, and std::codecvt_utf8_utf16) are deprecated in C++17. (The std::codecvt class template is NOT
        # deprecated.) The C++ Standard doesn't provide equivalent non-deprecated functionality; consider using
        # MultiByteToWideChar() and WideCharToMultiByte() from <Windows.h> instead. You can define
        # _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING or _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS to suppress this
        # warning.
        target_compile_definitions(${TARGET} PUBLIC _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING)

        # Microsoft renamed some POSIX and Microsoft-specific library functions in the CRT to conform with C99 and C++03
        # constraints on reserved and global implementation-defined names. If you need to use the existing function names
        # for portability reasons, you can turn off these warnings. The functions are still available in the library under
        # their original names.
        target_compile_definitions(${TARGET} PUBLIC _CRT_NONSTDC_NO_WARNINGS)
    else()
    #     # Enable warning
    #     target_compile_options(${TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:DPCPP>:SHELL:--compiler-options -Wall>" "$<$<NOT:$<COMPILE_LANGUAGE:DPCPP>>:-Wall>")
    #     target_compile_options(${TARGET} PRIVATE "$<$<NOT:$<COMPILE_LANGUAGE:DPCPP>>:-Wextra>")
    #     if(NOT CMAKE_BUILD_TYPE MATCHES "[Dd]ebug")
    #         target_compile_options(${TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:DPCPP>:SHELL:--compiler-options -Werror>" "$<$<NOT:$<COMPILE_LANGUAGE:DPCPP>>:-Werror>")
    #         target_compile_options(${TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:DPCPP>:SHELL:--compiler-options -Wno-error=deprecated-declarations>" "$<$<NOT:$<COMPILE_LANGUAGE:DPCPP>>:-Wno-error=deprecated-declarations>")
    #     endif()
    endif()
endfunction()

function(add_executable_w_warning TARGET)
    add_executable(${TARGET} ${ARGN})
    if(NS_USE_OMP)
      target_link_libraries(${TARGET} PUBLIC OpenMP::OpenMP_CXX OpenMP::OpenMP_C)
    endif()
    set_target_properties(${TARGET} PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
    set_target_properties(${TARGET} PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
    warning_check(${TARGET})
endfunction()

function(add_library_w_warning_ TARGET)
    add_library(${TARGET} ${ARGN})
    if(NS_USE_OMP)
      target_link_libraries(${TARGET} PUBLIC OpenMP::OpenMP_CXX OpenMP::OpenMP_C)
    endif()
    set_target_properties(${TARGET} PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
    set_target_properties(${TARGET} PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
    warning_check(${TARGET})
endfunction()

function(add_library_w_warning TARGET)
    add_library_w_warning_(${TARGET} STATIC ${ARGN})
endfunction()

function(add_shared_library_w_warning TARGET)
    add_library_w_warning_(${TARGET} SHARED ${ARGN})
endfunction()

function(add_shareable_library_w_warning TARGET)
    if (BUILD_SHARED_LIBS)
        add_library_w_warning_(${TARGET} SHARED ${ARGN})
    else()
        add_library_w_warning_(${TARGET} STATIC ${ARGN})
    endif()
endfunction()
