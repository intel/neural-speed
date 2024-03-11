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

find_package(Threads REQUIRED)
file(GLOB layers_srcs "layers/*.cpp")
file(GLOB test_srcs "layers/*test*.cpp")
list(REMOVE_ITEM layers_srcs ${test_srcs})
set(sources ne_layers.c ${layers_srcs})

add_shareable_library_w_warning(ne_layers "${sources}")

target_include_directories(ne_layers PUBLIC .)
target_compile_features(ne_layers PUBLIC c_std_11) # don't bump
set_target_properties(ne_layers PROPERTIES POSITION_INDEPENDENT_CODE ON)
if (NS_TP)
  find_package(oneCCL REQUIRED)
  find_package(MPI REQUIRED)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  add_library(parallel_context STATIC parallel_context.cpp)
  target_link_libraries(ne_layers PUBLIC Threads::Threads bestla ne_vec MPI::MPI_CXX ccl parallel_context)
else ()
  target_link_libraries(ne_layers PUBLIC Threads::Threads bestla ne_vec)
endif()

if(NOT WIN32)
  target_link_libraries(ne_layers PUBLIC rt)
else()
  target_link_options(ne_layers PUBLIC /STACK:5242880)
endif()

if (NS_BUILD_TESTS)

function(add_test_target src)  # ARGN: additional source
  get_filename_component(test_target ${src} NAME_WE)
  get_filename_component(src_dir ${src} DIRECTORY)
  string(REGEX REPLACE [/\\] "_" src_dir ${src_dir})
  if(src_dir)
    set (test_target "${src_dir}_${test_target}")
  endif()
  set (test_target "test_${test_target}")
  add_executable_w_warning(${test_target} ${src} ${ARGN})
  target_compile_definitions(${test_target} PRIVATE NS_TESTS)
  target_compile_options(${test_target} PRIVATE -fsanitize=address)
  target_link_options(${test_target} PRIVATE -fsanitize=address)
  target_include_directories(${test_target} PUBLIC .)
  target_link_libraries(${test_target} PUBLIC Threads::Threads bestla ne_vec)
  if(NOT WIN32)
    target_link_libraries(${test_target} PUBLIC rt)
  endif()
  add_test(NAME ${test_target} COMMAND ${test_target})
  set_tests_properties(${test_target} PROPERTIES LABELS "${src_dir}_test")
endfunction()

add_test_target(layers/mha_dense.cpp layers/mha_dense_tests.cpp)

endif()
