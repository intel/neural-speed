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

include(cmake/ISA.cmake)
include(cmake/Common.cmake)
include(cmake/ClangTidy.cmake)

set(COMMON_HEADER_DIRS ./)
if(NS_GPU)
  list(APPEND COMMON_HEADER_DIRS ${GPU_ROOT}/include)
  list(APPEND COMMON_LIB_DIRS ${GPU_ROOT})
endif()

include_directories(${COMMON_HEADER_DIRS})
link_directories(${COMMON_LIB_DIRS})

add_subdirectory(core)
add_subdirectory(vectors)
add_subdirectory(models)

if (NS_BUILD_APPLICATIONS)
    add_subdirectory(application)
endif()
