# Install script for directory: /home/wangzhe/neural-speed/bestla

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/home/wangzhe/.local/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib64/cmake/jblas" TYPE FILE FILES
    "/home/wangzhe/neural-speed/bestla/build/jblas-config.cmake"
    "/home/wangzhe/neural-speed/bestla/build/jblas-config-version.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib64/cmake/jblas/jblas-targets.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib64/cmake/jblas/jblas-targets.cmake"
         "/home/wangzhe/neural-speed/bestla/build/CMakeFiles/Export/0e3c3720227cc946cfd005ef603d3bbd/jblas-targets.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib64/cmake/jblas/jblas-targets-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib64/cmake/jblas/jblas-targets.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib64/cmake/jblas" TYPE FILE FILES "/home/wangzhe/neural-speed/bestla/build/CMakeFiles/Export/0e3c3720227cc946cfd005ef603d3bbd/jblas-targets.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/jblas" TYPE FILE FILES
    "/home/wangzhe/neural-speed/bestla/jblas/jit_base.h"
    "/home/wangzhe/neural-speed/bestla/jblas/jit_blas.h"
    "/home/wangzhe/neural-speed/bestla/jblas/jit_blas_device.h"
    "/home/wangzhe/neural-speed/bestla/jblas/jit_blas_epilogue.h"
    "/home/wangzhe/neural-speed/bestla/jblas/jit_blas_gemm.h"
    "/home/wangzhe/neural-speed/bestla/jblas/jit_blas_parallel.h"
    "/home/wangzhe/neural-speed/bestla/jblas/jit_blas_prologue_a.h"
    "/home/wangzhe/neural-speed/bestla/jblas/jit_blas_prologue_b.h"
    "/home/wangzhe/neural-speed/bestla/jblas/jit_blas_storage.h"
    "/home/wangzhe/neural-speed/bestla/jblas/jit_blas_utils.h"
    "/home/wangzhe/neural-speed/bestla/jblas/jit_blas_wrapper.h"
    "/home/wangzhe/neural-speed/bestla/jblas/kernel_avx2.h"
    "/home/wangzhe/neural-speed/bestla/jblas/kernel_avx512_bf16.h"
    "/home/wangzhe/neural-speed/bestla/jblas/kernel_avx512f.h"
    "/home/wangzhe/neural-speed/bestla/jblas/kernel_jit.h"
    "/home/wangzhe/neural-speed/bestla/jblas/kernel_jit_injector.h"
    "/home/wangzhe/neural-speed/bestla/jblas/kernel_ref.h"
    "/home/wangzhe/neural-speed/bestla/jblas/kernel_wrapper.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/jblas/xbyak" TYPE FILE FILES
    "/home/wangzhe/neural-speed/bestla/jblas/xbyak/xbyak.h"
    "/home/wangzhe/neural-speed/bestla/jblas/xbyak/xbyak_bin2hex.h"
    "/home/wangzhe/neural-speed/bestla/jblas/xbyak/xbyak_mnemonic.h"
    "/home/wangzhe/neural-speed/bestla/jblas/xbyak/xbyak_util.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/wangzhe/neural-speed/bestla/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
