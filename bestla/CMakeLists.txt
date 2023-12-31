cmake_minimum_required(VERSION 3.12)

project(bestla LANGUAGES CXX VERSION 0.1.0)
file(GLOB headers ${PROJECT_NAME}/*.h ${PROJECT_NAME}/*.hpp)
file(GLOB xbyak_headers ${PROJECT_NAME}/xbyak/*.h ${PROJECT_NAME}/xbyak/*.hpp)

option(BTLA_UT_ALL "Enable all unit tests" OFF)
option(BTLA_UT_DEBUG "Enable debug unit tests" OFF)
option(BTLA_UT_EPILOGUE "Enable unit test for epilogue" OFF)
option(BTLA_UT_PROLOGUE_A "Enable unit test for activation prologue" OFF)
option(BTLA_UT_PROLOGUE_B "Enable unit test for weight prologue" OFF)
option(BTLA_UT_GEMM "Enable unit test for micro gemm kernels" OFF)
option(BTLA_UT_WRAPPER "Enable unit test for parallel gemms" OFF)
option(BTLA_UT_PARALLEL "Enable unit test for parallel set" OFF)
option(BTLA_UT_KERNEL_JIT "Enable unit test for jit kernels" OFF)
option(BTLA_UT_KERNEL_INTRIN "Enable unit test for intrinsic kernels" OFF)
option(BTLA_UT_KERNEL_WRAPPER "Enable unit test for runtime ISA kernels" OFF)
option(BTLA_UT_NOASAN "Disable sanitize" OFF)
option(BTLA_UT_BENCHMARK "Benchmark ON may take a long time to finish all tests" OFF)
option(BTLA_UT_OPENMP "Use OpenMP" ON)

if(BTLA_UT_ALL)
set(BTLA_UT_EPILOGUE ON)
set(BTLA_UT_PROLOGUE_A ON)
set(BTLA_UT_PROLOGUE_B ON)
set(BTLA_UT_GEMM ON)
set(BTLA_UT_WRAPPER ON)
set(BTLA_UT_PARALLEL ON)
set(BTLA_UT_KERNEL_JIT ON)
set(BTLA_UT_KERNEL_INTRIN ON)
set(BTLA_UT_KERNEL_WRAPPER ON)
endif(BTLA_UT_ALL)

set(UT_BUILD FALSE)
if(BTLA_UT_DEBUG OR BTLA_UT_PROLOGUE_A OR BTLA_UT_PROLOGUE_B OR BTLA_UT_EPILOGUE OR BTLA_UT_GEMM
OR BTLA_UT_WRAPPER OR BTLA_UT_PARALLEL OR BTLA_UT_KERNEL_JIT OR BTLA_UT_KERNEL_INTRIN
OR BTLA_UT_KERNEL_WRAPPER)
set(UT_BUILD TRUE)
endif()

include(GNUInstallDirs)
add_library(${PROJECT_NAME} INTERFACE)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
	
target_include_directories(
	${PROJECT_NAME} INTERFACE
	"$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
	"$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

install(
	TARGETS ${PROJECT_NAME}
	EXPORT ${PROJECT_NAME}-targets
	INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
	cmake/config.cmake.in
	"${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake"
	INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)
write_basic_package_version_file(
	"${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake"
	COMPATIBILITY SameMajorVersion
)

install(
	FILES
		"${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake"
		"${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake"
	DESTINATION
		${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

install(
	EXPORT ${PROJECT_NAME}-targets
	NAMESPACE ${PROJECT_NAME}::
	DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

if(WIN32)
	target_compile_definitions(${PROJECT_NAME} INTERFACE _CRT_SECURE_NO_WARNINGS NOMINMAX)
	target_compile_options(${PROJECT_NAME} INTERFACE /wd4068 /wd4849 /wd6262 /wd4702 /wd4100) 
	#4068 ignore unroll and GCC flags
	#4849 ignore collapse
	#6262 ignore stack too large
	#4702 unreachable code(false warning on constexpr condition)
	#4100 unreferenced formal parameter

	target_link_options(${PROJECT_NAME} INTERFACE /STACK:5242880) #Stack requires up to L2 cache size
endif(WIN32)



set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

function(add_ut_flag UT_OPTION)
	if(${${UT_OPTION}})
	  target_compile_definitions(${PROJECT_NAME}_ut PRIVATE ${UT_OPTION})
	endif()
endfunction()

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)
if(UT_BUILD)
	file(GLOB srcs ${PROJECT_NAME}/ut/*.cc ${PROJECT_NAME}/ut/*.cpp) #compile everthing even run parts of UTs
	file(GLOB ut_headers ${PROJECT_NAME}/ut/*.h)
  include_directories(${PROJECT_NAME})
	add_executable(${PROJECT_NAME}_ut ${srcs} ${headers} ${ut_headers})
  if(BTLA_UT_OPENMP)
    include(FindOpenMP)
    target_link_libraries(${PROJECT_NAME}_ut PRIVATE OpenMP::OpenMP_CXX)
  endif()
	if(NOT WIN32)
		if(NOT BTLA_UT_NOASAN)
		  target_compile_options(${PROJECT_NAME}_ut PRIVATE -fsanitize=address)
		  target_link_options(${PROJECT_NAME}_ut PRIVATE -fsanitize=address)
		endif()
		target_link_options(${PROJECT_NAME}_ut PRIVATE -lpthread)
	endif()

	add_ut_flag(BTLA_UT_DEBUG)
	add_ut_flag(BTLA_UT_EPILOGUE)
	add_ut_flag(BTLA_UT_PROLOGUE_A)
	add_ut_flag(BTLA_UT_PROLOGUE_B)
	add_ut_flag(BTLA_UT_GEMM)
	add_ut_flag(BTLA_UT_PARALLEL)
	add_ut_flag(BTLA_UT_WRAPPER)
	add_ut_flag(BTLA_UT_KERNEL_INTRIN)
	add_ut_flag(BTLA_UT_KERNEL_JIT)
	add_ut_flag(BTLA_UT_KERNEL_WRAPPER)
	add_ut_flag(BTLA_UT_BENCHMARK)

	target_link_libraries(${PROJECT_NAME}_ut PRIVATE ${PROJECT_NAME})
endif(UT_BUILD)

install(
	FILES ${headers}
	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
)
install(
	FILES ${xbyak_headers}
	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/xbyak/
)
