cmake_minimum_required(VERSION 3.5)

project(jblas LANGUAGES CXX VERSION 0.1.0)
file(GLOB headers ${PROJECT_NAME}/*.h ${PROJECT_NAME}/*.hpp)
file(GLOB xbyak_headers ${PROJECT_NAME}/xbyak/*.h ${PROJECT_NAME}/xbyak/*.hpp)

option(JBLAS_UT_ALL "Enable all unit tests" OFF)
option(JBLAS_UT_DEBUG "Enable debug unit tests" ON)
option(JBLAS_UT_WEIGHT_COMPRESSION "Enable unit test for weight compression" OFF)
option(JBLAS_UT_TRANSFORMER "Enable unit test for transformer fusion" OFF)
option(JBLAS_UT_EPILOGUE "Enable unit test for epilogue" OFF)
option(JBLAS_UT_PROLOGUE "Enable unit test for prologue" OFF)
option(JBLAS_UT_GEMM "Enable unit test for micro gemm kernels" OFF)
option(JBLAS_UT_WRAPPER "Enable unit test for parallel gemms" OFF)
option(JBLAS_UT_UTILS "Enable unit test for utils" OFF)
option(JBLAS_UT_KERNEL_JIT "Enable unit test for jit kernels" OFF)
option(JBLAS_UT_KERNEL_INTRIN "Enable unit test for intrinsic kernels" OFF)
option(JBLAS_UT_KERNEL_WRAPPER "Enable unit test for runtime ISA kernels" OFF)
option(JBLAS_UT_NOASAN "Disable sanitize" OFF)

if(JBLAS_UT_ALL)
set(JBLAS_UT_WEIGHT_COMPRESSION ON)
set(JBLAS_UT_TRANSFORMER ON)
set(JBLAS_UT_EPILOGUE ON)
set(JBLAS_UT_PROLOGUE ON)
set(JBLAS_UT_GEMM ON)
set(JBLAS_UT_WRAPPER ON)
set(JBLAS_UT_UTILS ON)
set(JBLAS_UT_KERNEL_JIT ON)
set(JBLAS_UT_KERNEL_INTRIN ON)
set(JBLAS_UT_KERNEL_WRAPPER ON)
endif(JBLAS_UT_ALL)

set(UT_BUILD FALSE)
if(JBLAS_UT_DEBUG OR JBLAS_UT_WEIGHT_COMPRESSION OR JBLAS_UT_TRANSFORMER OR JBLAS_UT_EPILOGUE OR JBLAS_UT_GEMM
OR JBLAS_UT_PROLOGUE OR JBLAS_UT_WRAPPER OR JBLAS_UT_UTILS OR JBLAS_UT_KERNEL_JIT OR JBLAS_UT_KERNEL_INTRIN
OR JBLAS_UT_KERNEL_WRAPPER)
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

	target_link_options(${PROJECT_NAME} INTERFACE /STACK:3145728) #Stack requires up to L2 cache size
endif(WIN32)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

function(add_ut_flag UT_OPTION)
	if(${${UT_OPTION}})
	  target_compile_definitions(${PROJECT_NAME}_ut PRIVATE ${UT_OPTION})
	endif()
endfunction()

include(FindOpenMP)
target_link_libraries(${PROJECT_NAME} INTERFACE OpenMP::OpenMP_CXX)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)
if(UT_BUILD)
file(GLOB srcs ${PROJECT_NAME}/ut/*.cc ${PROJECT_NAME}/ut/*.cpp) #compile everthing even run parts of UTs
file(GLOB ut_headers ${PROJECT_NAME}/ut/*.h)
add_executable(${PROJECT_NAME}_ut ${srcs} ${headers} ${ut_headers})

if(NOT WIN32)
	if(NOT JBLAS_UT_NOASAN)
	target_compile_options(${PROJECT_NAME}_ut PRIVATE -fsanitize=address)
	target_link_options(${PROJECT_NAME}_ut PRIVATE -fsanitize=address)
	endif()
endif()

add_ut_flag(JBLAS_UT_DEBUG)
add_ut_flag(JBLAS_UT_WEIGHT_COMPRESSION)
add_ut_flag(JBLAS_UT_TRANSFORMER)
add_ut_flag(JBLAS_UT_EPILOGUE)
add_ut_flag(JBLAS_UT_PROLOGUE)
add_ut_flag(JBLAS_UT_GEMM)
add_ut_flag(JBLAS_UT_UTILS)
add_ut_flag(JBLAS_UT_WRAPPER)
add_ut_flag(JBLAS_UT_KERNEL_INTRIN)
add_ut_flag(JBLAS_UT_KERNEL_JIT)
add_ut_flag(JBLAS_UT_KERNEL_WRAPPER)

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
