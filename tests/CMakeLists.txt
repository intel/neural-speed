set(TEST_GPU_ARCH "xe_lpg" CACHE STRING "Set gpu_arch to test. Options: xe_lpg,xe_hpg,xe_hpc")
string(TOUPPER "${TEST_GPU_ARCH}" TEST_GPU_ARCH)
add_compile_definitions("TEST_GPU_ARCH_${TEST_GPU_ARCH}")

add_subdirectory(./integration)
add_subdirectory(./unit)
