set(TARGET stream_k_gemm)

#build test
add_executable(${TARGET} stream_k_gemm.cpp)
# Disable vector combine, to remove redundant loads and stores
# target_compile_options(${TARGET} PRIVATE -mllvm -disable-vector-combine -fsycl -fsycl-targets=spir64_gen)
