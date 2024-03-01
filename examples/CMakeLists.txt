include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR})

# Creates a separate device code module for each SYCL* kernel
# so that kernel for Dg2 and Xe will be JIT separately
add_compile_options(-fsycl-device-code-split=per_kernel)
add_link_options(-fsycl-device-code-split=per_kernel)

add_subdirectory(01_gemm_universal)
add_subdirectory(02_basic_gemm)
add_subdirectory(03_gemm_relu_bias)
add_subdirectory(04_gemm_polynomial)
add_subdirectory(05_batch_gemm)
add_subdirectory(06_gemm_softmax)
add_subdirectory(07_multi_layer_perceptron)
add_subdirectory(08_scaled_dot_product_attention)
add_subdirectory(09_gate_recurrent_unit)
add_subdirectory(10_gemm_large_n)
if(UNIX)  # pvc not available on win?
add_subdirectory(11_stream_k_gemm)
endif()
