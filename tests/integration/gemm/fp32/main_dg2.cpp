/*******************************************************************************
* Copyright (c) 2022-2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "xetla.hpp"
#include <utils/utils.hpp>
using namespace gpu::xetla;
//The number of times the kernel is executed
constexpr int ITER = 1;

class t1 {
public:
    //Extract the parameters required by different test cases
    static constexpr size_t mat_m = 16;
    static constexpr size_t mat_n = 32;
    static constexpr size_t mat_k = 32;
    static constexpr size_t wg_m = 16;
    static constexpr size_t wg_n = 16;
    static constexpr size_t sg_m = 16;
    static constexpr size_t sg_n = 16;
    static constexpr size_t sg_k = 16;
    static constexpr uint32_t global_kslicing = 1;
    static constexpr uint32_t local_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = float;
    using data_type_b = float;
    using data_type_c = float;
    using data_type_acc = float;
};

class t2 {
public:
    //Extract the parameters required by different test cases
    static constexpr size_t mat_m = 1024;
    static constexpr size_t mat_n = 4096;
    static constexpr size_t mat_k = 4096;
    static constexpr size_t wg_m = 64;
    static constexpr size_t wg_n = 64;
    static constexpr size_t sg_m = 16;
    static constexpr size_t sg_n = 2;
    static constexpr size_t sg_k = 16;
    static constexpr uint32_t global_kslicing = 1;
    static constexpr uint32_t local_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = float;
    using data_type_b = float;
    using data_type_c = float;
    using data_type_acc = float;
};
class t3 {
public:
    //Extract the parameters required by different test cases
    static constexpr size_t mat_m = 1024;
    static constexpr size_t mat_n = 4096;
    static constexpr size_t mat_k = 4096;
    static constexpr size_t wg_m = 64;
    static constexpr size_t wg_n = 64;
    static constexpr size_t sg_m = 16;
    static constexpr size_t sg_n = 4;
    static constexpr size_t sg_k = 16;
    static constexpr uint32_t global_kslicing = 1;
    static constexpr uint32_t local_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = float;
    using data_type_b = float;
    using data_type_c = float;
    using data_type_acc = float;
};

template <typename data_type_a, typename data_type_b, typename data_type_c,
        typename data_type_acc = float, typename data_type_bias = data_type_a>
int gemm_result_validate(data_type_a *A, data_type_b *B, data_type_c *C,
        uint32_t m, uint32_t k, uint32_t n,
        mem_layout mem_layout_a_ = mem_layout::row_major,
        mem_layout mem_layout_b_ = mem_layout::row_major) {
    buff_cmp::buff_vals<data_type_c> data(C, m, n, n);
    std::vector<data_type_acc> gold_C(m * n, 0);
    get_gemm_gold<data_type_a, data_type_b, data_type_acc>(
            m, n, k, mem_layout_a_, mem_layout_b_, A, B, gold_C.data());

    buff_cmp::buff_vals<data_type_c, data_type_acc> other(
            gold_C.data(), m, n, n);

    bool result = buff_cmp::xetla_buff_cmp(data, other, "gemm validation");

    std::cout << (!result ? "FAILED\n" : "PASSED\n");
    return result ? 0 : 1;
}

template <class Test>
void fpu_fp32_gemm_run(int iter) {
    using namespace gpu;
    // Accept incoming parameters
    constexpr size_t matrix_m = Test::mat_m;
    constexpr size_t matrix_n = Test::mat_n;
    constexpr size_t matrix_k = Test::mat_k;
    constexpr mem_layout layout_a = Test::layout_a;
    constexpr mem_layout layout_b = Test::layout_b;
    constexpr uint32_t global_kslicing = Test::global_kslicing;
    constexpr uint32_t local_kslicing = Test::local_kslicing;

    constexpr size_t wg_tile_m = Test::wg_m;
    constexpr size_t wg_tile_n = Test::wg_n;
    constexpr size_t sg_tile_m = Test::sg_m;
    constexpr size_t sg_tile_n = Test::sg_n;
    constexpr size_t sg_tile_k = Test::sg_k;
    using data_type_a = typename Test::data_type_a;
    using data_type_b = typename Test::data_type_b;
    using data_type_c = typename Test::data_type_c;
    using data_type_acc = float;

    constexpr size_t size_a = matrix_m * matrix_k;
    constexpr size_t size_b = matrix_k * matrix_n;
    constexpr size_t size_c = matrix_m * matrix_n;

    uint32_t lda = layout_a == mem_layout::row_major ? matrix_k : matrix_m;
    uint32_t ldb = layout_b == mem_layout::row_major ? matrix_n : matrix_k;
    uint32_t ldc = matrix_n;

    // Turn on the enable_profiling property to facilitate subsequent profiling
    sycl::property_list properties {sycl::property::queue::enable_profiling()};
    auto queue = sycl::queue(properties);
    auto context = queue.get_info<info::queue::context>();
    auto device = queue.get_info<info::queue::device>();

    std::cout << "Running on " << device.get_info<info::device::name>() << "\n";

    using tile_shape = xetla::group::tile_shape_t<wg_tile_n, wg_tile_m,
            sg_tile_n, sg_tile_m>;
    static constexpr uint32_t periodic_sync_interval = 0;
    static constexpr uint32_t prefetch_distance = 1;

    using mem_desc_a_t = xetla::mem_desc_t<data_type_a, layout_a,
            mem_space::global, DEVICE_MEM_ALIGNMENT / sizeof(data_type_a)>;
    using mem_desc_b_t = xetla::mem_desc_t<data_type_b, layout_b,
            mem_space::global, DEVICE_MEM_ALIGNMENT / sizeof(data_type_b)>;
    using mem_desc_c_t = xetla::mem_desc_t<data_type_c, mem_layout::row_major,
            mem_space::global, DEVICE_MEM_ALIGNMENT / sizeof(data_type_c)>;

    using compute_attr = xetla::group::compute_attr_t<data_type_acc,
            data_type_acc, data_type_acc>;
    using perf_tuning_knob = xetla::group::perf_tuning_knob_t<sg_tile_k,
            prefetch_distance, periodic_sync_interval>;

    using compute_policy
            = xetla::group::compute_policy_default_fpu<compute_attr,
                    perf_tuning_knob, gpu_arch::Dg2>;

    using gemm_t = xetla::group::gemm_t<compute_policy, tile_shape,
            mem_desc_a_t, mem_desc_b_t>;

    using epilogue_t = xetla::group::epilogue_t<
            xetla::group::epilogue_policy_default<gpu_arch::Dg2>, tile_shape,
            mem_desc_c_t>;

    using group_swizzle = xetla::kernel::group_swizzle_default<gpu_arch::Dg2>;
    using gemm_op_t = xetla::kernel::gemm_universal_t<
            gpu::xetla::kernel::dispatch_policy_kslicing<group_swizzle,
                    global_kslicing, local_kslicing>,
            gemm_t, epilogue_t>;

    size_t size_acc = gemm_op_t::get_acc_buf_size(matrix_m, matrix_n);
    size_t size_cnt = gemm_op_t::get_cnt_buf_size(matrix_m, matrix_n);

    //Define and initialize the data required for the calculation
    auto *A_h = static_cast<data_type_a *>(
            malloc_host(size_a * sizeof(data_type_a), context));
    auto *B_h = static_cast<data_type_b *>(
            malloc_host(size_b * sizeof(data_type_b), context));
    auto *C_h = static_cast<data_type_c *>(
            malloc_host(size_c * sizeof(data_type_c), context));
    auto *Acc_h = static_cast<data_type_acc *>(
            malloc_host(size_acc * sizeof(data_type_acc), context));
    auto *Cnt_h = static_cast<uint32_t *>(
            malloc_host(size_cnt * sizeof(uint32_t), context));

    auto *A_d = static_cast<data_type_a *>(
            aligned_alloc_device(DEVICE_MEM_ALIGNMENT,
                    size_a * sizeof(data_type_a), device, context));
    auto *B_d = static_cast<data_type_b *>(
            aligned_alloc_device(DEVICE_MEM_ALIGNMENT,
                    size_b * sizeof(data_type_b), device, context));
    auto *C_d = static_cast<data_type_c *>(
            aligned_alloc_device(DEVICE_MEM_ALIGNMENT,
                    size_c * sizeof(data_type_c), device, context));
    auto *Acc_d = static_cast<data_type_acc *>(
            aligned_alloc_device(DEVICE_MEM_ALIGNMENT,
                    size_acc * sizeof(data_type_acc), device, context));
    auto *Cnt_d
            = static_cast<uint32_t *>(aligned_alloc_device(DEVICE_MEM_ALIGNMENT,
                    size_cnt * sizeof(uint32_t), device, context));

    for (unsigned i = 0; i < size_a; ++i) {
        A_h[i] = random_float();
        // A_h[i] = i ;
        // A_h[i] = 1.f;
    }

    for (unsigned i = 0; i < size_b; ++i) {
        // B_h[i] = i % 16 + i / 16  * 100;
        B_h[i] = random_float();
    }

    for (unsigned i = 0; i < size_c; ++i) {
        C_h[i] = 0;
    }
    for (unsigned i = 0; i < size_acc; ++i) {
        Acc_h[i] = 0;
    }
    for (unsigned i = 0; i < size_cnt; ++i) {
        Cnt_h[i] = 0;
    }

    queue.memcpy((void *)A_d, (void *)A_h, size_a * sizeof(data_type_a)).wait();
    queue.memcpy((void *)B_d, (void *)B_h, size_b * sizeof(data_type_b)).wait();
    queue.memcpy((void *)C_d, (void *)C_h, size_c * sizeof(data_type_c)).wait();
    queue.memcpy((void *)Acc_d, (void *)Acc_h, size_acc * sizeof(data_type_acc))
            .wait();
    queue.memcpy((void *)Cnt_d, (void *)Cnt_h, size_cnt * sizeof(uint32_t))
            .wait();

    typename gemm_op_t::arguments_t gemm_arg(matrix_m, matrix_k, matrix_n, A_d,
            lda, B_d, ldb, C_d, ldc, Acc_d, Cnt_d);
    cl::sycl::nd_range<3> nd_range = gemm_op_t::get_nd_range(gemm_arg);
    if (!gemm_op_t::can_implement(gemm_arg)) {
        std::cout << "The arguments cannot be supported, aborting ... "
                  << std::endl;
        FAIL();
    }

    size_t ops = 2 * matrix_m * matrix_n * matrix_k + matrix_m * matrix_n;
    profiling_helper prof("dequantize_gemm", ops, "gflops");
    int constexpr warm = 0;
    try {
        for (int i = 0; i < iter + warm; i++) {
            if (i >= warm) prof.cpu_start();
            auto e_esimd = queue.submit([&](handler &cgh) {
                cgh.parallel_for(
                        nd_range, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
                            // allocate slm and nbarrier resource
                            slm_barrier_init<gemm_op_t>();
                            gemm_op_t gemm_op;
                            gemm_op(item, gemm_arg);
                        });
            });
            if (i >= warm) {
                e_esimd.wait();
                prof.cpu_end();
                prof.add_gpu_event(e_esimd);
            }
        }
    } catch (cl::sycl::exception const &e) {
        std::cout << "SYCL exception caught: " << e.what() << '\n';
        FAIL();
    }

    //performance
    prof.print_profiling_result(profiling_selector::GPU);
    queue.memcpy((void *)C_h, (void *)C_d, size_c * sizeof(data_type_c)).wait();
    ASSERT_EQ(0,
            gemm_result_validate(A_h, B_h, C_h, matrix_m, matrix_k, matrix_n,
                    layout_a, layout_b));

    free(A_h, context);
    free(B_h, context);
    free(C_h, context);
    free(A_d, context);
    free(B_d, context);
    free(C_d, context);
    free(Acc_h, context);
    free(Cnt_h, context);
    free(Acc_d, context);
    free(Cnt_d, context);
}

template <typename T>
class fpu_fp32_gemm_test : public ::testing::Test {};
TYPED_TEST_SUITE_P(fpu_fp32_gemm_test);

TYPED_TEST_P(fpu_fp32_gemm_test, esimd) {
    // sycl_fpu_fp32_gemm_run<TypeParam>(ITER);
    fpu_fp32_gemm_run<TypeParam>(ITER);
}

REGISTER_TYPED_TEST_SUITE_P(fpu_fp32_gemm_test, esimd);
using tests = ::testing::Types<t1>;

INSTANTIATE_TYPED_TEST_SUITE_P(
        fpu_fp32_gemm_test_suite, fpu_fp32_gemm_test, tests);
