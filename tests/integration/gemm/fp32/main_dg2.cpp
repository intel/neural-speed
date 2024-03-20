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

#include <utils/utils.hpp>
#include "xetla.hpp"
// #define UT_DEBUG 1
using namespace gpu::xetla;
//The number of times the kernel is executed
constexpr int ITER = 10;

class t1 {
public:
	//Extract the parameters required by different test cases
	static constexpr size_t mat_m = 16;
	static constexpr size_t mat_n = 16;
	static constexpr size_t mat_k = 16;
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
int gemm_result_validate(data_type_a* A, data_type_b* B, data_type_c* C,
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

	// Turn on the enable_profiling property to facilitate subsequent profiling
	sycl::property_list properties{ sycl::property::queue::enable_profiling() };
	auto queue = sycl::queue(properties);
	auto context = queue.get_info<info::queue::context>();
	auto device = queue.get_info<info::queue::device>();

	std::cout << "Running on " << device.get_info<info::device::name>() << "\n";

	using tile_shape = xetla::group::tile_shape_t<wg_tile_n, wg_tile_m,
		sg_tile_n, sg_tile_m>;
	static constexpr uint32_t periodic_sync_interval = 0;
	static constexpr uint32_t prefetch_distance = 1;

	using mem_desc_a_t = xetla::mem_desc_t<data_type_a, mem_layout::row_major,
		mem_space::global, DEVICE_MEM_ALIGNMENT / sizeof(data_type_a)>;
	using mem_desc_b_t = xetla::mem_desc_t<data_type_b, mem_layout::row_major,
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


	using epilogue_t = xetla::group::epilogue_t<xetla::group::epilogue_policy_default<gpu_arch::Dg2>,
		tile_shape, mem_desc_c_t>;

	using group_swizzle = xetla::kernel::group_swizzle_default<gpu_arch::Dg2>;
	using gemm_op_t = xetla::kernel::gemm_universal_t<
		gpu::xetla::kernel::dispatch_policy_kslicing<
		group_swizzle, global_kslicing, local_kslicing>,
		gemm_t, epilogue_t>;

	size_t size_acc = gemm_op_t::get_acc_buf_size(matrix_m, matrix_n);
	size_t size_cnt = gemm_op_t::get_cnt_buf_size(matrix_m, matrix_n);

	//Define and initialize the data required for the calculation
	auto* A_h = static_cast<data_type_a*>(
		malloc_host(size_a * sizeof(data_type_a), context));
	auto* B_h = static_cast<data_type_b*>(
		malloc_host(size_b * sizeof(data_type_b), context));
	auto* C_h = static_cast<data_type_c*>(
		malloc_host(size_c * sizeof(data_type_c), context));
	auto* Acc_h = static_cast<data_type_acc*>(
		malloc_host(size_acc * sizeof(data_type_acc), context));
	auto* Cnt_h = static_cast<uint32_t*>(
		malloc_host(size_cnt * sizeof(uint32_t), context));

	auto* A_d = static_cast<data_type_a*>(
		aligned_alloc_device(DEVICE_MEM_ALIGNMENT,
			size_a * sizeof(data_type_a), device, context));
	auto* B_d = static_cast<data_type_b*>(
		aligned_alloc_device(DEVICE_MEM_ALIGNMENT,
			size_b * sizeof(data_type_b), device, context));
	auto* C_d = static_cast<data_type_c*>(
		aligned_alloc_device(DEVICE_MEM_ALIGNMENT,
			size_c * sizeof(data_type_c), device, context));
	auto* Acc_d = static_cast<data_type_acc*>(
		aligned_alloc_device(DEVICE_MEM_ALIGNMENT,
			size_acc * sizeof(data_type_acc), device, context));
	auto* Cnt_d
		= static_cast<uint32_t*>(aligned_alloc_device(DEVICE_MEM_ALIGNMENT,
			size_cnt * sizeof(uint32_t), device, context));

	for (unsigned i = 0; i < size_a; ++i) {
		A_h[i] = random_float();
	}
	for (unsigned i = 0; i < size_b; ++i) {
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


	queue.memcpy((void*)A_d, (void*)A_h, size_a * sizeof(data_type_a)).wait();
	queue.memcpy((void*)B_d, (void*)B_h, size_b * sizeof(data_type_b)).wait();
	queue.memcpy((void*)C_d, (void*)C_h, size_c * sizeof(data_type_c)).wait();
	queue.memcpy((void*)Acc_d, (void*)Acc_h, size_acc * sizeof(data_type_acc))
		.wait();
	queue.memcpy((void*)Cnt_d, (void*)Cnt_h, size_cnt * sizeof(uint32_t))
		.wait();

	typename gemm_op_t::arguments_t gemm_arg(matrix_m, matrix_k, matrix_n, A_d,
		matrix_k, B_d, matrix_n, C_d, matrix_n,
		Acc_d, Cnt_d);
	cl::sycl::nd_range<3> nd_range = gemm_op_t::get_nd_range(gemm_arg);
	if (!gemm_op_t::can_implement(gemm_arg)) {
		std::cout << "The arguments cannot be supported, aborting ... "
			<< std::endl;
		FAIL();
	}

	size_t ops = 2 * matrix_m * matrix_n * matrix_k + matrix_m * matrix_n;
	profiling_helper prof("dequantize_gemm", ops, "gflops");
	int constexpr warm = 10;
	try {
		for (int i = 0; i < iter + warm; i++) {
			if (i >= warm)
				prof.cpu_start();
			auto e_esimd = queue.submit([&](handler& cgh) {
				cgh.parallel_for(
					nd_range, [=](nd_item<3> item) SYCL_ESIMD_KERNEL{
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
	}
	catch (cl::sycl::exception const& e) {
		std::cout << "SYCL exception caught: " << e.what() << '\n';
		FAIL();
	}

	//performance
	prof.print_profiling_result(profiling_selector::GPU);
	queue.memcpy((void*)C_h, (void*)C_d, size_c * sizeof(data_type_c)).wait();
	ASSERT_EQ(0,
		gemm_result_validate(A_h, B_h, C_h, matrix_m, matrix_k, matrix_n));

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

#if 0
template <typename T>
class fpu_fp32_gemm_test : public ::testing::Test {};
TYPED_TEST_SUITE_P(fpu_fp32_gemm_test);

TYPED_TEST_P(fpu_fp32_gemm_test, esimd) {
	fpu_fp32_gemm_run<TypeParam>(ITER);
}

REGISTER_TYPED_TEST_SUITE_P(fpu_fp32_gemm_test, esimd);
using tests = ::testing::Types<t1, t2, t3>;

INSTANTIATE_TYPED_TEST_SUITE_P(
	fpu_fp32_gemm_test_suite, fpu_fp32_gemm_test, tests);
#endif


class sycl1 {
public:
	static constexpr size_t mat_m = 64;
	static constexpr size_t mat_n = 64;
	static constexpr size_t mat_k = 32;
	static int constexpr sub_size = 16;
	static int constexpr thread_tile_m = 1;
	static int constexpr thread_tile_n = 4;
	static int constexpr thread_tile_k = 4;
	static int constexpr group_m = 16;
	static int constexpr group_n = 16;
	using data_type_a = float;
	using data_type_b = float;
	using data_type_c = float;
	using data_type_acc = float;
};

class sycl2 {
public:
	static constexpr size_t mat_m = 1024;
	static constexpr size_t mat_n = 4096;
	static constexpr size_t mat_k = 4096;
	static int constexpr sub_size = 16;
	static int constexpr thread_tile_m = 8;
	static int constexpr thread_tile_n = 4;
	static int constexpr thread_tile_k = 16;
	static int constexpr group_m = 16;
	static int constexpr group_n = 16;
	using data_type_a = float;
	using data_type_b = float;
	using data_type_c = float;
	using data_type_acc = float;
};

class sycl3 {
public:
	static constexpr size_t mat_m = 1024;
	static constexpr size_t mat_n = 4096;
	static constexpr size_t mat_k = 4096;
	static int constexpr sub_size = 16;
	static int constexpr thread_tile_m = 8;
	static int constexpr thread_tile_n = 4;
	static int constexpr thread_tile_k = 4;
	static int constexpr group_m = 8;
	static int constexpr group_n = 16;
	using data_type_a = float;
	using data_type_b = float;
	using data_type_c = float;
	using data_type_acc = float;
};


template <class Test>
void sycl_fpu_fp32_gemm_run(int iter) {
	using namespace gpu;
	// Accept incoming parameters
	constexpr size_t matrix_m = Test::mat_m;
	constexpr size_t matrix_n = Test::mat_n;
	constexpr size_t matrix_k = Test::mat_k;

	using data_type_a = typename Test::data_type_a;
	using data_type_b = typename Test::data_type_b;
	using data_type_c = typename Test::data_type_c;
	using data_type_acc = typename Test::data_type_acc;

	constexpr size_t size_a = matrix_m * matrix_k;
	constexpr size_t size_b = matrix_k * matrix_n;
	constexpr size_t size_c = matrix_m * matrix_n;

	// Turn on the enable_profiling property to facilitate subsequent profiling
	sycl::property_list properties{ sycl::property::queue::enable_profiling() };
	auto queue = sycl::queue(properties);
	auto context = queue.get_info<info::queue::context>();
	auto device = queue.get_info<info::queue::device>();

	std::cout << "Running on " << device.get_info<info::device::name>() << "\n";

	static constexpr uint32_t periodic_sync_interval = 0;
	static constexpr uint32_t prefetch_distance = 1;



	//Define and initialize the data required for the calculation
	auto* A_h = static_cast<data_type_a*>(
		malloc_host(size_a * sizeof(data_type_a), context));
	auto* B_h = static_cast<data_type_b*>(
		malloc_host(size_b * sizeof(data_type_b), context));
	auto* C_h = static_cast<data_type_c*>(
		malloc_host(size_c * sizeof(data_type_c), context));

	auto* A_d = static_cast<data_type_a*>(
		aligned_alloc_device(DEVICE_MEM_ALIGNMENT,
			size_a * sizeof(data_type_a), device, context));
	auto* B_d = static_cast<data_type_b*>(
		aligned_alloc_device(DEVICE_MEM_ALIGNMENT,
			size_b * sizeof(data_type_b), device, context));
	auto* C_d = static_cast<data_type_c*>(
		aligned_alloc_device(DEVICE_MEM_ALIGNMENT,
			size_c * sizeof(data_type_c), device, context));

	for (unsigned i = 0; i < size_a; ++i) {
		A_h[i] = random_float();
	}
	for (unsigned i = 0; i < size_b; ++i) {
		B_h[i] = random_float();
	}

	for (unsigned i = 0; i < size_c; ++i) {
		C_h[i] = 0;
	}

	queue.memcpy((void*)A_d, (void*)A_h, size_a * sizeof(data_type_a)).wait();
	queue.memcpy((void*)B_d, (void*)B_h, size_b * sizeof(data_type_b)).wait();
	queue.memcpy((void*)C_d, (void*)C_h, size_c * sizeof(data_type_c)).wait();

	size_t ops = 2 * matrix_m * matrix_n * matrix_k + matrix_m * matrix_n;
	profiling_helper prof("dequantize_gemm", ops, "gflops");
	int constexpr warm = 10;
	try {
		for (int i = 0; i < iter + warm; i++) {
			if (i >= warm)
				prof.cpu_start();
			auto constexpr SubSize = Test::sub_size;
			int constexpr GroupM = Test::group_m;
			int constexpr GroupN = Test::group_n;
			int constexpr GroupWorkers = GroupM * GroupN;
			int constexpr TileM = Test::thread_tile_m;
			int constexpr TileN = Test::thread_tile_n;
			int constexpr TileK = Test::thread_tile_k;
			sycl::range<2> group{ GroupM,GroupN };
			sycl::range<2> problem{ matrix_m / TileM, matrix_n / TileN };
			int constexpr SLM_B_Stride = GroupN * TileN + 0;
			int constexpr SLM_B_Size = SLM_B_Stride * TileK;
			int constexpr SLM_A_Size = GroupM * TileM * TileK;
			auto e_esimd = queue.submit([&](handler& cgh) {
				sycl::accessor<float, 1, sycl::access::mode::read_write,
				sycl::access::target::local>
				slm_b(sycl::range(SLM_B_Size * 1), cgh);

			sycl::stream out(65536, 512, cgh);
			cgh.parallel_for(
				sycl::nd_range<2>(problem, group), [=](sycl::nd_item<2> it) [[intel::reqd_sub_group_size(SubSize)]] {
					int gm = it.get_group(0);
					int gn = it.get_group(1);
					int globalId = it.get_global_linear_id();
					auto sg = it.get_sub_group();
					int sgSize = sg.get_local_range()[0];
					int sgGroupId = sg.get_group_id()[0];
					int sgId = sg.get_local_id()[0];
					float tmp[TileM * TileN];
					for (size_t im = 0; im < TileM; im++)
						for (size_t in = 0; in < TileN; in++)
							tmp[im * TileN + in] = 0.f;

					int tm = gm * GroupM + sgGroupId;
					tm *= TileM;
					int tn = gn * GroupN + sgId;
					tn *= TileN;
					int subn = gn * GroupN * TileN;
					size_t i = 0;

					for (; i < matrix_k; i += TileK)
					{
						if (sgGroupId < TileK)
							//static_assert(GroupM == TileK);
						{
							for (size_t in = 0; in < TileN; in += 4)
							{
								*(sycl::float4*)&slm_b[sgGroupId * SLM_B_Stride + sgId * TileN + in] =
									*(sycl::float4*)&B_d[tn + in + (i + sgGroupId) * matrix_n];
							}
						}
						it.barrier(sycl::access::fence_space::local_space);
						int constexpr UnrollK = 2;
#pragma unroll(UnrollK)
						for (size_t ik = 0; ik < TileK; ik += 1)
						{
							float tmpB[TileN];
							for (size_t in = 0; in < TileN; in += 4)
							{
								*(sycl::float4*)&tmpB[in] =
									*(sycl::float4*)&slm_b[sgId * TileN + in + (ik)*SLM_B_Stride];
							}
							//static_assert(SubSize == TileM);
							//auto tmpA = sgId < TileM ? A_d[(tm + sgId) * matrix_k + (i + ik)] : 0.f;
							for (size_t im = 0; im < TileM; im++)
							{
								auto tmpA = A_d[(tm + im) * matrix_k + (i + ik)];
								//auto tmpA = slm_b[sgGroupId * TileK * TileM + im + ik * TileM];
								//auto tmpA = A_d[im + (i + ik) * TileK + tm * matrix_k];
								//auto tmpA = slm_a[sgGroupId * TileM + im + (ik)*GroupM * TileM];
								//auto A = sg.shuffle(tmpA, im);
								for (size_t in = 0; in < TileN; in++)
								{
									tmp[im * TileN + in] += tmpA * tmpB[in];
								}
							}
						}
						it.barrier(sycl::access::fence_space::local_space);

					}
					for (size_t im = 0; im < TileM; im++) {
						for (size_t in = 0; in < TileN; in += 4)
						{
							*(sycl::float4*)&C_d[(tm + im) * matrix_n + tn + in] =
								*(sycl::float4*)&tmp[im * TileN + in];
						}
					}

				});
				});
			if (i >= warm) {
				e_esimd.wait();
				prof.cpu_end();
				prof.add_gpu_event(e_esimd);
			}
		}
	}
	catch (cl::sycl::exception const& e) {
		std::cout << "SYCL exception caught: " << e.what() << '\n';
		FAIL();
	}

	//performance
	prof.print_profiling_result(profiling_selector::GPU);
	queue.memcpy((void*)C_h, (void*)C_d, size_c * sizeof(data_type_c)).wait();
	ASSERT_EQ(0,
		gemm_result_validate(A_h, B_h, C_h, matrix_m, matrix_k, matrix_n));

	free(A_h, context);
	free(B_h, context);
	free(C_h, context);
	free(A_d, context);
	free(B_d, context);
	free(C_d, context);
}


class sycl2d_0 {
public:
	static constexpr size_t mat_m = 1024;
	static constexpr size_t mat_n = 4096;
	static constexpr size_t mat_k = 4096;
	static int constexpr sub_size = 16;
	static int constexpr thread_tile_m = 8;
	static int constexpr thread_tile_n = 4;
	static int constexpr thread_tile_k = 16;
	static int constexpr group_m = 32;
	static int constexpr group_n = 32;
	using data_type_a = float;
	using data_type_b = float;
	using data_type_c = float;
	using data_type_acc = float;
};
template <class Test>
void sycl2d_fpu_fp32_gemm_run(int iter) {
	using namespace gpu;
	// Accept incoming parameters
	constexpr size_t matrix_m = Test::mat_m;
	constexpr size_t matrix_n = Test::mat_n;
	constexpr size_t matrix_k = Test::mat_k;

	using data_type_a = typename Test::data_type_a;
	using data_type_b = typename Test::data_type_b;
	using data_type_c = typename Test::data_type_c;
	using data_type_acc = typename Test::data_type_acc;

	constexpr size_t size_a = matrix_m * matrix_k;
	constexpr size_t size_b = matrix_k * matrix_n;
	constexpr size_t size_c = matrix_m * matrix_n;

	// Turn on the enable_profiling property to facilitate subsequent profiling
	sycl::property_list properties{ sycl::property::queue::enable_profiling() };
	auto queue = sycl::queue(properties);
	auto context = queue.get_info<info::queue::context>();
	auto device = queue.get_info<info::queue::device>();

	std::cout << "Running on " << device.get_info<info::device::name>() << "\n";


	//Define and initialize the data required for the calculation
	auto* A_h = static_cast<data_type_a*>(
		malloc_host(size_a * sizeof(data_type_a), context));
	auto* B_h = static_cast<data_type_b*>(
		malloc_host(size_b * sizeof(data_type_b), context));
	auto* C_h = static_cast<data_type_c*>(
		malloc_host(size_c * sizeof(data_type_c), context));

	auto* A_d = static_cast<data_type_a*>(
		aligned_alloc_device(DEVICE_MEM_ALIGNMENT,
			size_a * sizeof(data_type_a), device, context));
	auto* B_d = static_cast<data_type_b*>(
		aligned_alloc_device(DEVICE_MEM_ALIGNMENT,
			size_b * sizeof(data_type_b), device, context));
	auto* C_d = static_cast<data_type_c*>(
		aligned_alloc_device(DEVICE_MEM_ALIGNMENT,
			size_c * sizeof(data_type_c), device, context));

	for (unsigned i = 0; i < size_a; ++i) {
		A_h[i] = random_float();
	}
	for (unsigned i = 0; i < size_b; ++i) {
		B_h[i] = random_float();
	}

	for (unsigned i = 0; i < size_c; ++i) {
		C_h[i] = 0;
	}

	queue.memcpy((void*)A_d, (void*)A_h, size_a * sizeof(data_type_a)).wait();
	queue.memcpy((void*)B_d, (void*)B_h, size_b * sizeof(data_type_b)).wait();
	queue.memcpy((void*)C_d, (void*)C_h, size_c * sizeof(data_type_c)).wait();

	size_t ops = 2 * matrix_m * matrix_n * matrix_k + matrix_m * matrix_n;
	profiling_helper prof("dequantize_gemm", ops, "gflops");
	int constexpr warm = 10;
	try {
		for (int i = 0; i < iter + warm; i++) {
			if (i >= warm)
				prof.cpu_start();
			int constexpr SubSize = Test::sub_size;
			int constexpr GroupM = Test::group_m;
			int constexpr GroupN = Test::group_n;
			auto constexpr SubNStride = GroupN / SubSize;
			int constexpr GroupWorkers = GroupM * GroupN;
			int constexpr SubCount = GroupWorkers / SubSize;
			int constexpr TileM = Test::thread_tile_m;
			int constexpr TileN = Test::thread_tile_n;
			int constexpr TileK = Test::thread_tile_k;
			int constexpr GroupNEle = GroupN * TileN;
			int constexpr GroupMEle = GroupM * TileM;
			int constexpr SubGroupNEle = SubSize * TileN;
			int constexpr SLM_B_Size = GroupNEle * TileK;
			int constexpr SLM_A_Size = GroupMEle * TileK;
			sycl::range<2> group{ GroupM,GroupN };
			sycl::range<2> problem{ matrix_m / TileM, matrix_n / TileN };
			auto e_esimd = queue.submit([&](handler& cgh) {
				sycl::accessor<float, 1, sycl::access::mode::read_write,
				sycl::access::target::local>
				slm_b(sycl::range(SLM_B_Size), cgh);
			sycl::accessor<float, 1, sycl::access::mode::read_write,
				sycl::access::target::local>
				slm_a(sycl::range(SLM_A_Size), cgh);
			sycl::stream out(65536, 512, cgh);
			cgh.parallel_for(
				sycl::nd_range<2>(problem, group), [=](sycl::nd_item<2> it) [[intel::reqd_sub_group_size(SubSize)]] {
					int g_idxm = it.get_group(0);
					int g_idxn = it.get_group(1);
					auto sg = it.get_sub_group();
					int sgGroupId = sg.get_group_id()[0];
					int sgId = sg.get_local_id()[0];
					float tmp[TileM * TileN];
					for (size_t im = 0; im < TileM; im++)
						for (size_t in = 0; in < TileN; in++)
							tmp[im * TileN + in] = 0.f;
					int sg_idxn = sgGroupId % SubNStride;
					int sg_idxm = sgGroupId / SubNStride;
					int gm = g_idxm * GroupM;
					int gn = g_idxn * GroupN;
					int sgm = gm + sg_idxm;
					int sgn = gn + sg_idxn * SubSize;
					int tm = sgm * TileM;
					int tn = (sgn + sgId) * TileN;

					for (size_t i = 0; i < matrix_k; i += TileK)
					{
						int constexpr Iter_PerWorker = (TileK + GroupM - 1) / GroupM;
						for (size_t icp = 0; icp < Iter_PerWorker; icp++)
						{
							if (sg_idxm + icp * GroupM < TileK)
							{
								for (size_t in = 0; in < TileN; in++)
								{
									slm_b[(sg_idxm + icp * GroupM) * GroupNEle + (sg_idxn * SubSize + sgId) * TileN + in] = B_d[tn + in + (i + sg_idxm + icp * GroupM) * matrix_n];
								}
							}
						}
						/*static_assert(TileK == SubSize);
						int constexpr IterA_PerWorker = (GroupMEle + SubCount - 1) / SubCount;
						for (size_t icp = 0; icp < IterA_PerWorker; icp++)
						{
							auto t_gm = sgGroupId + icp * SubCount;
							if (t_gm < GroupMEle)
							{
								slm_a[t_gm + sgId * GroupMEle] = A_d[(gm * TileM + t_gm) * matrix_k + sgId + i];
							}
						}*/
						it.barrier(sycl::access::fence_space::local_space);
#pragma unroll(2)
						for (size_t ik = 0; ik < TileK; ik++)
						{
							float tmpB[TileN];
							for (size_t in = 0; in < TileN; in++)
							{
								tmpB[in] = slm_b[sg_idxn * SubGroupNEle + sgId * TileN + in + ik * GroupNEle];
							}
							for (size_t im = 0; im < TileM; im++)
							{
								//auto tmpA = A_d[(tm + im) * matrix_k + i + ik];
								auto tmpA = slm_b[im + ik * TileM];
								//auto tmpA = A_d[(tm + im)  + (i + ik) * matrix_m];
								//auto tmpA = slm_a[(sg_idxm * TileM + im) + ik * GroupMEle];
								for (size_t in = 0; in < TileN; in++)
								{
									tmp[im * TileN + in] += tmpA * tmpB[in];
								}
							}
						}
						it.barrier(sycl::access::fence_space::local_space);

					}
					for (size_t im = 0; im < TileM; im++) {
						for (size_t in = 0; in < TileN; in++)
						{
							C_d[(tm + im) * matrix_n + tn + in] = tmp[im * TileN + in];
						}
					}
				});
				});
			if (i >= warm) {
				e_esimd.wait();
				prof.cpu_end();
				prof.add_gpu_event(e_esimd);
			}
		}
	}
	catch (cl::sycl::exception const& e) {
		std::cout << "SYCL exception caught: " << e.what() << '\n';
		FAIL();
	}

	//performance
	prof.print_profiling_result(profiling_selector::GPU);
	queue.memcpy((void*)C_h, (void*)C_d, size_c * sizeof(data_type_c)).wait();
	ASSERT_EQ(0,
		gemm_result_validate(A_h, B_h, C_h, matrix_m, matrix_k, matrix_n));

	free(A_h, context);
	free(B_h, context);
	free(C_h, context);
	free(A_d, context);
	free(B_d, context);
	free(C_d, context);
}


template <typename T>
class sycl_fp32_gemm_test : public ::testing::Test {};
TYPED_TEST_SUITE_P(sycl_fp32_gemm_test);
//#define SYCL2D
TYPED_TEST_P(sycl_fp32_gemm_test, sycl) {
#ifdef SYCL2D
	sycl2d_fpu_fp32_gemm_run<TypeParam>(ITER);
#else
	sycl_fpu_fp32_gemm_run<TypeParam>(ITER);
#endif
}

REGISTER_TYPED_TEST_SUITE_P(sycl_fp32_gemm_test, sycl);
#ifdef SYCL2D
using tests_sycl = ::testing::Types<sycl2d_0>;
#else
using tests_sycl = ::testing::Types<sycl2>;
#endif

INSTANTIATE_TYPED_TEST_SUITE_P(
	sycl_fp32_gemm_test_suite, sycl_fp32_gemm_test, tests_sycl);