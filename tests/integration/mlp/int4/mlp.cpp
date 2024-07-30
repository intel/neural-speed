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
#include "int4_mlp_gate_mul_up_fwd.hpp"
#include "xetla.hpp"

// #define UT_DEBUG
using namespace gpu::xetla;
using namespace gpu::xetla::group;
// The number of times the kernel is executed
#ifdef UT_DEBUG
constexpr int ITER = 1;
#else
constexpr int ITER = 200;
#endif
constexpr size_t UNDEFINED_DATA_SIZE = 1024;

class test_col_major_1 {
 public:
  // Extract the parameters required by different test cases
  static constexpr size_t mat_m = 1;
  static constexpr size_t mat_n = 11008;
  static constexpr size_t mat_k = 4096;
  static constexpr size_t wg_m = 1;
  static constexpr size_t wg_n = 1;
  static constexpr size_t sg_m = 1;
  static constexpr size_t sg_n = 1;
  static constexpr size_t sg_k = 512;
  static constexpr size_t dequant_s = 128;
  static constexpr quant_mode quant_mode = quant_mode::I4_ASYM;

  static constexpr size_t local_kslicing = 1;
  static constexpr size_t global_kslicing = 1;
  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = mem_layout::col_major;
  static constexpr mma_engine mma_eng = mma_engine::fpu;
  static constexpr gpu_arch arch = gpu_arch::XeHpg;
  using data_type_a = fp16;
  using data_type_b = int4x8;
  using data_type_c = fp16;
};

class test_col_major_2 {
 public:
  // Extract the parameters required by different test cases
  static constexpr size_t mat_m = 256;
  static constexpr size_t mat_n = 11008;
  static constexpr size_t mat_k = 1024;
  static constexpr size_t wg_m = 4;
  static constexpr size_t wg_n = 1;
  static constexpr size_t sg_m = 4;
  static constexpr size_t sg_n = 1;
  static constexpr size_t sg_k = 1024 / 4;
  static constexpr size_t dequant_s = 128;
  static constexpr quant_mode quant_mode = quant_mode::I4_SYM;

  static constexpr size_t local_kslicing = 1;
  static constexpr size_t global_kslicing = 1;
  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = mem_layout::col_major;
  static constexpr mma_engine mma_eng = mma_engine::fpu;
  static constexpr gpu_arch arch = gpu_arch::XeHpg;
  using data_type_a = fp16;
  using data_type_b = int4x8;
  using data_type_c = fp16;
};

template <
    typename data_type_a,
    typename data_type_b,
    typename data_type_c,
    typename data_type_acc = float>
int int4_mlp_result_validate(
    data_type_a* A,
    data_type_b* up_proj,
    data_type_b* gate_proj,
    data_type_c* C,
    uint32_t m,
    uint32_t k,
    uint32_t n,
    mem_layout mem_layout_a_ = mem_layout::row_major,
    mem_layout mem_layout_b_ = mem_layout::row_major) {
  buff_cmp::buff_vals<data_type_c> data(C, m, n, n);
  std::vector<data_type_acc> gold_up_proj_out(m * n, 0);
  std::vector<data_type_acc> gold_gate_proj_out(m * n, 0);
  std::vector<data_type_acc> gold_C(m * n, 0);
  get_gemm_gold<data_type_a, data_type_b, data_type_acc>(
      m,
      n,
      k,
      mem_layout_a_,
      mem_layout_b_,
      A,
      up_proj,
      gold_up_proj_out.data());
  get_gemm_gold<data_type_a, data_type_b, data_type_acc>(
      m,
      n,
      k,
      mem_layout_a_,
      mem_layout_b_,
      A,
      gate_proj,
      gold_gate_proj_out.data());
  for (uint32_t i = 0; i < m; i++) {
    for (uint32_t j = 0; j < n; j++) {
      gold_C[i * n + j] = gold_up_proj_out[i * n + j] *
          (gold_gate_proj_out[i * n + j] /
           (1 + std::exp(-gold_gate_proj_out[i * n + j])));
    }
  }

  buff_cmp::buff_vals<data_type_c, data_type_acc> other(gold_C.data(), m, n, n);

  bool result =
      buff_cmp::xetla_buff_cmp(data, other, "int4_mlp-fusion validation");

#ifdef UT_DEBUG
  // for (uint32_t i = 0; i < m; i++) {
  //   for (uint32_t j = 0; j < n; j++) {
  //     std::cout << float(sycl::half(C[i * n + j])) << " ";
  //   }
  //   std::cout << std::endl;
  // }
#endif
  std::cout << (!result ? "FAILED\n" : "PASSED\n");
  return result ? 0 : 1;
}

template <
    quant_mode quant_mode = quant_mode::I4_SYM,
    typename data_type_acc_in = fp16,
    typename data_type_b,
    typename data_type_scale,
    typename data_type_zero_pt>
std::vector<fp16> convert_int4(
    data_type_b data_b,
    data_type_scale scale,
    data_type_zero_pt zero_pt) {
  std::vector<fp16> dequant_fp16(sizeof(data_type_b) * 2);

  int8_t zero_pt_i8 = zero_pt & 0xf;
  for (uint32_t i = 0; i < dequant_fp16.size(); i++) {
    int8_t dequant_8bit = data_b & 0xf;
    if constexpr (quant_mode == quant_mode::I4_SYM) {
      dequant_fp16[i] = scale * (dequant_8bit - 8);
    } else {
      dequant_fp16[i] = scale * (dequant_8bit - zero_pt_i8);
    }
    data_b = data_b >> 4;
  }
  return dequant_fp16;
}

template <
    size_t dequant_s,
    mem_layout layout_b = mem_layout::col_major,
    quant_mode quant_mode = quant_mode::I4_SYM,
    typename data_type_acc_in = fp16,
    typename data_type_b,
    typename data_type_scale,
    typename data_type_zero_pt>
std::vector<data_type_acc_in> dequantize_weight(
    size_t matrix_k,
    size_t matrix_n,
    data_type_b* b,
    data_type_scale* scale,
    data_type_zero_pt* zero_pt) {
  std::vector<data_type_acc_in> b_out(matrix_k * matrix_n, 0);
  constexpr size_t pack_radio = 2 * sizeof(data_type_b);
  size_t width = layout_b == mem_layout::row_major ? matrix_n / pack_radio
                                                   : matrix_k / pack_radio;
  size_t height = layout_b == mem_layout::row_major ? matrix_k : matrix_n;
  size_t step = layout_b == mem_layout::row_major ? 1 : dequant_s / pack_radio;

  for (uint32_t i = 0; i < height; i++) {
    for (uint32_t j = 0; j < width; j += step) {
      int start_b_in = i * width + j;
      int start_scale_in = start_b_in / step;
      int start_zero_pt_in =
          (j / step) * (matrix_n / pack_radio) + i / pack_radio;
      int start_out =
          layout_b == mem_layout::row_major ? 0 : i * matrix_k + j * pack_radio;
      for (uint32_t jj = 0; jj < step; jj++) {
        std::vector<fp16> dequant_fp16 = convert_int4<quant_mode>(
            b[start_b_in + jj],
            scale[start_scale_in],
            zero_pt[start_zero_pt_in] >> (4 * (i % pack_radio)));
        for (uint32_t jjj = 0; jjj < dequant_fp16.size(); jjj++) {
          b_out[start_out + pack_radio * jj + jjj] = dequant_fp16[jjj];
        }
      }
    }
  }
#ifdef UT_DEBUG
  // for (uint32_t i = 0; i < matrix_n; i++) {
  //   for (uint32_t j = 0; j < matrix_k; j++) {
  //     std::cout << float(sycl::half(b_out[i * matrix_k + j])) << " ";
  //   }
  //   std::cout << std::endl;
  // }
#endif
  return b_out;
}

template <class Test>
void dequantize_int4_mlp_run(int iter) {
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
  constexpr size_t dequant_s = std::min(Test::dequant_s, matrix_k);
  constexpr quant_mode quant_mode = Test::quant_mode;
  using data_type_a = typename Test::data_type_a;
  using data_type_b = typename Test::data_type_b;
  using data_type_c = typename Test::data_type_c;
  using data_type_zero_pt = data_type_b;
  using data_type_scale = fp16;
  using data_type_acc_in = fp16;
  using data_type_acc = float;

  constexpr mem_layout layout_a = Test::layout_a;
  constexpr mem_layout layout_b = Test::layout_b;

  constexpr size_t size_a = matrix_m * matrix_k;
  constexpr size_t size_b = matrix_k * matrix_n / (2 * sizeof(data_type_b));

  constexpr size_t size_scale_k = matrix_k / dequant_s;
  constexpr size_t size_scale_n = matrix_n;
  constexpr size_t size_scale = size_scale_k * size_scale_n;

  constexpr size_t size_zero_pt_k = matrix_k / dequant_s;
  constexpr size_t size_zero_pt_n = matrix_n;
  constexpr size_t size_zero_pt =
      size_zero_pt_k * size_zero_pt_n / (2 * sizeof(data_type_b));

  constexpr size_t size_c = matrix_m * matrix_n;

  uint32_t lda = layout_a == mem_layout::row_major ? matrix_k : matrix_m;
  uint32_t ldb = layout_b == mem_layout::row_major ? matrix_n : matrix_k;
  uint32_t ldc = matrix_n;
  uint32_t ld_scale =
      layout_b == mem_layout::row_major ? size_scale_n : size_scale_k;
  uint32_t ld_zero_pt = size_zero_pt_n;

  // Turn on the enable_profiling property to facilitate subsequent profiling
  sycl::property_list properties{
      sycl::property::queue::enable_profiling(),
      sycl::property::queue::in_order()};
  auto queue = sycl::queue(properties);
  auto context = queue.get_info<info::queue::context>();
  auto device = queue.get_info<info::queue::device>();

  std::cout << "Running on " << device.get_info<info::device::name>() << "\n";

  using tile_shape =
      xetla::group::tile_shape_t<wg_tile_n, wg_tile_m, sg_tile_n, sg_tile_m>;
  static constexpr uint32_t periodic_sync_interval = 0;
  static constexpr uint32_t prefetch_distance = 0;

  using mem_desc_a_t = xetla::mem_desc_t<
      data_type_a,
      layout_a,
      mem_space::global,
      DEVICE_MEM_ALIGNMENT / sizeof(data_type_a)>;
  using mem_desc_b_t = xetla::mem_desc_t<
      data_type_b,
      layout_b,
      mem_space::global,
      DEVICE_MEM_ALIGNMENT / sizeof(data_type_b)>;
  using mem_desc_c_t = xetla::mem_desc_t<
      data_type_c,
      mem_layout::row_major,
      mem_space::global,
      DEVICE_MEM_ALIGNMENT / sizeof(data_type_c)>;

  using compute_attr = xetla::group::
      compute_attr_t<data_type_acc_in, data_type_acc_in, data_type_acc>;
  using perf_tuning_knob = xetla::group::
      perf_tuning_knob_t<sg_tile_k, prefetch_distance, periodic_sync_interval>;
  static constexpr quant_info quant_info{quant_mode, Test::dequant_s, layout_b};
  using compute_policy = xetla::group::compute_policy_int4_dequantize<
      compute_attr,
      perf_tuning_knob,
      data_type_scale,
      data_type_zero_pt,
      quant_info,
      Test::mma_eng,
      Test::arch>;

  using gemm_t = xetla::group::
      gemm_t<compute_policy, tile_shape, mem_desc_a_t, mem_desc_b_t>;

  using epilogue_t = xetla::group::epilogue_t<
      xetla::group::epilogue_policy_default<Test::arch>,
      tile_shape,
      mem_desc_c_t>;

  using post_ops_up_t = subgroup::chained_tile_op_t<>;
  using post_ops_gate_t = subgroup::chained_tile_op_t<subgroup::silu_op_t>;

  using int4_mlp_op_t = xetla::mlp::int4_mlp_gate_mul_up_fwd_t<
      Test::arch,
      global_kslicing,
      local_kslicing,
      gemm_t,
      post_ops_up_t,
      post_ops_gate_t,
      epilogue_t>;

  size_t size_acc = int4_mlp_op_t::get_acc_buf_size(matrix_m, matrix_n);
  size_t size_cnt = int4_mlp_op_t::get_cnt_buf_size(matrix_m, matrix_n);

  // Define and initialize the data required for the calculation
  auto* A_h = static_cast<data_type_a*>(
      malloc_host(size_a * sizeof(data_type_a), context));
  auto* up_proj_h = static_cast<data_type_b*>(malloc_host(
      (size_b + UNDEFINED_DATA_SIZE) * sizeof(data_type_b), context));
  auto* gate_proj_h = static_cast<data_type_b*>(malloc_host(
      (size_b + UNDEFINED_DATA_SIZE) * sizeof(data_type_b), context));
  auto* C_h = static_cast<data_type_c*>(
      malloc_host(size_c * sizeof(data_type_c), context));
  auto* Acc_up_proj_h = static_cast<data_type_acc*>(
      malloc_host(size_acc * sizeof(data_type_acc), context));
  auto* Acc_gate_proj_h = static_cast<data_type_acc*>(
      malloc_host(size_acc * sizeof(data_type_acc), context));
  auto* Cnt_h =
      static_cast<uint32_t*>(malloc_host(size_cnt * sizeof(uint32_t), context));
  auto* scale_up_proj_h = static_cast<data_type_scale*>(malloc_host(
      (size_scale + UNDEFINED_DATA_SIZE) * sizeof(data_type_scale), context));
  auto* scale_gate_proj_h = static_cast<data_type_scale*>(malloc_host(
      (size_scale + UNDEFINED_DATA_SIZE) * sizeof(data_type_scale), context));

  auto* zero_pt_up_proj_h = static_cast<data_type_zero_pt*>(malloc_host(
      (size_zero_pt + UNDEFINED_DATA_SIZE) * sizeof(data_type_zero_pt),
      context));
  auto* zero_pt_gate_proj_h = static_cast<data_type_zero_pt*>(malloc_host(
      (size_zero_pt + UNDEFINED_DATA_SIZE) * sizeof(data_type_zero_pt),
      context));

  auto* A_d = static_cast<data_type_a*>(aligned_alloc_device(
      DEVICE_MEM_ALIGNMENT, size_a * sizeof(data_type_a), device, context));
  auto* up_proj_d = static_cast<data_type_b*>(aligned_alloc_device(
      DEVICE_MEM_ALIGNMENT,
      (size_b + UNDEFINED_DATA_SIZE) * sizeof(data_type_b),
      device,
      context));
  auto* gate_proj_d = static_cast<data_type_b*>(aligned_alloc_device(
      DEVICE_MEM_ALIGNMENT,
      (size_b + UNDEFINED_DATA_SIZE) * sizeof(data_type_b),
      device,
      context));
  auto* C_d = static_cast<data_type_c*>(aligned_alloc_device(
      DEVICE_MEM_ALIGNMENT, size_c * sizeof(data_type_c), device, context));
  auto* Acc_up_proj_d = static_cast<data_type_acc*>(aligned_alloc_device(
      DEVICE_MEM_ALIGNMENT, size_acc * sizeof(data_type_acc), device, context));
  auto* Acc_gate_proj_d = static_cast<data_type_acc*>(aligned_alloc_device(
      DEVICE_MEM_ALIGNMENT, size_acc * sizeof(data_type_acc), device, context));
  auto* Cnt_d = static_cast<uint32_t*>(aligned_alloc_device(
      DEVICE_MEM_ALIGNMENT, size_cnt * sizeof(uint32_t), device, context));
  auto* scale_up_proj_d = static_cast<data_type_scale*>(aligned_alloc_device(
      DEVICE_MEM_ALIGNMENT,
      (size_scale + UNDEFINED_DATA_SIZE) * sizeof(data_type_scale),
      device,
      context));
  auto* scale_gate_proj_d = static_cast<data_type_scale*>(aligned_alloc_device(
      DEVICE_MEM_ALIGNMENT,
      (size_scale + UNDEFINED_DATA_SIZE) * sizeof(data_type_scale),
      device,
      context));
  auto* zero_pt_up_proj_d =
      static_cast<data_type_zero_pt*>(aligned_alloc_device(
          DEVICE_MEM_ALIGNMENT,
          (size_zero_pt + UNDEFINED_DATA_SIZE) * sizeof(data_type_zero_pt),
          device,
          context));
  auto* zero_pt_gate_proj_d =
      static_cast<data_type_zero_pt*>(aligned_alloc_device(
          DEVICE_MEM_ALIGNMENT,
          (size_zero_pt + UNDEFINED_DATA_SIZE) * sizeof(data_type_zero_pt),
          device,
          context));

  for (unsigned i = 0; i < size_a; ++i) {
    A_h[i] = random_float();
#ifdef UT_DEBUG
    A_h[i] = 1;
    // A_h[i] = layout_a == mem_layout::row_major
    //     ? (i % matrix_k + i / matrix_k * 100)
    //     : (i % matrix_m + i / matrix_m * 100);
#endif
  }

  for (unsigned i = 0; i < size_b + UNDEFINED_DATA_SIZE; ++i) {
    if constexpr (std::is_same_v<int4x2, data_type_b>) {
      up_proj_h[i] = random_uint8();
      gate_proj_h[i] = random_uint8();
#ifdef UT_DEBUG
      up_proj_h[i] = 0x77;
      gate_proj_h[i] = 0x77;
#endif
    } else if constexpr (std::is_same_v<int4x8, data_type_b>) {
      up_proj_h[i] = random_uint32();
      gate_proj_h[i] = random_uint32();
#ifdef UT_DEBUG
      up_proj_h[i] = 0x77777777;
      gate_proj_h[i] = 0x77777777;
#endif
    }
  }

  for (unsigned i = 0; i < size_scale; ++i) {
    scale_up_proj_h[i] = random_float();
    scale_gate_proj_h[i] = random_float();
#ifdef UT_DEBUG
    scale_up_proj_h[i] = 1.f;
    scale_gate_proj_h[i] = 1.f;
#endif
  }
  for (unsigned i = size_scale; i < size_scale + UNDEFINED_DATA_SIZE; ++i) {
    scale_up_proj_h[i] = INFINITY;
    scale_gate_proj_h[i] = INFINITY;
  }
  for (unsigned i = 0; i < size_zero_pt + UNDEFINED_DATA_SIZE; ++i) {
    if constexpr (std::is_same_v<int4x2, data_type_b>) {
      zero_pt_up_proj_h[i] = random_uint8();
      zero_pt_gate_proj_h[i] = random_uint8();
#ifdef UT_DEBUG
      zero_pt_up_proj_h[i] = 0x12 << i;
      zero_pt_gate_proj_h[i] = 0x12 << i;
#endif
    } else if constexpr (std::is_same_v<int4x8, data_type_b>) {
      zero_pt_up_proj_h[i] = random_uint32();
      zero_pt_gate_proj_h[i] = random_uint32();
#ifdef UT_DEBUG
      zero_pt_up_proj_h[i] = 0x33333333;
      zero_pt_gate_proj_h[i] = 0x33333333;
#endif
    }
  }

  for (unsigned i = 0; i < size_c; ++i) {
    C_h[i] = random_float();
  }

  for (unsigned i = 0; i < size_acc; ++i) {
    Acc_up_proj_h[i] = random_float();
    Acc_gate_proj_h[i] = random_float();
  }

  for (unsigned i = 0; i < size_cnt; ++i) {
    Cnt_h[i] = random_uint8();
  }

  queue.memcpy((void*)A_d, (void*)A_h, size_a * sizeof(data_type_a)).wait();
  queue
      .memcpy(
          (void*)up_proj_d,
          (void*)up_proj_h,
          (size_b + UNDEFINED_DATA_SIZE) * sizeof(data_type_b))
      .wait();
  queue
      .memcpy(
          (void*)gate_proj_d,
          (void*)gate_proj_h,
          (size_b + UNDEFINED_DATA_SIZE) * sizeof(data_type_b))
      .wait();
  queue.memcpy((void*)C_d, (void*)C_h, size_c * sizeof(data_type_c)).wait();
  queue
      .memcpy(
          (void*)Acc_up_proj_d,
          (void*)Acc_up_proj_h,
          size_acc * sizeof(data_type_acc))
      .wait();
  queue
      .memcpy(
          (void*)Acc_gate_proj_d,
          (void*)Acc_gate_proj_h,
          size_acc * sizeof(data_type_acc))
      .wait();
  queue.memcpy((void*)Cnt_d, (void*)Cnt_h, size_cnt * sizeof(uint32_t)).wait();
  queue
      .memcpy(
          (void*)scale_up_proj_d,
          (void*)scale_up_proj_h,
          (size_scale + UNDEFINED_DATA_SIZE) * sizeof(data_type_scale))
      .wait();
  queue
      .memcpy(
          (void*)scale_gate_proj_d,
          (void*)scale_gate_proj_h,
          (size_scale + UNDEFINED_DATA_SIZE) * sizeof(data_type_scale))
      .wait();

  queue
      .memcpy(
          (void*)zero_pt_up_proj_d,
          (void*)zero_pt_up_proj_h,
          (size_zero_pt + UNDEFINED_DATA_SIZE) * sizeof(data_type_zero_pt))
      .wait();
  queue
      .memcpy(
          (void*)zero_pt_gate_proj_d,
          (void*)zero_pt_gate_proj_h,
          (size_zero_pt + UNDEFINED_DATA_SIZE) * sizeof(data_type_zero_pt))
      .wait();

  queue.memset(Cnt_d, 0, size_cnt * sizeof(uint32_t)).wait();
  queue.memset(Acc_up_proj_d, 0, size_acc * sizeof(data_type_acc)).wait();
  queue.memset(Acc_gate_proj_d, 0, size_acc * sizeof(data_type_acc)).wait();

  // set up int4_mlp arguments
  typename int4_mlp_op_t::quant_param_t quant_arg(
      scale_up_proj_d,
      scale_gate_proj_d,
      zero_pt_up_proj_d,
      zero_pt_gate_proj_d,
      ld_scale,
      ld_zero_pt);
  typename int4_mlp_op_t::arguments_t int4_mlp_arg(
      matrix_m,
      matrix_k,
      matrix_n,
      A_d,
      lda,
      up_proj_d,
      gate_proj_d,
      ldb,
      C_d,
      ldc,
      Acc_up_proj_d,
      Acc_gate_proj_d,
      Cnt_d,
      quant_arg,
      {},
      {{}});

  cl::sycl::nd_range<3> nd_range = int4_mlp_op_t::get_nd_range(int4_mlp_arg);
  // 2 gemm(2*m*n*k) + mul(m*n) + silu(3*m*n)
  size_t ops = 4 * matrix_m * matrix_n * matrix_k + 8 * matrix_m * matrix_n;
  profiling_helper prof("int4_mlp-fusion", ops, "gflops");
#ifdef UT_DEBUG
  int constexpr warm = 0;
#else
  int constexpr warm = 100;
#endif
  try {
    for (int i = 0; i < iter + warm; i++) {
      if (i >= warm)
        prof.cpu_start();
      auto e_esimd = queue.submit([&](handler& cgh) {
        cgh.parallel_for(nd_range, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
          // allocate slm and nbarrier resource
          slm_barrier_init<int4_mlp_op_t>();
          int4_mlp_op_t int4_mlp_op;
          int4_mlp_op(item, int4_mlp_arg);
        });
      });
      if (i >= warm) {
        e_esimd.wait();
        prof.cpu_end();
        prof.add_gpu_event(e_esimd);
      }
    }
  } catch (cl::sycl::exception const& e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    FAIL();
  }

  // performance
  prof.print_profiling_result(profiling_selector::GPU);
  // check result
  std::vector<typename Test::data_type_a> dequantize_up_proj =
      dequantize_weight<dequant_s, layout_b, compute_policy::quant_mode>(
          matrix_k, matrix_n, up_proj_h, scale_up_proj_h, zero_pt_up_proj_h);
  std::vector<typename Test::data_type_a> dequantize_gate_proj =
      dequantize_weight<dequant_s, layout_b, compute_policy::quant_mode>(
          matrix_k,
          matrix_n,
          gate_proj_h,
          scale_gate_proj_h,
          zero_pt_gate_proj_h);

  queue.memcpy((void*)C_h, (void*)C_d, size_c * sizeof(data_type_c)).wait();
  ASSERT_EQ(
      0,
      int4_mlp_result_validate(
          A_h,
          dequantize_up_proj.data(),
          dequantize_gate_proj.data(),
          C_h,
          matrix_m,
          matrix_k,
          matrix_n,
          layout_a,
          layout_b));

  free(A_h, context);
  free(up_proj_h, context);
  free(gate_proj_h, context);
  free(C_h, context);
  free(scale_up_proj_h, context);
  free(scale_gate_proj_h, context);
  free(zero_pt_up_proj_h, context);
  free(zero_pt_gate_proj_h, context);
  free(A_d, context);
  free(up_proj_d, context);
  free(gate_proj_d, context);
  free(C_d, context);
  free(scale_up_proj_d, context);
  free(scale_gate_proj_d, context);
  free(zero_pt_up_proj_d, context);
  free(zero_pt_gate_proj_d, context);
  free(Acc_up_proj_h, context);
  free(Acc_gate_proj_h, context);
  free(Cnt_h, context);
  free(Acc_up_proj_d, context);
  free(Acc_gate_proj_d, context);
  free(Cnt_d, context);
}

template <typename T>
class int4_mlp_test : public ::testing::Test {};
TYPED_TEST_SUITE_P(int4_mlp_test);

TYPED_TEST_P(int4_mlp_test, esimd) {
  dequantize_int4_mlp_run<TypeParam>(ITER);
}

REGISTER_TYPED_TEST_SUITE_P(int4_mlp_test, esimd);
using tests = ::testing::Types<test_col_major_1>;

INSTANTIATE_TYPED_TEST_SUITE_P(int4_mlp_test_suite, int4_mlp_test, tests);
