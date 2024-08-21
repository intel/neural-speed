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
// #define UT_DEBUG
using namespace gpu::xetla;
using namespace gpu::xetla::group;
// The number of times the kernel is executed
#ifdef UT_DEBUG
constexpr int ITER = 1;
#else
constexpr int ITER = 200;
#endif
constexpr size_t UNDEFINED_DATA_SIZE = 512;

class test_col_major_1 {
 public:
  // Extract the parameters required by different test cases
  static constexpr size_t mat_n = 4096;
  static constexpr size_t mat_k = 4096;
  static constexpr size_t wg_k = 128;
  static constexpr size_t wg_n = 16;
  static constexpr size_t sg_k = 128;
  static constexpr size_t sg_n = 16;
  static constexpr size_t k_stride = 32;
  static constexpr size_t dequant_s = 128;
  // static constexpr quant_mode quant_mode = quant_mode::I4_ASYM;
  // static constexpr quant_mode quant_mode = quant_mode::I4_SYM;
  // static constexpr quant_mode quant_mode = quant_mode::DEGREE5_APPROX_NF4;
  static constexpr quant_mode quant_mode = quant_mode::NF4;

  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = mem_layout::col_major;
  static constexpr gpu_arch arch = gpu_arch::XeHpg;
  using data_type_b = int4x8;
  using data_type_c = fp16;
};

template <
    quant_mode quant_mode = quant_mode::I4_SYM,
    typename data_type_acc_in = fp16,
    typename data_type_b,
    typename data_type_scale,
    typename data_type_zero_pt>
std::vector<fp16> convert_bit4(
    data_type_b data_b,
    data_type_scale scale,
    data_type_zero_pt zero_pt) {
  std::vector<fp16> dequant_fp16(sizeof(data_type_b) * 2);

  float nf4_LUT alignas(64)[] = {
      -1.f,
      -0.6961928009986877f,
      -0.5250730514526367f,
      -0.39491748809814453f,
      -0.28444138169288635f,
      -0.18477343022823334f,
      -0.09105003625154495f,
      0.f,
      0.07958029955625534f,
      0.16093020141124725f,
      0.24611230194568634f,
      0.33791524171829224f,
      0.44070982933044434f,
      0.5626170039176941f,
      0.7229568362236023f,
      1.0f};

  int8_t zero_pt_i8 = zero_pt & 0xf;
  for (uint32_t i = 0; i < dequant_fp16.size(); i++) {
    int8_t dequant_8bit = data_b & 0xf;
    if constexpr (quant_mode == quant_mode::I4_SYM) {
      dequant_fp16[i] = scale * (dequant_8bit - 8);
    } else if constexpr (quant_mode == quant_mode::I4_ASYM) {
      dequant_fp16[i] = scale * (dequant_8bit - zero_pt_i8);
    } else if constexpr (quant_mode == quant_mode::DEGREE5_APPROX_NF4) {
      float tmp = 1.831e-05;
      tmp = tmp * dequant_8bit - 0.0006863;
      tmp = tmp * dequant_8bit + 0.01005;
      tmp = tmp * dequant_8bit - 0.07231;
      tmp = tmp * dequant_8bit + 0.3462;
      tmp = tmp * dequant_8bit - 0.9942;
      dequant_fp16[i] = scale * tmp;
    } else if constexpr (quant_mode == quant_mode::NF4) {
      dequant_fp16[i] = scale * nf4_LUT[dequant_8bit];
    } else {
      assert(0);
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
        std::vector<fp16> dequant_fp16 = convert_bit4<quant_mode>(
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

template <typename T>
int int4_dequantize_result_validate(T* gold, T* out, size_t k, size_t n) {
  int err_num = 0;
  for (uint32_t i = 0; i < k; i++) {
    for (uint32_t j = 0; j < n; j++) {
      if (gold[i * n + j] != out[i * n + j]) {
        if (err_num < 10)
          std::cout << i * n + j << " " << gold[i * n + j] << " "
                    << out[i * n + j] << std::endl;
        err_num++;
      }
    }
  }
  if (err_num == 0) {
    std::cout << "Test Passed!!!" << std::endl;
  }
  return err_num;
}

template <class Test>
void dequantize_run(int iter) {
  using namespace gpu;
  // Accept incoming parameters
  constexpr size_t matrix_n = Test::mat_n;
  constexpr size_t matrix_k = Test::mat_k;

  constexpr size_t wg_tile_n = Test::wg_n;
  constexpr size_t wg_tile_k = Test::wg_k;
  constexpr size_t sg_tile_n = Test::sg_n;
  constexpr size_t sg_tile_k = Test::sg_k;
  constexpr size_t k_stride = Test::k_stride;
  constexpr size_t dequant_s = std::min(Test::dequant_s, matrix_k);
  constexpr quant_mode quant_mode = Test::quant_mode;
  using data_type_b = typename Test::data_type_b;
  using data_type_c = typename Test::data_type_c;
  using data_type_zero_pt = data_type_b;
  using data_type_scale = fp16;

  constexpr mem_layout layout_b = Test::layout_b;

  constexpr size_t size_b = matrix_k * matrix_n / (2 * sizeof(data_type_b));

  constexpr size_t size_scale_k = matrix_k / dequant_s;
  constexpr size_t size_scale_n = matrix_n;
  constexpr size_t size_scale = size_scale_k * size_scale_n;

  constexpr size_t size_zero_pt_k = matrix_k / dequant_s;
  constexpr size_t size_zero_pt_n = matrix_n;
  constexpr size_t size_zero_pt =
      size_zero_pt_k * size_zero_pt_n / (2 * sizeof(data_type_b));

  constexpr size_t size_c = matrix_k * matrix_n;

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

  using bit4_dequantize_attr = gpu::xetla::kernel::bit4_dequantize_attr_t<
      wg_tile_n,
      wg_tile_k,
      sg_tile_n,
      sg_tile_k,
      k_stride>;
  static constexpr quant_info q_info{quant_mode, Test::dequant_s, layout_b};
  using int4_dequantize_kernel = gpu::xetla::kernel::int4_dequantize_t<
      data_type_b,
      data_type_scale,
      data_type_zero_pt,
      data_type_c,
      layout_b,
      layout_b,
      mem_layout::row_major,
      mem_layout::row_major,
      q_info,
      bit4_dequantize_attr,
      Test::arch>;

  using f4_dequantize_kernel = gpu::xetla::kernel::f4_dequantize_t<
      data_type_b,
      data_type_scale,
      data_type_c,
      layout_b,
      layout_b,
      mem_layout::row_major,
      q_info,
      bit4_dequantize_attr,
      Test::arch>;

  // Define and initialize the data required for the calculation
  auto* B_h = static_cast<data_type_b*>(malloc_host(
      (size_b + UNDEFINED_DATA_SIZE) * sizeof(data_type_b), context));
  auto* C_h = static_cast<data_type_c*>(
      malloc_host(size_c * sizeof(data_type_c), context));
  auto* scale_h = static_cast<data_type_scale*>(malloc_host(
      (size_scale + UNDEFINED_DATA_SIZE) * sizeof(data_type_scale), context));
  auto* zero_pt_h = static_cast<data_type_zero_pt*>(malloc_host(
      (size_zero_pt + UNDEFINED_DATA_SIZE) * sizeof(data_type_zero_pt),
      context));

  auto* B_d = static_cast<data_type_b*>(aligned_alloc_device(
      DEVICE_MEM_ALIGNMENT,
      (size_b + UNDEFINED_DATA_SIZE) * sizeof(data_type_b),
      device,
      context));
  auto* C_d = static_cast<data_type_c*>(aligned_alloc_device(
      DEVICE_MEM_ALIGNMENT, size_c * sizeof(data_type_c), device, context));
  auto* scale_d = static_cast<data_type_scale*>(aligned_alloc_device(
      DEVICE_MEM_ALIGNMENT,
      (size_scale + UNDEFINED_DATA_SIZE) * sizeof(data_type_scale),
      device,
      context));
  auto* zero_pt_d = static_cast<data_type_zero_pt*>(aligned_alloc_device(
      DEVICE_MEM_ALIGNMENT,
      (size_zero_pt + UNDEFINED_DATA_SIZE) * sizeof(data_type_zero_pt),
      device,
      context));

  for (unsigned i = 0; i < size_b + UNDEFINED_DATA_SIZE; ++i) {
    if constexpr (std::is_same_v<int4x2, data_type_b>) {
      B_h[i] = random_uint8();
#ifdef UT_DEBUG
      B_h[i] = 0x77;
#endif
    } else if constexpr (std::is_same_v<int4x8, data_type_b>) {
      B_h[i] = random_uint32();
#ifdef UT_DEBUG
      B_h[i] = i < size_b / 2 ? 0x77777777 : 0x66666666;
#endif
    }
  }

  for (unsigned i = 0; i < size_scale; ++i) {
    scale_h[i] = random_float() + 1.f;
#ifdef UT_DEBUG
    scale_h[i] = 1.f;
#endif
  }
  for (unsigned i = size_scale; i < size_scale + UNDEFINED_DATA_SIZE; ++i) {
    scale_h[i] = INFINITY;
  }
  for (unsigned i = 0; i < size_zero_pt + UNDEFINED_DATA_SIZE; ++i) {
    if constexpr (std::is_same_v<int4x2, data_type_b>) {
      zero_pt_h[i] = random_uint8();
#ifdef UT_DEBUG
      zero_pt_h[i] = 0x12 << i;
#endif
    } else if constexpr (std::is_same_v<int4x8, data_type_b>) {
      zero_pt_h[i] = random_uint32();
#ifdef UT_DEBUG
      zero_pt_h[i] = 0x33333333;
#endif
    }
  }

  for (unsigned i = 0; i < size_c; ++i) {
    C_h[i] = random_float();
  }

  queue
      .memcpy(
          (void*)B_d,
          (void*)B_h,
          (size_b + UNDEFINED_DATA_SIZE) * sizeof(data_type_b))
      .wait();
  queue.memcpy((void*)C_d, (void*)C_h, size_c * sizeof(data_type_c)).wait();
  queue
      .memcpy(
          (void*)scale_d,
          (void*)scale_h,
          (size_scale + UNDEFINED_DATA_SIZE) * sizeof(data_type_scale))
      .wait();
  queue
      .memcpy(
          (void*)zero_pt_d,
          (void*)zero_pt_h,
          (size_zero_pt + UNDEFINED_DATA_SIZE) * sizeof(data_type_zero_pt))
      .wait();

  typename int4_dequantize_kernel::arguments_t i4_args(
      matrix_k,
      matrix_n,
      B_d,
      scale_d,
      zero_pt_d,
      C_d,
      ldb,
      ldc,
      ld_scale,
      ld_zero_pt);

  typename f4_dequantize_kernel::arguments_t f4_args(
      matrix_k, matrix_n, B_d, scale_d, C_d, ldb, ldc, ld_scale);

  cl::sycl::nd_range<3> nd_range =
      int4_dequantize_kernel::get_nd_range(matrix_k, matrix_n);

  size_t bytes = matrix_n * matrix_k / 2 +
      matrix_k * matrix_n * sizeof(data_type_c) +
      size_scale * sizeof(data_type_scale);
  if (Test::quant_mode == quant_mode::I4_ASYM)
    bytes += size_zero_pt * sizeof(data_type_zero_pt);
  profiling_helper prof("int4_dequantize kernel bandwidth", bytes, "GB/s");
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
          if constexpr (
              quant_mode == quant_mode::I4_SYM ||
              quant_mode == quant_mode::I4_ASYM) {
            int4_dequantize_kernel::call(item, i4_args);
          } else {
            f4_dequantize_kernel::call(item, f4_args);
          }
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
  std::vector<typename Test::data_type_c> dequantize_b =
      dequantize_weight<dequant_s, layout_b, Test::quant_mode>(
          matrix_k, matrix_n, B_h, scale_h, zero_pt_h);
  std::vector<typename Test::data_type_c> trans_dq_b;
  trans_dq_b.resize(matrix_n * matrix_k);
  // transpose dq b
  for (size_t i = 0; i < matrix_n; i++) {
    for (size_t j = 0; j < matrix_k; j++) {
      trans_dq_b[j * matrix_n + i] = dequantize_b[i * matrix_k + j];
    }
  }

  queue.memcpy((void*)C_h, (void*)C_d, size_c * sizeof(data_type_c)).wait();

  ASSERT_EQ(
      0,
      (int4_dequantize_result_validate<data_type_c>(
          trans_dq_b.data(), C_h, Test::mat_k, Test::mat_n)));

  free(B_h, context);
  free(C_h, context);
  free(scale_h, context);
  free(zero_pt_h, context);
  free(B_d, context);
  free(C_d, context);
  free(scale_d, context);
  free(zero_pt_d, context);
}

template <typename T>
class dequantize_test : public ::testing::Test {};
TYPED_TEST_SUITE_P(dequantize_test);

TYPED_TEST_P(dequantize_test, esimd) {
  dequantize_run<TypeParam>(ITER);
}

REGISTER_TYPED_TEST_SUITE_P(dequantize_test, esimd);
using tests = ::testing::Types<test_col_major_1>;

INSTANTIATE_TYPED_TEST_SUITE_P(dequantize_test_suite, dequantize_test, tests);
