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

#pragma once

#include <utils/buff_compare.hpp>
#include <utils/common.hpp>
#include "kernel_func.hpp"
// #include <gtest/gtest.h>

class TestBase {
 public:
  static std::string name(
      size_t mat_m,
      size_t mat_n,
      size_t mat_k,
      size_t wg_m,
      size_t wg_n,
      size_t sg_m,
      size_t sg_n,
      [[maybe_unused]] size_t sg_k,
      mem_layout layout_a,
      mem_layout layout_b) {
    std::string mem_layout_a_str =
        layout_a == mem_layout::col_major ? "col_major" : "row_major";
    std::string mem_layout_b_str =
        layout_b == mem_layout::col_major ? "col_major" : "row_major";
    std::string name = std::string("bgemm_") + std::to_string(mat_m) + "x" +
        std::to_string(mat_n) + "x" + std::to_string(mat_k) + "_" +
        std::to_string(wg_m) + "x" + std::to_string(wg_n) + "_" +
        std::to_string(sg_m) + "x" + std::to_string(sg_n) + "_" +
        mem_layout_a_str + "_" + mem_layout_b_str;
    return name;
  }

  static constexpr gpu_arch gpu_arch = gpu_arch::XeHpc;
  //static constexpr gpu_arch gpu_arch = gpu_arch::XeHpg;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr uint32_t wg_num_n = 64;
};

class TestBaseBF16x : public TestBase {
 public:
  using data_type_a = bf16;
  using data_type_b = bf16;
  using data_type_c = bf16;
  using data_type_acc = float;
  static constexpr mma_engine engine = mma_engine::xmx;
};

class TestBaseBF16f : public TestBase {
 public:
  using data_type_a = bf16;
  using data_type_b = bf16;
  using data_type_c = bf16;
  using data_type_acc = float;
  static constexpr mma_engine engine = mma_engine::fpu;
};

class Test0x : public TestBaseBF16x {
 public:
  static constexpr size_t mat_m = 256;
  static constexpr size_t mat_n = 256;
  static constexpr size_t mat_k = 256;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = mem_layout::row_major;
};

class Test1x : public TestBaseBF16x {
 public:
  static constexpr size_t mat_m = 256;
  static constexpr size_t mat_n = 256;
  static constexpr size_t mat_k = 256;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 16;
  static constexpr mem_layout layout_a = mem_layout::col_major;
  static constexpr mem_layout layout_b = mem_layout::row_major;
};

class Test2x : public TestBaseBF16x {
 public:
  static constexpr size_t mat_m = 256;
  static constexpr size_t mat_n = 256;
  static constexpr size_t mat_k = 256;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = mem_layout::col_major;
};

class Test3x : public TestBaseBF16x {
 public:
  static constexpr size_t mat_m = 256;
  static constexpr size_t mat_n = 256;
  static constexpr size_t mat_k = 256;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 16;
  static constexpr mem_layout layout_a = mem_layout::col_major;
  static constexpr mem_layout layout_b = mem_layout::col_major;
};

class Test4x : public TestBaseBF16x {
 public:
  static constexpr size_t mat_m = 256;
  static constexpr size_t mat_n = 256;
  static constexpr size_t mat_k = 256;
  static constexpr size_t wg_m = 16;
  static constexpr size_t wg_n = 32;
  static constexpr size_t sg_m = 8;
  static constexpr size_t sg_n = 16;
  static constexpr size_t sg_k = 16;
  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = mem_layout::row_major;
  using data_type_a = bf16;
  using data_type_b = bf16;
  using data_type_c = float;
  using data_type_acc = float;
};

class Test5x : public TestBaseBF16x {
 public:
  static constexpr size_t mat_m = 192;
  static constexpr size_t mat_n = 256;
  static constexpr size_t mat_k = 256;
  // static constexpr size_t wg_m = 48;
  // If want ot allow any kind of wg_m and wg_n instead of the power of 2
  // DG2 still need check workgroup oob on both direction by using block_1d load
  static constexpr size_t wg_m = 64;
  // static constexpr size_t wg_n = 80;
  static constexpr size_t wg_n = 128;
  // static constexpr size_t sg_m = 24;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr mem_layout layout_a = mem_layout::col_major;
  static constexpr mem_layout layout_b = mem_layout::row_major;
  using data_type_a = bf16;
  using data_type_b = bf16;
  using data_type_c = float;
  using data_type_acc = float;
};

class Test6x : public TestBaseBF16x {
 public:
  static constexpr size_t mat_m = 96;
  static constexpr size_t mat_n = 256;
  static constexpr size_t mat_k = 256;
  // static constexpr size_t wg_m = 40;
  static constexpr size_t wg_m = 64;
  static constexpr size_t wg_n = 256;
  // static constexpr size_t sg_m = 24;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = mem_layout::row_major;
  using data_type_a = bf16;
  using data_type_b = bf16;
  using data_type_c = float;
  using data_type_acc = float;
};

class Test7x : public TestBaseBF16x {
 public:
  static constexpr size_t mat_m = 80;
  static constexpr size_t mat_n = 256;
  static constexpr size_t mat_k = 256;
  static constexpr size_t wg_m = 96;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = mem_layout::row_major;
};

class Test8x : public TestBaseBF16x {
 public:
  static constexpr size_t mat_m = 256;
  static constexpr size_t mat_n = 256;
  static constexpr size_t mat_k = 256;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr uint32_t global_kslicing = 2;
  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = mem_layout::row_major;
  using data_type_a = bf16;
  using data_type_b = bf16;
  using data_type_c = float;
  using data_type_acc = float;
};

class Test9x : public TestBaseBF16x {
 public:
  static constexpr size_t mat_m = 256;
  static constexpr size_t mat_n = 256;
  static constexpr size_t mat_k = 256;
  static constexpr size_t wg_m = 128;
  static constexpr size_t wg_n = 128;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 32;
  static constexpr size_t sg_k = 32;
  // static constexpr uint32_t local_kslicing = 2;
  // Look like local_kslicing will fail on DG2
  static constexpr uint32_t local_kslicing = 1;
  // global_kslicing work for aligned case on DG2
  static constexpr uint32_t global_kslicing = 4;
  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = mem_layout::row_major;
  using data_type_a = bf16;
  using data_type_b = bf16;
  using data_type_c = bf16;
  using data_type_acc = float;
};

class Test10x : public TestBaseBF16x {
 public:
  static constexpr size_t mat_m = 256;
  static constexpr size_t mat_n = 256;
  static constexpr size_t mat_k = 256;
  static constexpr size_t wg_m = 64;
  static constexpr size_t wg_n = 128;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 32;
  static constexpr size_t sg_k = 32;
  // static constexpr uint32_t local_kslicing = 4;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = mem_layout::row_major;
  using data_type_a = bf16;
  using data_type_b = bf16;
  using data_type_c = bf16;
  using data_type_acc = float;
};

class Test11x : public TestBaseBF16x {
 public:
  static constexpr size_t mat_m = 256;
  static constexpr size_t mat_n = 256;
  static constexpr size_t mat_k = 1024;
  static constexpr size_t wg_m = 8;
  static constexpr size_t wg_n = 32;
  static constexpr size_t sg_m = 8;
  static constexpr size_t sg_n = 32;
  static constexpr size_t sg_k = 32;
  static constexpr uint32_t local_kslicing = 16;
  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = mem_layout::row_major;
};

class Test12x : public TestBaseBF16x {
 public:
  static constexpr size_t mat_m = 1;
  static constexpr size_t mat_n = 13824;
  static constexpr size_t mat_k = 5120;
  static constexpr size_t wg_m = 1;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 1;
  static constexpr size_t sg_n = 32;
  static constexpr size_t sg_k = 32;
  static constexpr uint32_t local_kslicing = 4;
  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = mem_layout::row_major;
};

class Test13x : public TestBaseBF16x {
 public:
  static constexpr size_t mat_m = 8192;
  static constexpr size_t mat_n = 8192;
  static constexpr size_t mat_k = 8192;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = mem_layout::row_major;
};

class Test14x : public TestBaseBF16x {
 public:
  static constexpr size_t mat_m = 4096;
  static constexpr size_t mat_n = 4096;
  static constexpr size_t mat_k = 4096;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = mem_layout::row_major;
};

class Test15x : public TestBaseBF16x {
 public:
  static constexpr size_t mat_m = 1024;
  static constexpr size_t mat_n = 28672;
  static constexpr size_t mat_k = 8192;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = mem_layout::row_major;
};

class Test16x : public TestBaseBF16x {
 public:
  static constexpr size_t mat_m = 4;
  static constexpr size_t mat_n = 12288;
  static constexpr size_t mat_k = 4096;
  static constexpr size_t wg_m = 8; // wg_m = 4 will fail on DG2
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 8;
  static constexpr size_t sg_n = 32;
  static constexpr size_t sg_k = 32;
  static constexpr uint32_t local_kslicing = 4;
  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = mem_layout::row_major;
};

class Test17x : public TestBaseBF16x {
 public:
  static constexpr size_t mat_m = 3072;
  static constexpr size_t mat_n = 3072;
  static constexpr size_t mat_k = 4096;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = mem_layout::row_major;
};

class Test18x : public TestBaseBF16x { // Get better perf on DG2, ~15.48 TFlops
 public:
  static constexpr size_t mat_m = 4096;
  static constexpr size_t mat_n = 4096;
  static constexpr size_t mat_k = 4096;
  static constexpr size_t wg_m = 32;
  static constexpr size_t wg_n = 512;
  static constexpr size_t sg_m = 16;
  static constexpr size_t sg_n = 32;
  static constexpr size_t sg_k = 16;
  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = mem_layout::row_major;
};

template <class Test>
class result_validate {
 public:
  using dtype_a = typename Test::data_type_a;
  using dtype_b = typename Test::data_type_b;
  using dtype_c = typename Test::data_type_c;
  using dtype_acc = typename Test::data_type_acc;

  int operator()(dtype_a* A, dtype_b* B, dtype_c* C, sycl::queue& queue) {
    return gemm_result_validate<dtype_a, dtype_b, dtype_c, dtype_acc>(
        A,
        B,
        C,
        1,
        Test::mat_m,
        Test::mat_k,
        Test::mat_n,
        queue,
        Test::layout_a,
        Test::layout_b);
  }
};

template <class Test>
using bf16_gemm_func = bf16_gemm_test_func<
    typename Test::data_type_a,
    typename Test::data_type_b,
    typename Test::data_type_c,
    typename Test::data_type_acc,
    Test::wg_m,
    Test::wg_n,
    Test::sg_m,
    Test::sg_n,
    Test::sg_k,
    Test::layout_a,
    Test::layout_b,
    Test::global_kslicing,
    Test::local_kslicing,
    Test::wg_num_n,
    Test::engine,
    Test::gpu_arch>;
