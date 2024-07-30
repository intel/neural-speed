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

#include <gtest/gtest.h>
#include <utils/buff_compare.hpp>
#include <utils/common.hpp>
#include "kernel_func.hpp"

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

  static constexpr gpu_arch arch_tag = gpu_arch::XeHpc;
  //static constexpr gpu_arch arch_tag = gpu_arch::XeHpg;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr uint32_t lda_alignment = 1;
  static constexpr uint32_t ldb_alignment = 1;
  static constexpr uint32_t ldc_alignment = 1;
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

class TestBaseFP16x : public TestBase {
 public:
  using data_type_a = fp16;
  using data_type_b = fp16;
  using data_type_c = fp16;
  using data_type_acc = float;
  static constexpr mma_engine engine = mma_engine::xmx;
};

class TestBaseFP16f : public TestBase {
 public:
  using data_type_a = fp16;
  using data_type_b = fp16;
  using data_type_c = fp16;
  using data_type_acc = float;
  static constexpr mma_engine engine = mma_engine::fpu;
};

class Test0x : public TestBaseBF16x {
 public:
  static constexpr size_t mat_m = 253;
  static constexpr size_t mat_n = 257;
  static constexpr size_t mat_k = 255;
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
  static constexpr size_t mat_m = 253;
  static constexpr size_t mat_n = 1023;
  static constexpr size_t mat_k = 767;
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
  static constexpr size_t mat_m = 253;
  static constexpr size_t mat_n = 1011;
  static constexpr size_t mat_k = 511;
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
  static constexpr size_t mat_m = 253;
  static constexpr size_t mat_n = 767;
  static constexpr size_t mat_k = 1023;
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
  static constexpr size_t mat_m = 257;
  static constexpr size_t mat_n = 257;
  static constexpr size_t mat_k = 256;
  static constexpr size_t wg_m = 16;
  static constexpr size_t wg_n = 32;
  static constexpr size_t sg_m = 8;
  static constexpr size_t sg_n = 16;
  static constexpr size_t sg_k = 16;
  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = mem_layout::row_major;
  static constexpr uint32_t lda_alignment = 8;
};

class Test5x : public TestBaseBF16x {
 public:
  static constexpr size_t mat_m = 191;
  static constexpr size_t mat_n = 251;
  static constexpr size_t mat_k = 253;
  static constexpr size_t wg_m = 48;
  static constexpr size_t wg_n = 80;
  static constexpr size_t sg_m = 24;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = mem_layout::row_major;
  using data_type_c = float;
};

class Test6x : public TestBaseBF16x {
 public:
  static constexpr size_t mat_m = 93;
  static constexpr size_t mat_n = 253;
  static constexpr size_t mat_k = 251;
  static constexpr size_t wg_m = 40;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 24;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = mem_layout::col_major;
  using data_type_c = float;
};

class Test7x : public TestBaseBF16x {
 public:
  static constexpr size_t mat_m = 80;
  static constexpr size_t mat_n = 251;
  static constexpr size_t mat_k = 253;
  static constexpr size_t wg_m = 128;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = mem_layout::row_major;
  using data_type_c = float;
};

class Test8x : public TestBaseBF16x {
 public:
  static constexpr size_t mat_m = 257;
  static constexpr size_t mat_n = 255;
  static constexpr size_t mat_k = 253;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  // static constexpr uint32_t global_kslicing = 2; //will compile fail on DG2
  static constexpr uint32_t global_kslicing = 1;
  static constexpr mem_layout layout_a = mem_layout::col_major;
  static constexpr mem_layout layout_b = mem_layout::row_major;
  using data_type_c = float;
};

class Test9x : public TestBaseBF16x {
 public:
  static constexpr size_t mat_m = 251;
  static constexpr size_t mat_n = 253;
  static constexpr size_t mat_k = 255;
  static constexpr size_t wg_m = 128;
  static constexpr size_t wg_n = 128;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 32;
  static constexpr size_t sg_k = 32;
  // static constexpr uint32_t global_kslicing = 4; //will compile fail on DG2
  static constexpr uint32_t global_kslicing = 1;
  static constexpr mem_layout layout_a = mem_layout::col_major;
  static constexpr mem_layout layout_b = mem_layout::col_major;
};

class Test10x : public TestBaseBF16x {
 public:
  static constexpr size_t mat_m = 1;
  static constexpr size_t mat_n = 253;
  static constexpr size_t mat_k = 259;
  static constexpr size_t wg_m = 32;
  static constexpr size_t wg_n = 64;
  static constexpr size_t sg_m = 8;
  static constexpr size_t sg_n = 16;
  static constexpr size_t sg_k = 16;
  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = mem_layout::row_major;
};

class Test11x : public TestBaseBF16x {
 public:
  static constexpr size_t mat_m = 1;
  static constexpr size_t mat_n = 1025;
  static constexpr size_t mat_k = 256;
  static constexpr size_t wg_m = 8;
  static constexpr size_t wg_n = 64;
  static constexpr size_t sg_m = 8;
  static constexpr size_t sg_n = 16;
  static constexpr size_t sg_k = 32;
  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = mem_layout::col_major;
  static constexpr uint32_t lda_alignment = 8;
};

class Test12x : public TestBaseBF16x {
 public:
  static constexpr size_t mat_m = 4095;
  static constexpr size_t mat_n = 4097;
  static constexpr size_t mat_k = 4091;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = mem_layout::row_major;
};

class Test13x : public TestBaseBF16x {
 public:
  static constexpr size_t mat_m = 4096;
  static constexpr size_t mat_n = 4096;
  static constexpr size_t mat_k = 4095;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = mem_layout::row_major;
  static constexpr uint32_t ldb_alignment = 8;
  static constexpr uint32_t ldc_alignment = 8;
};

class Test14x : public TestBaseBF16x {
 public:
  static constexpr size_t mat_m = 4096;
  static constexpr size_t mat_n = 4097;
  static constexpr size_t mat_k = 4096;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = mem_layout::row_major;
  static constexpr uint32_t lda_alignment = 8;
};

class Test15x : public TestBaseBF16x {
 public:
  static constexpr size_t mat_m = 4096;
  static constexpr size_t mat_n = 4096;
  static constexpr size_t mat_k = 4096;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 16;
  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = mem_layout::row_major;
  static constexpr uint32_t lda_alignment = 8;
  static constexpr uint32_t ldb_alignment = 8;
  static constexpr uint32_t ldc_alignment = 8;
};

class Test16x : public TestBaseBF16x { // Get better perf on DG2
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
  static constexpr uint32_t lda_alignment = 8;
  static constexpr uint32_t ldb_alignment = 8;
  static constexpr uint32_t ldc_alignment = 8;
};

class Test17x : public TestBaseFP16x {
 public:
  static constexpr size_t mat_m = 4096;
  static constexpr size_t mat_n = 4096;
  static constexpr size_t mat_k = 4096;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 16;
  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = mem_layout::col_major;
  static constexpr uint32_t lda_alignment = 8;
  static constexpr uint32_t ldb_alignment = 8;
  static constexpr uint32_t ldc_alignment = 8;
};

class Test18x : public TestBaseFP16x {
 public:
  static constexpr size_t mat_m = 1024;
  static constexpr size_t mat_n = 2560;
  static constexpr size_t mat_k = 5120;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 16;
  static constexpr mem_layout layout_a = mem_layout::col_major;
  static constexpr mem_layout layout_b = mem_layout::row_major;
  static constexpr uint32_t lda_alignment = 8;
  static constexpr uint32_t ldb_alignment = 8;
  static constexpr uint32_t ldc_alignment = 8;
};

class Test19x : public TestBaseBF16x {
 public:
  static constexpr size_t mat_m = 4;
  static constexpr size_t mat_n = 12288;
  static constexpr size_t mat_k = 4096;
  static constexpr size_t wg_m = 8;   //DG@ will fail on wg_m = 4
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 8;
  static constexpr size_t sg_n = 32;
  static constexpr size_t sg_k = 16;
  static constexpr uint32_t local_kslicing = 4;
  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = mem_layout::row_major;
  static constexpr uint32_t lda_alignment = 8;
  static constexpr uint32_t ldb_alignment = 8;
  static constexpr uint32_t ldc_alignment = 8;
};

class Test19f : public TestBaseBF16f {
 public:
  static constexpr size_t mat_m = 4;
  static constexpr size_t mat_n = 12288;
  static constexpr size_t mat_k = 4096;
  static constexpr size_t wg_m = 4;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 4;
  static constexpr size_t sg_n = 32;
  static constexpr size_t sg_k = 16;
  static constexpr uint32_t local_kslicing = 4;
  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = mem_layout::row_major;
  static constexpr uint32_t lda_alignment = 8;
  static constexpr uint32_t ldb_alignment = 8;
  static constexpr uint32_t ldc_alignment = 8;
};

class Test20f : public TestBaseFP16f {
 public:
  static constexpr size_t mat_m = 1;
  static constexpr size_t mat_n = 4096;
  static constexpr size_t mat_k = 4096;
  static constexpr size_t wg_m = 1;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 1;
  static constexpr size_t sg_n = 32;
  static constexpr size_t sg_k = 16;
  static constexpr uint32_t local_kslicing = 4;
  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = mem_layout::row_major;
  static constexpr uint32_t lda_alignment = 8;
  static constexpr uint32_t ldb_alignment = 8;
  static constexpr uint32_t ldc_alignment = 8;
};

class Test20x : public TestBaseFP16x {
 public:
  static constexpr size_t mat_m = 1;
  static constexpr size_t mat_n = 4096;
  static constexpr size_t mat_k = 4096;
  static constexpr size_t wg_m = 1;
  static constexpr size_t wg_n = 64;
  static constexpr size_t sg_m = 1;
  static constexpr size_t sg_n = 16;
  static constexpr size_t sg_k = 16;
  static constexpr uint32_t local_kslicing = 8;
  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = mem_layout::row_major;
  static constexpr uint32_t lda_alignment = 8;
  static constexpr uint32_t ldb_alignment = 8;
  static constexpr uint32_t ldc_alignment = 8;
};

class Test21x : public TestBaseFP16x {
 public:
  static constexpr size_t mat_m = 4;
  static constexpr size_t mat_n = 4096;
  static constexpr size_t mat_k = 4096;
  static constexpr size_t wg_m = 4;
  static constexpr size_t wg_n = 64;
  static constexpr size_t sg_m = 4;
  static constexpr size_t sg_n = 16;
  static constexpr size_t sg_k = 16;
  static constexpr uint32_t local_kslicing = 8;
  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = mem_layout::row_major;
  static constexpr uint32_t lda_alignment = 8;
  static constexpr uint32_t ldb_alignment = 8;
  static constexpr uint32_t ldc_alignment = 8;
};

class Test21f : public TestBaseFP16f {
 public:
  static constexpr size_t mat_m = 4;
  static constexpr size_t mat_n = 4096;
  static constexpr size_t mat_k = 4096;
  static constexpr size_t wg_m = 4;
  static constexpr size_t wg_n = 64;
  static constexpr size_t sg_m = 4;
  static constexpr size_t sg_n = 16;
  static constexpr size_t sg_k = 16;
  static constexpr uint32_t local_kslicing = 8;
  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = mem_layout::row_major;
  static constexpr uint32_t lda_alignment = 8;
  static constexpr uint32_t ldb_alignment = 8;
  static constexpr uint32_t ldc_alignment = 8;
};

class Test22f : public TestBaseFP16f {
 public:
  static constexpr size_t mat_m = 1;
  static constexpr size_t mat_n = 4096;
  static constexpr size_t mat_k = 4096;
  static constexpr size_t wg_m = 1;
  static constexpr size_t wg_n = 64;
  static constexpr size_t sg_m = 1;
  static constexpr size_t sg_n = 16;
  static constexpr size_t sg_k = 16;
  static constexpr uint32_t local_kslicing = 8;
  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = mem_layout::col_major;
  static constexpr uint32_t lda_alignment = 8;
  static constexpr uint32_t ldb_alignment = 8;
  static constexpr uint32_t ldc_alignment = 8;
};

class Test22x : public TestBaseFP16x {
 public:
  static constexpr size_t mat_m = 1;
  static constexpr size_t mat_n = 4096;
  static constexpr size_t mat_k = 4096;
  static constexpr size_t wg_m = 1;
  static constexpr size_t wg_n = 64;
  static constexpr size_t sg_m = 1;
  static constexpr size_t sg_n = 16;
  static constexpr size_t sg_k = 16;
  static constexpr uint32_t local_kslicing = 8;
  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = mem_layout::col_major;
  static constexpr uint32_t lda_alignment = 8;
  static constexpr uint32_t ldb_alignment = 8;
  static constexpr uint32_t ldc_alignment = 8;
};

class Test23f : public TestBaseFP16f {
 public:
  static constexpr size_t mat_m = 4;
  static constexpr size_t mat_n = 4096;
  static constexpr size_t mat_k = 4096;
  static constexpr size_t wg_m = 4;
  static constexpr size_t wg_n = 64;
  static constexpr size_t sg_m = 4;
  static constexpr size_t sg_n = 16;
  static constexpr size_t sg_k = 16;
  static constexpr uint32_t local_kslicing = 8;
  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = mem_layout::col_major;
  static constexpr uint32_t lda_alignment = 8;
  static constexpr uint32_t ldb_alignment = 8;
  static constexpr uint32_t ldc_alignment = 8;
};

class Test23x : public TestBaseFP16x {
 public:
  static constexpr size_t mat_m = 4;
  static constexpr size_t mat_n = 4096;
  static constexpr size_t mat_k = 4096;
  static constexpr size_t wg_m = 4;
  static constexpr size_t wg_n = 64;
  static constexpr size_t sg_m = 4;
  static constexpr size_t sg_n = 16;
  static constexpr size_t sg_k = 16;
  static constexpr uint32_t local_kslicing = 8;
  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = mem_layout::col_major;
  static constexpr uint32_t lda_alignment = 8;
  static constexpr uint32_t ldb_alignment = 8;
  static constexpr uint32_t ldc_alignment = 8;
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
using unaligned_gemm_func = unaligned_gemm_test_func<
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
    Test::lda_alignment,
    Test::ldb_alignment,
    Test::ldc_alignment,
    Test::global_kslicing,
    Test::local_kslicing,
    Test::engine,
    Test::arch_tag>;
