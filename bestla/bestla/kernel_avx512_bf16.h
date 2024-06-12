//  Copyright (c) 2023 Intel Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
#pragma once
#include "kernel_avx512f.h"

namespace bestla {
namespace kernel {
namespace avx512f {
namespace avx512_bf16 {
#if CompileBF16()
#if defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx512bf16", "avx512vl", "avx512bw")
#endif
static inline __m512 zmm_cvt_bf16_fp32(__m256i vbf16) {
  auto vf32 = _mm512_cvtpbh_ps(vbf16);
  return vf32;
}

static inline __m256i zmm_cvt_fp32_bf16(__m512 vfp32) { return _mm512_cvtneps_pbh(vfp32); }

static inline __m512 load_bf16_fp32(const utils::bf16* srcptr) {
  auto tmp = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(srcptr));
  auto vf32 = zmm_cvt_bf16_fp32(tmp);
  return vf32;
}

static inline BTLA_CODE bf16_cvt_fp32_2D_write_back(const utils::bf16* src_ptr, float* dst_ptr, int row, int col,
                                                    int src_step, int dst_step, bool zeropadding) {
  const int npadding = (dst_step - col) * sizeof(float);
  constexpr int simd_proc_elt = 16;
  auto col_body = col / simd_proc_elt * simd_proc_elt;
  auto col_tail = col % simd_proc_elt;
  const auto tail_mask = _cvtu32_mask16((1U << col_tail) - 1);
  for (int i = 0; i < row; i++) {
    auto src = const_cast<utils::bf16*>(src_ptr + i * src_step);
    auto dst = dst_ptr + i * dst_step;
    int j = 0;
    for (; j < col_body; j += simd_proc_elt) _mm512_storeu_ps(dst + j, load_bf16_fp32(src + j));
    if (col_tail > 0) {
      __m256i tmp = _mm256_setzero_si256();
      tmp = _mm256_mask_loadu_epi16(tmp, tail_mask, src + j);
      _mm512_mask_storeu_ps(dst + j, tail_mask, zmm_cvt_bf16_fp32(tmp));
    }
    if (zeropadding && npadding) std::memset(dst + col, 0, npadding);
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE fp32_cvt_bf16_2D_write_back(const void* raw_srcptr, void* raw_dstptr, int row, int col,
                                                    int srcstride, int dststride, bool zeropadding) {
  auto srcptr = reinterpret_cast<const char*>(raw_srcptr);
  auto dstptr = reinterpret_cast<char*>(raw_dstptr);
  constexpr int simd_proc_elt = 32;
  auto col_body_loop = col / simd_proc_elt;
  auto col_tail = col % simd_proc_elt;
  const uint32_t tail_mask = (1U << col_tail) - 1;
  int npadding = dststride - col * sizeof(utils::bf16);
  for (int i = 0; i < row; i++) {
    auto src = srcptr + i * srcstride;
    auto dst = dstptr + i * dststride;
    int j = 0;
    for (; j < col_body_loop; j++) {
      _mm512_storeu_epi16(
          (dst + (j * simd_proc_elt) * sizeof(utils::bf16)),
          (__m512i)_mm512_cvtne2ps_pbh(_mm512_loadu_ps(src + sizeof(float) * simd_proc_elt * j + sizeof(float) * 16),
                                       _mm512_loadu_ps(src + sizeof(float) * simd_proc_elt * j + sizeof(float) * 0)));
    }
    if (col_tail > 0) {
      _mm512_mask_storeu_epi16(
          (dst + (j * simd_proc_elt) * sizeof(utils::bf16)), tail_mask,  //
          (__m512i)_mm512_cvtne2ps_pbh(
              _mm512_maskz_loadu_ps(tail_mask >> 16, src + sizeof(float) * simd_proc_elt * j + sizeof(float) * 16),
              _mm512_maskz_loadu_ps(tail_mask >> 0, src + sizeof(float) * simd_proc_elt * j + sizeof(float) * 0)));
    }
    if (zeropadding && npadding) {
      std::memset(dst + col * sizeof(utils::bf16), 0, npadding);
    }
  }
  return BTLA_CODE::Success;
}
#if defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif
}  // namespace avx512_bf16
}  // namespace avx512f
}  // namespace kernel
}  // namespace bestla
