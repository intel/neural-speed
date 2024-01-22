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

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <malloc.h>  // using malloc.h with MSC/MINGW
#elif !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#include <alloca.h>
#endif

#include <assert.h>
#include <errno.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <float.h>
#include <limits.h>

#ifndef __STDC_VERSION__
#define restrict
#elif __STDC_VERSION__ < 199901L
#define restrict
#endif

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <intrin.h>
#else
#include <immintrin.h>
#endif

#include "vectors/cpu/simd.h"
#include "core/data_types.h"
#include "vectors/cpu/quantize.h"

#define NE_VEC_DOT_UNROLL 2

#ifdef __cplusplus
extern "C" {
#endif

static void ne_vec_dot_f32(const int n, float* restrict s, const float* restrict x, const float* restrict y) {
#ifdef NE_SIMD
  float sumf = 0.0f;
  const int np = (n & ~(NE_F32_STEP - 1));

  NE_F32_VEC sum[NE_F32_ARR] = {NE_F32_VEC_ZERO};

  NE_F32_VEC ax[NE_F32_ARR];
  NE_F32_VEC ay[NE_F32_ARR];

  for (int i = 0; i < np; i += NE_F32_STEP) {
    for (int j = 0; j < NE_F32_ARR; j++) {
      ax[j] = NE_F32_VEC_LOAD(x + i + j * NE_F32_EPR);
      ay[j] = NE_F32_VEC_LOAD(y + i + j * NE_F32_EPR);

      sum[j] = NE_F32_VEC_FMA(sum[j], ax[j], ay[j]);
    }
  }

  // reduce sum0..sum3 to sum0
  NE_F32_VEC_REDUCE(sumf, sum);

  // leftovers
  for (int i = np; i < n; ++i) {
    sumf += x[i] * y[i];
  }
#else
  // scalar
  ne_float sumf = 0.0;
  for (int i = 0; i < n; ++i) {
    sumf += (ne_float)(x[i] * y[i]);
  }
#endif

  *s = sumf;
}

static void ne_vec_dot_f16(const int n, float* restrict s, ne_fp16_t* restrict x, ne_fp16_t* restrict y) {
  ne_float sumf = 0.0;

// NS_SIMD_VEC_DOT_F16 (sum order may affect logits, like padding and no padding)
#if defined(NE_SIMD) && defined(NS_SIMD_VEC_DOT_F16)
  const int np = (n & ~(NE_F16_STEP - 1));

  NE_F16_VEC sum[NE_F16_ARR] = {NE_F16_VEC_ZERO};

  NE_F16_VEC ax[NE_F16_ARR];
  NE_F16_VEC ay[NE_F16_ARR];

  for (int i = 0; i < np; i += NE_F16_STEP) {
    for (int j = 0; j < NE_F16_ARR; j++) {
      ax[j] = NE_F16_VEC_LOAD(x + i + j * NE_F16_EPR, j);
      ay[j] = NE_F16_VEC_LOAD(y + i + j * NE_F16_EPR, j);

      sum[j] = NE_F16_VEC_FMA(sum[j], ax[j], ay[j]);
    }
  }

  // reduce sum0..sum3 to sum0
  NE_F16_VEC_REDUCE(sumf, sum);

  // leftovers
  for (int i = np; i < n; ++i) {
    sumf += (ne_float)(NE_FP16_TO_FP32(x[i]) * NE_FP16_TO_FP32(y[i]));
  }
#else
  for (int i = 0; i < n; ++i) {
    sumf += (ne_float)(NE_FP16_TO_FP32(x[i]) * NE_FP16_TO_FP32(y[i]));
  }
#endif

  *s = sumf;
}

static void ne_vec_dot_q4_0_q8_0(const int n, float* restrict s, const void* restrict vx, const void* restrict vy) {
  const int qk = QK8_0;
  const int nb = n / qk;

  assert(n % qk == 0);
  assert(nb % 2 == 0);

  const block_q4_0* restrict x = (const block_q4_0*)vx;
  const block_q8_0* restrict y = (const block_q8_0*)vy;

#if defined(__AVX2__)
  // Initialize accumulator with zeros
  __m256 acc = _mm256_setzero_ps();

  // Main loop
  for (int i = 0; i < nb; ++i) {
    /* Compute combined scale for the block */
    const __m256 d = _mm256_set1_ps(NE_FP16_TO_FP32(x[i].d) * NE_FP16_TO_FP32(y[i].d));

    __m256i bx = bytes_from_nibbles_32(x[i].qs);

    // Now we have a vector with bytes in [ 0 .. 15 ] interval. Offset them into [ -8 .. +7 ] interval.
    const __m256i off = _mm256_set1_epi8(8);
    bx = _mm256_sub_epi8(bx, off);

    __m256i by = _mm256_loadu_si256((const __m256i*)y[i].qs);

    const __m256 q = mul_sum_i8_pairs_float(bx, by);

    /* Multiply q with scale and accumulate */
    acc = _mm256_fmadd_ps(d, q, acc);
  }

  *s = hsum_float_8(acc);
#elif defined(__AVX__)
  // Initialize accumulator with zeros
  __m256 acc = _mm256_setzero_ps();

  // Main loop
  for (int i = 0; i < nb; ++i) {
    // Compute combined scale for the block
    const __m256 d = _mm256_set1_ps(NE_FP16_TO_FP32(x[i].d) * NE_FP16_TO_FP32(y[i].d));

    const __m128i lowMask = _mm_set1_epi8(0xF);
    const __m128i off = _mm_set1_epi8(8);

    const __m128i tmp = _mm_loadu_si128((const __m128i*)x[i].qs);

    __m128i bx = _mm_and_si128(lowMask, tmp);
    __m128i by = _mm_loadu_si128((const __m128i*)y[i].qs);
    bx = _mm_sub_epi8(bx, off);
    const __m128i i32_0 = mul_sum_i8_pairs(bx, by);

    bx = _mm_and_si128(lowMask, _mm_srli_epi64(tmp, 4));
    by = _mm_loadu_si128((const __m128i*)(y[i].qs + 16));
    bx = _mm_sub_epi8(bx, off);
    const __m128i i32_1 = mul_sum_i8_pairs(bx, by);

    // Convert int32_t to float
    __m256 p = _mm256_cvtepi32_ps(_mm256_set_m128i(i32_0, i32_1));

    // Apply the scale, and accumulate
    acc = _mm256_add_ps(_mm256_mul_ps(d, p), acc);
  }

  *s = hsum_float_8(acc);
#elif defined(__SSSE3__)
  // set constants
  const __m128i lowMask = _mm_set1_epi8(0xF);
  const __m128i off = _mm_set1_epi8(8);

  // Initialize accumulator with zeros
  __m128 acc_0 = _mm_setzero_ps();
  __m128 acc_1 = _mm_setzero_ps();
  __m128 acc_2 = _mm_setzero_ps();
  __m128 acc_3 = _mm_setzero_ps();

  // First round without accumulation
  {
    _mm_prefetch(&x[0] + sizeof(block_q4_0), _MM_HINT_T0);
    _mm_prefetch(&y[0] + sizeof(block_q8_0), _MM_HINT_T0);

    // Compute combined scale for the block 0 and 1
    const __m128 d_0_1 = _mm_set1_ps(NE_FP16_TO_FP32(x[0].d) * NE_FP16_TO_FP32(y[0].d));

    const __m128i tmp_0_1 = _mm_loadu_si128((const __m128i*)x[0].qs);

    __m128i bx_0 = _mm_and_si128(lowMask, tmp_0_1);
    __m128i by_0 = _mm_loadu_si128((const __m128i*)y[0].qs);
    bx_0 = _mm_sub_epi8(bx_0, off);
    const __m128i i32_0 = mul_sum_i8_pairs(bx_0, by_0);

    __m128i bx_1 = _mm_and_si128(lowMask, _mm_srli_epi64(tmp_0_1, 4));
    __m128i by_1 = _mm_loadu_si128((const __m128i*)(y[0].qs + 16));
    bx_1 = _mm_sub_epi8(bx_1, off);
    const __m128i i32_1 = mul_sum_i8_pairs(bx_1, by_1);

    _mm_prefetch(&x[1] + sizeof(block_q4_0), _MM_HINT_T0);
    _mm_prefetch(&y[1] + sizeof(block_q8_0), _MM_HINT_T0);

    // Compute combined scale for the block 2 and 3
    const __m128 d_2_3 = _mm_set1_ps(NE_FP16_TO_FP32(x[1].d) * NE_FP16_TO_FP32(y[1].d));

    const __m128i tmp_2_3 = _mm_loadu_si128((const __m128i*)x[1].qs);

    __m128i bx_2 = _mm_and_si128(lowMask, tmp_2_3);
    __m128i by_2 = _mm_loadu_si128((const __m128i*)y[1].qs);
    bx_2 = _mm_sub_epi8(bx_2, off);
    const __m128i i32_2 = mul_sum_i8_pairs(bx_2, by_2);

    __m128i bx_3 = _mm_and_si128(lowMask, _mm_srli_epi64(tmp_2_3, 4));
    __m128i by_3 = _mm_loadu_si128((const __m128i*)(y[1].qs + 16));
    bx_3 = _mm_sub_epi8(bx_3, off);
    const __m128i i32_3 = mul_sum_i8_pairs(bx_3, by_3);

    // Convert int32_t to float
    __m128 p0 = _mm_cvtepi32_ps(i32_0);
    __m128 p1 = _mm_cvtepi32_ps(i32_1);
    __m128 p2 = _mm_cvtepi32_ps(i32_2);
    __m128 p3 = _mm_cvtepi32_ps(i32_3);

    // Apply the scale
    acc_0 = _mm_mul_ps(d_0_1, p0);
    acc_1 = _mm_mul_ps(d_0_1, p1);
    acc_2 = _mm_mul_ps(d_2_3, p2);
    acc_3 = _mm_mul_ps(d_2_3, p3);
  }

  // Main loop
  for (int i = 2; i < nb; i += 2) {
    _mm_prefetch(&x[i] + sizeof(block_q4_0), _MM_HINT_T0);
    _mm_prefetch(&y[i] + sizeof(block_q8_0), _MM_HINT_T0);

    // Compute combined scale for the block 0 and 1
    const __m128 d_0_1 = _mm_set1_ps(NE_FP16_TO_FP32(x[i].d) * NE_FP16_TO_FP32(y[i].d));

    const __m128i tmp_0_1 = _mm_loadu_si128((const __m128i*)x[i].qs);

    __m128i bx_0 = _mm_and_si128(lowMask, tmp_0_1);
    __m128i by_0 = _mm_loadu_si128((const __m128i*)y[i].qs);
    bx_0 = _mm_sub_epi8(bx_0, off);
    const __m128i i32_0 = mul_sum_i8_pairs(bx_0, by_0);

    __m128i bx_1 = _mm_and_si128(lowMask, _mm_srli_epi64(tmp_0_1, 4));
    __m128i by_1 = _mm_loadu_si128((const __m128i*)(y[i].qs + 16));
    bx_1 = _mm_sub_epi8(bx_1, off);
    const __m128i i32_1 = mul_sum_i8_pairs(bx_1, by_1);

    _mm_prefetch(&x[i] + 2 * sizeof(block_q4_0), _MM_HINT_T0);
    _mm_prefetch(&y[i] + 2 * sizeof(block_q8_0), _MM_HINT_T0);

    // Compute combined scale for the block 2 and 3
    const __m128 d_2_3 = _mm_set1_ps(NE_FP16_TO_FP32(x[i + 1].d) * NE_FP16_TO_FP32(y[i + 1].d));

    const __m128i tmp_2_3 = _mm_loadu_si128((const __m128i*)x[i + 1].qs);

    __m128i bx_2 = _mm_and_si128(lowMask, tmp_2_3);
    __m128i by_2 = _mm_loadu_si128((const __m128i*)y[i + 1].qs);
    bx_2 = _mm_sub_epi8(bx_2, off);
    const __m128i i32_2 = mul_sum_i8_pairs(bx_2, by_2);

    __m128i bx_3 = _mm_and_si128(lowMask, _mm_srli_epi64(tmp_2_3, 4));
    __m128i by_3 = _mm_loadu_si128((const __m128i*)(y[i + 1].qs + 16));
    bx_3 = _mm_sub_epi8(bx_3, off);
    const __m128i i32_3 = mul_sum_i8_pairs(bx_3, by_3);

    // Convert int32_t to float
    __m128 p0 = _mm_cvtepi32_ps(i32_0);
    __m128 p1 = _mm_cvtepi32_ps(i32_1);
    __m128 p2 = _mm_cvtepi32_ps(i32_2);
    __m128 p3 = _mm_cvtepi32_ps(i32_3);

    // Apply the scale
    __m128 p0_d = _mm_mul_ps(d_0_1, p0);
    __m128 p1_d = _mm_mul_ps(d_0_1, p1);
    __m128 p2_d = _mm_mul_ps(d_2_3, p2);
    __m128 p3_d = _mm_mul_ps(d_2_3, p3);

    // Acummulate
    acc_0 = _mm_add_ps(p0_d, acc_0);
    acc_1 = _mm_add_ps(p1_d, acc_1);
    acc_2 = _mm_add_ps(p2_d, acc_2);
    acc_3 = _mm_add_ps(p3_d, acc_3);
  }

  *s = hsum_float_4x4(acc_0, acc_1, acc_2, acc_3);
#else
  // scalar
  float sumf = 0.0;

  for (int i = 0; i < nb; i++) {
    int sumi = 0;

    for (int j = 0; j < qk / 2; ++j) {
      const int v0 = (x[i].qs[j] & 0x0F) - 8;
      const int v1 = (x[i].qs[j] >> 4) - 8;

      sumi += (v0 * y[i].qs[j]) + (v1 * y[i].qs[j + qk / 2]);
    }

    sumf += sumi * NE_FP16_TO_FP32(x[i].d) * NE_FP16_TO_FP32(y[i].d);
  }

  *s = sumf;
#endif
}

static void ne_vec_dot_q4_1_q8_1(const int n, float* restrict s, const void* restrict vx, const void* restrict vy) {
  const int qk = QK8_1;
  const int nb = n / qk;

  assert(n % qk == 0);
  assert(nb % 2 == 0);

  const block_q4_1* restrict x = (const block_q4_1*)vx;
  const block_q8_1* restrict y = (const block_q8_1*)vy;

  // TODO: add WASM SIMD
#if defined(__AVX2__) || defined(__AVX__)
  // Initialize accumulator with zeros
  __m256 acc = _mm256_setzero_ps();

  float summs = 0;

  // Main loop
  for (int i = 0; i < nb; ++i) {
    const float d0 = NE_FP16_TO_FP32(x[i].d);
    const float d1 = y[i].d;

    summs += NE_FP16_TO_FP32(x[i].m) * y[i].s;

    const __m256 d0v = _mm256_set1_ps(d0);
    const __m256 d1v = _mm256_set1_ps(d1);

    // Compute combined scales
    const __m256 d0d1 = _mm256_mul_ps(d0v, d1v);

    // Load 16 bytes, and unpack 4 bit fields into bytes, making 32 bytes
    const __m256i bx = bytes_from_nibbles_32(x[i].qs);
    const __m256i by = _mm256_loadu_si256((const __m256i*)y[i].qs);

    const __m256 xy = mul_sum_us8_pairs_float(bx, by);

    // Accumulate d0*d1*x*y
#if defined(__AVX2__)
    acc = _mm256_fmadd_ps(d0d1, xy, acc);
#else
    acc = _mm256_add_ps(_mm256_mul_ps(d0d1, xy), acc);
#endif
  }

  *s = hsum_float_8(acc) + summs;
#else
  // scalar
  float sumf = 0.0;

  for (int i = 0; i < nb; i++) {
    int sumi = 0;

    for (int j = 0; j < qk / 2; ++j) {
      const int v0 = (x[i].qs[j] & 0x0F);
      const int v1 = (x[i].qs[j] >> 4);

      sumi += (v0 * y[i].qs[j]) + (v1 * y[i].qs[j + qk / 2]);
    }

    sumf += (NE_FP16_TO_FP32(x[i].d) * y[i].d) * sumi + NE_FP16_TO_FP32(x[i].m) * y[i].s;
  }

  *s = sumf;
#endif
}

static void ne_vec_dot_q5_0_q8_0(const int n, float* restrict s, const void* restrict vx, const void* restrict vy) {
  const int qk = QK8_0;
  const int nb = n / qk;

  assert(n % qk == 0);
  assert(nb % 2 == 0);
  assert(qk == QK5_0);

  const block_q5_0* restrict x = (const block_q5_0*)vx;
  const block_q8_0* restrict y = (const block_q8_0*)vy;

#if defined(__AVX2__)
  // Initialize accumulator with zeros
  __m256 acc = _mm256_setzero_ps();

  // Main loop
  for (int i = 0; i < nb; i++) {
    /* Compute combined scale for the block */
    const __m256 d = _mm256_set1_ps(NE_FP16_TO_FP32(x[i].d) * NE_FP16_TO_FP32(y[i].d));

    __m256i bx = bytes_from_nibbles_32(x[i].qs);
    __m256i bxhi = bytes_from_bits_32(x[i].qh);
    bxhi = _mm256_andnot_si256(bxhi, _mm256_set1_epi8((char)0xF0));
    bx = _mm256_or_si256(bx, bxhi);

    __m256i by = _mm256_loadu_si256((const __m256i*)y[i].qs);

    const __m256 q = mul_sum_i8_pairs_float(bx, by);

    /* Multiply q with scale and accumulate */
    acc = _mm256_fmadd_ps(d, q, acc);
  }

  *s = hsum_float_8(acc);
#elif defined(__AVX__)
  // Initialize accumulator with zeros
  __m256 acc = _mm256_setzero_ps();
  __m128i mask = _mm_set1_epi8((char)0xF0);

  // Main loop
  for (int i = 0; i < nb; i++) {
    /* Compute combined scale for the block */
    const __m256 d = _mm256_set1_ps(NE_FP16_TO_FP32(x[i].d) * NE_FP16_TO_FP32(y[i].d));

    __m256i bx = bytes_from_nibbles_32(x[i].qs);
    const __m256i bxhi = bytes_from_bits_32(x[i].qh);
    __m128i bxhil = _mm256_castsi256_si128(bxhi);
    __m128i bxhih = _mm256_extractf128_si256(bxhi, 1);
    bxhil = _mm_andnot_si128(bxhil, mask);
    bxhih = _mm_andnot_si128(bxhih, mask);
    __m128i bxl = _mm256_castsi256_si128(bx);
    __m128i bxh = _mm256_extractf128_si256(bx, 1);
    bxl = _mm_or_si128(bxl, bxhil);
    bxh = _mm_or_si128(bxh, bxhih);
    bx = _mm256_set_m128i(bxh, bxl);

    const __m256i by = _mm256_loadu_si256((const __m256i*)y[i].qs);

    const __m256 q = mul_sum_i8_pairs_float(bx, by);

    /* Multiply q with scale and accumulate */
    acc = _mm256_add_ps(_mm256_mul_ps(d, q), acc);
  }

  *s = hsum_float_8(acc);
#else
  // scalar
  float sumf = 0.0;

  for (int i = 0; i < nb; i++) {
    uint32_t qh;
    memcpy(&qh, x[i].qh, sizeof(qh));

    int sumi = 0;

    for (int j = 0; j < qk / 2; ++j) {
      const uint8_t xh_0 = ((qh & (1u << (j + 0))) >> (j + 0)) << 4;
      const uint8_t xh_1 = ((qh & (1u << (j + 16))) >> (j + 12));

      const int32_t x0 = ((x[i].qs[j] & 0x0F) | xh_0) - 16;
      const int32_t x1 = ((x[i].qs[j] >> 4) | xh_1) - 16;

      sumi += (x0 * y[i].qs[j]) + (x1 * y[i].qs[j + qk / 2]);
    }

    sumf += (NE_FP16_TO_FP32(x[i].d) * NE_FP16_TO_FP32(y[i].d)) * sumi;
  }

  *s = sumf;
#endif
}

static void ne_vec_dot_q5_1_q8_1(const int n, float* restrict s, const void* restrict vx, const void* restrict vy) {
  const int qk = QK8_1;
  const int nb = n / qk;

  assert(n % qk == 0);
  assert(nb % 2 == 0);
  assert(qk == QK5_1);

  const block_q5_1* restrict x = (const block_q5_1*)vx;
  const block_q8_1* restrict y = (const block_q8_1*)vy;

#if defined(__AVX2__)
  // Initialize accumulator with zeros
  __m256 acc = _mm256_setzero_ps();

  float summs = 0.0f;

  // Main loop
  for (int i = 0; i < nb; i++) {
    const __m256 dx = _mm256_set1_ps(NE_FP16_TO_FP32(x[i].d));

    summs += NE_FP16_TO_FP32(x[i].m) * y[i].s;

    __m256i bx = bytes_from_nibbles_32(x[i].qs);
    __m256i bxhi = bytes_from_bits_32(x[i].qh);
    bxhi = _mm256_and_si256(bxhi, _mm256_set1_epi8(0x10));
    bx = _mm256_or_si256(bx, bxhi);

    const __m256 dy = _mm256_set1_ps(y[i].d);
    const __m256i by = _mm256_loadu_si256((const __m256i*)y[i].qs);

    const __m256 q = mul_sum_us8_pairs_float(bx, by);

    acc = _mm256_fmadd_ps(q, _mm256_mul_ps(dx, dy), acc);
  }

  *s = hsum_float_8(acc) + summs;
#elif defined(__AVX__)
  // Initialize accumulator with zeros
  __m256 acc = _mm256_setzero_ps();
  __m128i mask = _mm_set1_epi8(0x10);

  float summs = 0.0f;

  // Main loop
  for (int i = 0; i < nb; i++) {
    const __m256 dx = _mm256_set1_ps(NE_FP16_TO_FP32(x[i].d));

    summs += NE_FP16_TO_FP32(x[i].m) * y[i].s;

    __m256i bx = bytes_from_nibbles_32(x[i].qs);
    const __m256i bxhi = bytes_from_bits_32(x[i].qh);
    __m128i bxhil = _mm256_castsi256_si128(bxhi);
    __m128i bxhih = _mm256_extractf128_si256(bxhi, 1);
    bxhil = _mm_and_si128(bxhil, mask);
    bxhih = _mm_and_si128(bxhih, mask);
    __m128i bxl = _mm256_castsi256_si128(bx);
    __m128i bxh = _mm256_extractf128_si256(bx, 1);
    bxl = _mm_or_si128(bxl, bxhil);
    bxh = _mm_or_si128(bxh, bxhih);
    bx = _mm256_set_m128i(bxh, bxl);

    const __m256 dy = _mm256_set1_ps(y[i].d);
    const __m256i by = _mm256_loadu_si256((const __m256i*)y[i].qs);

    const __m256 q = mul_sum_us8_pairs_float(bx, by);

    acc = _mm256_add_ps(_mm256_mul_ps(q, _mm256_mul_ps(dx, dy)), acc);
  }

  *s = hsum_float_8(acc) + summs;
#else
  // scalar
  float sumf = 0.0;

  for (int i = 0; i < nb; i++) {
    uint32_t qh;
    memcpy(&qh, x[i].qh, sizeof(qh));

    int sumi = 0;

    for (int j = 0; j < qk / 2; ++j) {
      const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
      const uint8_t xh_1 = ((qh >> (j + 12))) & 0x10;

      const int32_t x0 = (x[i].qs[j] & 0xF) | xh_0;
      const int32_t x1 = (x[i].qs[j] >> 4) | xh_1;

      sumi += (x0 * y[i].qs[j]) + (x1 * y[i].qs[j + qk / 2]);
    }

    sumf += (NE_FP16_TO_FP32(x[i].d) * y[i].d) * sumi + NE_FP16_TO_FP32(x[i].m) * y[i].s;
  }

  *s = sumf;
#endif
}

static void ne_vec_dot_q8_0_q8_0(const int n, float* restrict s, const void* restrict vx, const void* restrict vy) {
  const int qk = QK8_0;
  const int nb = n / qk;

  assert(n % qk == 0);
  assert(nb % 2 == 0);

  const block_q8_0* restrict x = (const block_q8_0*)vx;
  const block_q8_0* restrict y = (const block_q8_0*)vy;

#if defined(__AVX2__) || defined(__AVX__)
  // Initialize accumulator with zeros
  __m256 acc = _mm256_setzero_ps();

  // Main loop
  for (int i = 0; i < nb; ++i) {
    // Compute combined scale for the block
    const __m256 d = _mm256_set1_ps(NE_FP16_TO_FP32(x[i].d) * NE_FP16_TO_FP32(y[i].d));
    __m256i bx = _mm256_loadu_si256((const __m256i*)x[i].qs);
    __m256i by = _mm256_loadu_si256((const __m256i*)y[i].qs);

    const __m256 q = mul_sum_i8_pairs_float(bx, by);

    // Multiply q with scale and accumulate
#if defined(__AVX2__)
    acc = _mm256_fmadd_ps(d, q, acc);
#else
    acc = _mm256_add_ps(_mm256_mul_ps(d, q), acc);
#endif
  }

  *s = hsum_float_8(acc);
#else
  // scalar
  float sumf = 0.0;

  for (int i = 0; i < nb; i++) {
    int sumi = 0;

    for (int j = 0; j < qk; j++) {
      sumi += x[i].qs[j] * y[i].qs[j];
    }

    sumf += sumi * (NE_FP16_TO_FP32(x[i].d) * NE_FP16_TO_FP32(y[i].d));
  }

  *s = sumf;
#endif
}

// compute NE_VEC_DOT_UNROLL dot products at once
// xs - x row stride in bytes
static void ne_vec_dot_f16_unroll(const int n, const int xs, float* restrict s, void* restrict xv,
                                  ne_fp16_t* restrict y) {
  ne_float sumf[NE_VEC_DOT_UNROLL] = {0.0};

  ne_fp16_t* restrict x[NE_VEC_DOT_UNROLL];

  for (int i = 0; i < NE_VEC_DOT_UNROLL; ++i) {
    x[i] = (ne_fp16_t*)((char*)xv + i * xs);
  }

#if defined(NE_SIMD)
  const int np = (n & ~(NE_F16_STEP - 1));

  NE_F16_VEC sum[NE_VEC_DOT_UNROLL][NE_F16_ARR] = {{NE_F16_VEC_ZERO}};

  NE_F16_VEC ax[NE_F16_ARR];
  NE_F16_VEC ay[NE_F16_ARR];

  for (int i = 0; i < np; i += NE_F16_STEP) {
    for (int j = 0; j < NE_F16_ARR; j++) {
      ay[j] = NE_F16_VEC_LOAD(y + i + j * NE_F16_EPR, j);

      for (int k = 0; k < NE_VEC_DOT_UNROLL; ++k) {
        ax[j] = NE_F16_VEC_LOAD(x[k] + i + j * NE_F16_EPR, j);

        sum[k][j] = NE_F16_VEC_FMA(sum[k][j], ax[j], ay[j]);
      }
    }
  }

  // reduce sum0..sum3 to sum0
  for (int k = 0; k < NE_VEC_DOT_UNROLL; ++k) {
    NE_F16_VEC_REDUCE(sumf[k], sum[k]);
  }

  // leftovers
  for (int i = np; i < n; ++i) {
    for (int j = 0; j < NE_VEC_DOT_UNROLL; ++j) {
      sumf[j] += (ne_float)(NE_FP16_TO_FP32(x[j][i]) * NE_FP16_TO_FP32(y[i]));
    }
  }
#else
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < NE_VEC_DOT_UNROLL; ++j) {
      sumf[j] += (ne_float)(NE_FP16_TO_FP32(x[j][i]) * NE_FP16_TO_FP32(y[i]));
    }
  }
#endif

  for (int i = 0; i < NE_VEC_DOT_UNROLL; ++i) {
    s[i] = sumf[i];
  }
}

//===================================== Dot ptoducts =================================

//
// Helper functions
//
#if __AVX__ || __AVX2__ || __AVX512F__

// shuffles to pick the required scales in dot products
static inline __m256i get_scale_shuffle_q3k(int i) {
  static const uint8_t k_shuffle[128] = {
      0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,
      2,  3,  2,  3,  2,  3,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  6,  7,  6,  7,
      6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,
      8,  9,  10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 12, 13, 12, 13, 12, 13, 12, 13,
      12, 13, 12, 13, 12, 13, 12, 13, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
  };
  return _mm256_loadu_si256((const __m256i*)k_shuffle + i);
}
static inline __m256i get_scale_shuffle_k4(int i) {
  static const uint8_t k_shuffle[256] = {
      0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,
      0,  1,  0,  1,  0,  1,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,
      2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,
      4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  6,  7,  6,  7,  6,  7,  6,  7,
      6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  8,  9,
      8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,
      8,  9,  8,  9,  10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11,
      10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13,
      12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
      14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15};
  return _mm256_loadu_si256((const __m256i*)k_shuffle + i);
}
static inline __m128i get_scale_shuffle(int i) {
  static const uint8_t k_shuffle[128] = {
      0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,
      3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,  4,  4,  5,  5,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,
      6,  6,  6,  6,  7,  7,  7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,  8,  8,  8,  9,  9,  9,  9,  9,  9,
      9,  9,  10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12,
      13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15};
  return _mm_loadu_si128((const __m128i*)k_shuffle + i);
}
#endif

#if QK_K == 256
static void ggml_vec_dot_q6_K_q8_K(const int n, float* restrict s, const void* restrict vx, const void* restrict vy) {
  assert(n % QK_K == 0);

  const block_q6_K* restrict x = (const block_q6_K*)vx;
  const block_q8_K* restrict y = (const block_q8_K*)vy;

  const int nb = n / QK_K;

#ifdef __ARM_NEON

  float sum = 0;

  const uint8x16_t m4b = vdupq_n_u8(0xF);
#if defined(__ARM_FEATURE_DOTPROD)
  const int32x4_t vzero = vdupq_n_s32(0);
#endif
  // const int8x16_t  m32s = vdupq_n_s8(32);

  const uint8x16_t mone = vdupq_n_u8(3);

  ggml_int8x16x4_t q6bytes;
  ggml_uint8x16x4_t q6h;

  for (int i = 0; i < nb; ++i) {
    const float d_all = NE_FP16_TO_FP32(x[i].d);

    const uint8_t* restrict q6 = x[i].ql;
    const uint8_t* restrict qh = x[i].qh;
    const int8_t* restrict q8 = y[i].qs;

    const int8_t* restrict scale = x[i].scales;

    const ggml_int16x8x2_t q8sums = ggml_vld1q_s16_x2(y[i].bsums);
    const int8x16_t scales = vld1q_s8(scale);
    const ggml_int16x8x2_t q6scales = {vmovl_s8(vget_low_s8(scales)), vmovl_s8(vget_high_s8(scales))};

    const int32x4_t prod =
        vaddq_s32(vaddq_s32(vmull_s16(vget_low_s16(q8sums.val[0]), vget_low_s16(q6scales.val[0])),
                            vmull_s16(vget_high_s16(q8sums.val[0]), vget_high_s16(q6scales.val[0]))),
                  vaddq_s32(vmull_s16(vget_low_s16(q8sums.val[1]), vget_low_s16(q6scales.val[1])),
                            vmull_s16(vget_high_s16(q8sums.val[1]), vget_high_s16(q6scales.val[1]))));
    int32_t isum_mins = vaddvq_s32(prod);

    int32_t isum = 0;

    for (int j = 0; j < QK_K / 128; ++j) {
      ggml_uint8x16x2_t qhbits = ggml_vld1q_u8_x2(qh);
      qh += 32;
      ggml_uint8x16x4_t q6bits = ggml_vld1q_u8_x4(q6);
      q6 += 64;
      ggml_int8x16x4_t q8bytes = ggml_vld1q_s8_x4(q8);
      q8 += 64;

      q6h.val[0] = vshlq_n_u8(vandq_u8(mone, qhbits.val[0]), 4);
      q6h.val[1] = vshlq_n_u8(vandq_u8(mone, qhbits.val[1]), 4);
      uint8x16_t shifted = vshrq_n_u8(qhbits.val[0], 2);
      q6h.val[2] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
      shifted = vshrq_n_u8(qhbits.val[1], 2);
      q6h.val[3] = vshlq_n_u8(vandq_u8(mone, shifted), 4);

      // q6bytes.val[0] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[0], m4b), q6h.val[0])), m32s);
      // q6bytes.val[1] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[1], m4b), q6h.val[1])), m32s);
      // q6bytes.val[2] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[2], m4b), q6h.val[2])), m32s);
      // q6bytes.val[3] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[3], m4b), q6h.val[3])), m32s);
      q6bytes.val[0] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[0], m4b), q6h.val[0]));
      q6bytes.val[1] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[1], m4b), q6h.val[1]));
      q6bytes.val[2] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[2], m4b), q6h.val[2]));
      q6bytes.val[3] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[3], m4b), q6h.val[3]));

#if defined(__ARM_FEATURE_DOTPROD)

      isum += vaddvq_s32(vdotq_s32(vzero, q6bytes.val[0], q8bytes.val[0])) * scale[0] +
              vaddvq_s32(vdotq_s32(vzero, q6bytes.val[1], q8bytes.val[1])) * scale[1] +
              vaddvq_s32(vdotq_s32(vzero, q6bytes.val[2], q8bytes.val[2])) * scale[2] +
              vaddvq_s32(vdotq_s32(vzero, q6bytes.val[3], q8bytes.val[3])) * scale[3];
      scale += 4;

#else

      int16x8_t p0 = vaddq_s16(vmull_s8(vget_low_s8(q6bytes.val[0]), vget_low_s8(q8bytes.val[0])),
                               vmull_s8(vget_high_s8(q6bytes.val[0]), vget_high_s8(q8bytes.val[0])));
      int16x8_t p1 = vaddq_s16(vmull_s8(vget_low_s8(q6bytes.val[1]), vget_low_s8(q8bytes.val[1])),
                               vmull_s8(vget_high_s8(q6bytes.val[1]), vget_high_s8(q8bytes.val[1])));
      isum += vaddvq_s16(p0) * scale[0] + vaddvq_s16(p1) * scale[1];
      scale += 2;

      int16x8_t p2 = vaddq_s16(vmull_s8(vget_low_s8(q6bytes.val[2]), vget_low_s8(q8bytes.val[2])),
                               vmull_s8(vget_high_s8(q6bytes.val[2]), vget_high_s8(q8bytes.val[2])));
      int16x8_t p3 = vaddq_s16(vmull_s8(vget_low_s8(q6bytes.val[3]), vget_low_s8(q8bytes.val[3])),
                               vmull_s8(vget_high_s8(q6bytes.val[3]), vget_high_s8(q8bytes.val[3])));
      isum += vaddvq_s16(p2) * scale[0] + vaddvq_s16(p3) * scale[1];
      scale += 2;
#endif

      q8bytes = ggml_vld1q_s8_x4(q8);
      q8 += 64;

      shifted = vshrq_n_u8(qhbits.val[0], 4);
      q6h.val[0] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
      shifted = vshrq_n_u8(qhbits.val[1], 4);
      q6h.val[1] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
      shifted = vshrq_n_u8(qhbits.val[0], 6);
      q6h.val[2] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
      shifted = vshrq_n_u8(qhbits.val[1], 6);
      q6h.val[3] = vshlq_n_u8(vandq_u8(mone, shifted), 4);

      // q6bytes.val[0] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[0], 4), q6h.val[0])), m32s);
      // q6bytes.val[1] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[1], 4), q6h.val[1])), m32s);
      // q6bytes.val[2] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[2], 4), q6h.val[2])), m32s);
      // q6bytes.val[3] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[3], 4), q6h.val[3])), m32s);
      q6bytes.val[0] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[0], 4), q6h.val[0]));
      q6bytes.val[1] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[1], 4), q6h.val[1]));
      q6bytes.val[2] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[2], 4), q6h.val[2]));
      q6bytes.val[3] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[3], 4), q6h.val[3]));

#if defined(__ARM_FEATURE_DOTPROD)

      isum += vaddvq_s32(vdotq_s32(vzero, q6bytes.val[0], q8bytes.val[0])) * scale[0] +
              vaddvq_s32(vdotq_s32(vzero, q6bytes.val[1], q8bytes.val[1])) * scale[1] +
              vaddvq_s32(vdotq_s32(vzero, q6bytes.val[2], q8bytes.val[2])) * scale[2] +
              vaddvq_s32(vdotq_s32(vzero, q6bytes.val[3], q8bytes.val[3])) * scale[3];
      scale += 4;

      // for (int l = 0; l < 4; ++l) {
      //     const int32x4_t p = vdotq_s32(vzero, q6bytes.val[l], q8bytes.val[l]);
      //     isum += vaddvq_s32(p) * *scale++;
      // }
#else
      p0 = vaddq_s16(vmull_s8(vget_low_s8(q6bytes.val[0]), vget_low_s8(q8bytes.val[0])),
                     vmull_s8(vget_high_s8(q6bytes.val[0]), vget_high_s8(q8bytes.val[0])));
      p1 = vaddq_s16(vmull_s8(vget_low_s8(q6bytes.val[1]), vget_low_s8(q8bytes.val[1])),
                     vmull_s8(vget_high_s8(q6bytes.val[1]), vget_high_s8(q8bytes.val[1])));
      isum += vaddvq_s16(p0) * scale[0] + vaddvq_s16(p1) * scale[1];
      scale += 2;

      p2 = vaddq_s16(vmull_s8(vget_low_s8(q6bytes.val[2]), vget_low_s8(q8bytes.val[2])),
                     vmull_s8(vget_high_s8(q6bytes.val[2]), vget_high_s8(q8bytes.val[2])));
      p3 = vaddq_s16(vmull_s8(vget_low_s8(q6bytes.val[3]), vget_low_s8(q8bytes.val[3])),
                     vmull_s8(vget_high_s8(q6bytes.val[3]), vget_high_s8(q8bytes.val[3])));
      isum += vaddvq_s16(p2) * scale[0] + vaddvq_s16(p3) * scale[1];
      scale += 2;
#endif
    }
    // sum += isum * d_all * y[i].d;
    sum += d_all * y[i].d * (isum - 32 * isum_mins);
  }
  *s = sum;

#elif defined __AVX2__

  const __m256i m4 = _mm256_set1_epi8(0xF);
  const __m256i m2 = _mm256_set1_epi8(3);
  const __m256i m32s = _mm256_set1_epi8(32);

  __m256 acc = _mm256_setzero_ps();

  for (int i = 0; i < nb; ++i) {
    const float d = y[i].d * NE_FP16_TO_FP32(x[i].d);

    const uint8_t* restrict q4 = x[i].ql;
    const uint8_t* restrict qh = x[i].qh;
    const int8_t* restrict q8 = y[i].qs;

    const __m128i scales = _mm_loadu_si128((const __m128i*)x[i].scales);

    __m256i sumi = _mm256_setzero_si256();

    int is = 0;

    for (int j = 0; j < QK_K / 128; ++j) {
      const __m128i scale_0 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 0));
      const __m128i scale_1 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 1));
      const __m128i scale_2 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 2));
      const __m128i scale_3 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 3));
      is += 4;

      const __m256i q4bits1 = _mm256_loadu_si256((const __m256i*)q4);
      q4 += 32;
      const __m256i q4bits2 = _mm256_loadu_si256((const __m256i*)q4);
      q4 += 32;
      const __m256i q4bitsH = _mm256_loadu_si256((const __m256i*)qh);
      qh += 32;

      const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m2), 4);
      const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 2), m2), 4);
      const __m256i q4h_2 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 4), m2), 4);
      const __m256i q4h_3 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 6), m2), 4);

      const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m4), q4h_0);
      const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m4), q4h_1);
      const __m256i q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m4), q4h_2);
      const __m256i q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m4), q4h_3);

      const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8);
      q8 += 32;
      const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8);
      q8 += 32;
      const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8);
      q8 += 32;
      const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8);
      q8 += 32;

      __m256i q8s_0 = _mm256_maddubs_epi16(m32s, q8_0);
      __m256i q8s_1 = _mm256_maddubs_epi16(m32s, q8_1);
      __m256i q8s_2 = _mm256_maddubs_epi16(m32s, q8_2);
      __m256i q8s_3 = _mm256_maddubs_epi16(m32s, q8_3);

      __m256i p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
      __m256i p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
      __m256i p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
      __m256i p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

      p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
      p16_1 = _mm256_sub_epi16(p16_1, q8s_1);
      p16_2 = _mm256_sub_epi16(p16_2, q8s_2);
      p16_3 = _mm256_sub_epi16(p16_3, q8s_3);

      p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
      p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
      p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
      p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);

      sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
      sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
    }

    acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
  }

  *s = hsum_float_8(acc);

#elif defined __AVX__

  const __m128i m4 = _mm_set1_epi8(0xF);
  const __m128i m3 = _mm_set1_epi8(3);
  const __m128i m32s = _mm_set1_epi8(32);
  const __m128i m2 = _mm_set1_epi8(2);

  __m256 acc = _mm256_setzero_ps();

  for (int i = 0; i < nb; ++i) {
    const float d = y[i].d * NE_FP16_TO_FP32(x[i].d);

    const uint8_t* restrict q4 = x[i].ql;
    const uint8_t* restrict qh = x[i].qh;
    const int8_t* restrict q8 = y[i].qs;

    const __m128i scales = _mm_loadu_si128((const __m128i*)x[i].scales);

    __m128i sumi_0 = _mm_setzero_si128();
    __m128i sumi_1 = _mm_setzero_si128();

    __m128i shuffle = _mm_set_epi64x(0x0101010101010101, 0x0000000000000000);
    for (int j = 0; j < QK_K / 128; ++j) {
      const __m128i q4bitsH_0 = _mm_loadu_si128((const __m128i*)qh);
      qh += 16;
      const __m128i q4bitsH_1 = _mm_loadu_si128((const __m128i*)qh);
      qh += 16;

      const __m128i q4h_0 = _mm_slli_epi16(_mm_and_si128(q4bitsH_0, m3), 4);
      const __m128i q4h_1 = _mm_slli_epi16(_mm_and_si128(q4bitsH_1, m3), 4);
      const __m128i q4h_2 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bitsH_0, 2), m3), 4);
      const __m128i q4h_3 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bitsH_1, 2), m3), 4);
      const __m128i q4h_4 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bitsH_0, 4), m3), 4);
      const __m128i q4h_5 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bitsH_1, 4), m3), 4);
      const __m128i q4h_6 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bitsH_0, 6), m3), 4);
      const __m128i q4h_7 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bitsH_1, 6), m3), 4);

      const __m128i q4bits1_0 = _mm_loadu_si128((const __m128i*)q4);
      q4 += 16;
      const __m128i q4bits1_1 = _mm_loadu_si128((const __m128i*)q4);
      q4 += 16;
      const __m128i q4bits2_0 = _mm_loadu_si128((const __m128i*)q4);
      q4 += 16;
      const __m128i q4bits2_1 = _mm_loadu_si128((const __m128i*)q4);
      q4 += 16;

      const __m128i q4_0 = _mm_or_si128(_mm_and_si128(q4bits1_0, m4), q4h_0);
      const __m128i q4_1 = _mm_or_si128(_mm_and_si128(q4bits1_1, m4), q4h_1);
      const __m128i q4_2 = _mm_or_si128(_mm_and_si128(q4bits2_0, m4), q4h_2);
      const __m128i q4_3 = _mm_or_si128(_mm_and_si128(q4bits2_1, m4), q4h_3);
      const __m128i q4_4 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits1_0, 4), m4), q4h_4);
      const __m128i q4_5 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits1_1, 4), m4), q4h_5);
      const __m128i q4_6 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits2_0, 4), m4), q4h_6);
      const __m128i q4_7 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits2_1, 4), m4), q4h_7);

      const __m128i q8_0 = _mm_loadu_si128((const __m128i*)q8);
      q8 += 16;
      const __m128i q8_1 = _mm_loadu_si128((const __m128i*)q8);
      q8 += 16;
      const __m128i q8_2 = _mm_loadu_si128((const __m128i*)q8);
      q8 += 16;
      const __m128i q8_3 = _mm_loadu_si128((const __m128i*)q8);
      q8 += 16;
      const __m128i q8_4 = _mm_loadu_si128((const __m128i*)q8);
      q8 += 16;
      const __m128i q8_5 = _mm_loadu_si128((const __m128i*)q8);
      q8 += 16;
      const __m128i q8_6 = _mm_loadu_si128((const __m128i*)q8);
      q8 += 16;
      const __m128i q8_7 = _mm_loadu_si128((const __m128i*)q8);
      q8 += 16;

      __m128i q8s_0 = _mm_maddubs_epi16(m32s, q8_0);
      __m128i q8s_1 = _mm_maddubs_epi16(m32s, q8_1);
      __m128i q8s_2 = _mm_maddubs_epi16(m32s, q8_2);
      __m128i q8s_3 = _mm_maddubs_epi16(m32s, q8_3);
      __m128i q8s_4 = _mm_maddubs_epi16(m32s, q8_4);
      __m128i q8s_5 = _mm_maddubs_epi16(m32s, q8_5);
      __m128i q8s_6 = _mm_maddubs_epi16(m32s, q8_6);
      __m128i q8s_7 = _mm_maddubs_epi16(m32s, q8_7);

      __m128i p16_0 = _mm_maddubs_epi16(q4_0, q8_0);
      __m128i p16_1 = _mm_maddubs_epi16(q4_1, q8_1);
      __m128i p16_2 = _mm_maddubs_epi16(q4_2, q8_2);
      __m128i p16_3 = _mm_maddubs_epi16(q4_3, q8_3);
      __m128i p16_4 = _mm_maddubs_epi16(q4_4, q8_4);
      __m128i p16_5 = _mm_maddubs_epi16(q4_5, q8_5);
      __m128i p16_6 = _mm_maddubs_epi16(q4_6, q8_6);
      __m128i p16_7 = _mm_maddubs_epi16(q4_7, q8_7);

      p16_0 = _mm_sub_epi16(p16_0, q8s_0);
      p16_1 = _mm_sub_epi16(p16_1, q8s_1);
      p16_2 = _mm_sub_epi16(p16_2, q8s_2);
      p16_3 = _mm_sub_epi16(p16_3, q8s_3);
      p16_4 = _mm_sub_epi16(p16_4, q8s_4);
      p16_5 = _mm_sub_epi16(p16_5, q8s_5);
      p16_6 = _mm_sub_epi16(p16_6, q8s_6);
      p16_7 = _mm_sub_epi16(p16_7, q8s_7);

      const __m128i scale_0 = _mm_shuffle_epi8(scales, shuffle);
      shuffle = _mm_add_epi8(shuffle, m2);
      const __m128i scale_1 = _mm_shuffle_epi8(scales, shuffle);
      shuffle = _mm_add_epi8(shuffle, m2);
      const __m128i scale_2 = _mm_shuffle_epi8(scales, shuffle);
      shuffle = _mm_add_epi8(shuffle, m2);
      const __m128i scale_3 = _mm_shuffle_epi8(scales, shuffle);
      shuffle = _mm_add_epi8(shuffle, m2);

      p16_0 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_0), p16_0);
      p16_1 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_unpackhi_epi64(scale_0, scale_0)), p16_1);
      p16_2 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_1), p16_2);
      p16_3 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_unpackhi_epi64(scale_1, scale_1)), p16_3);
      p16_4 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_2), p16_4);
      p16_5 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_unpackhi_epi64(scale_2, scale_2)), p16_5);
      p16_6 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_3), p16_6);
      p16_7 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_unpackhi_epi64(scale_3, scale_3)), p16_7);

      sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p16_0, p16_2));
      sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p16_1, p16_3));
      sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p16_4, p16_6));
      sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p16_5, p16_7));
    }

    __m256i sumi = MM256_SET_M128I(sumi_1, sumi_0);
    acc = _mm256_add_ps(_mm256_mul_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi)), acc);
  }

  *s = hsum_float_8(acc);

#elif defined __riscv_v_intrinsic

  float sumf = 0;
  for (int i = 0; i < nb; ++i) {
    const float d = NE_FP16_TO_FP32(x[i].d) * y[i].d;

    const uint8_t* restrict q6 = x[i].ql;
    const uint8_t* restrict qh = x[i].qh;
    const int8_t* restrict q8 = y[i].qs;

    const int8_t* restrict scale = x[i].scales;

    size_t vl;

    vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);

    int sum_t = 0;
    int is = 0;

    for (int j = 0; j < QK_K / 128; ++j) {
      vl = 32;

      // load qh
      vuint8m1_t qh_x = __riscv_vle8_v_u8m1(qh, vl);

      // load Q6
      vuint8m1_t q6_0 = __riscv_vle8_v_u8m1(q6, vl);
      vuint8m1_t q6_1 = __riscv_vle8_v_u8m1(q6 + 32, vl);

      vuint8m1_t q6a_0 = __riscv_vand_vx_u8m1(q6_0, 0x0F, vl);
      vuint8m1_t q6a_1 = __riscv_vand_vx_u8m1(q6_1, 0x0F, vl);
      vuint8m1_t q6s_0 = __riscv_vsrl_vx_u8m1(q6_0, 0x04, vl);
      vuint8m1_t q6s_1 = __riscv_vsrl_vx_u8m1(q6_1, 0x04, vl);

      vuint8m1_t qh_0 = __riscv_vand_vx_u8m1(qh_x, 0x03, vl);
      vuint8m1_t qh_1 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(qh_x, 0x2, vl), 0x03, vl);
      vuint8m1_t qh_2 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(qh_x, 0x4, vl), 0x03, vl);
      vuint8m1_t qh_3 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(qh_x, 0x6, vl), 0x03, vl);

      vuint8m1_t qhi_0 = __riscv_vor_vv_u8m1(q6a_0, __riscv_vsll_vx_u8m1(qh_0, 0x04, vl), vl);
      vuint8m1_t qhi_1 = __riscv_vor_vv_u8m1(q6a_1, __riscv_vsll_vx_u8m1(qh_1, 0x04, vl), vl);
      vuint8m1_t qhi_2 = __riscv_vor_vv_u8m1(q6s_0, __riscv_vsll_vx_u8m1(qh_2, 0x04, vl), vl);
      vuint8m1_t qhi_3 = __riscv_vor_vv_u8m1(q6s_1, __riscv_vsll_vx_u8m1(qh_3, 0x04, vl), vl);

      vint8m1_t a_0 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(qhi_0), 32, vl);
      vint8m1_t a_1 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(qhi_1), 32, vl);
      vint8m1_t a_2 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(qhi_2), 32, vl);
      vint8m1_t a_3 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(qhi_3), 32, vl);

      // load Q8 and take product
      vint16m2_t va_q_0 = __riscv_vwmul_vv_i16m2(a_0, __riscv_vle8_v_i8m1(q8, vl), vl);
      vint16m2_t va_q_1 = __riscv_vwmul_vv_i16m2(a_1, __riscv_vle8_v_i8m1(q8 + 32, vl), vl);
      vint16m2_t va_q_2 = __riscv_vwmul_vv_i16m2(a_2, __riscv_vle8_v_i8m1(q8 + 64, vl), vl);
      vint16m2_t va_q_3 = __riscv_vwmul_vv_i16m2(a_3, __riscv_vle8_v_i8m1(q8 + 96, vl), vl);

      vl = 16;

      vint32m2_t vaux_0 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_0, 0), scale[is + 0], vl);
      vint32m2_t vaux_1 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_0, 1), scale[is + 1], vl);
      vint32m2_t vaux_2 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_1, 0), scale[is + 2], vl);
      vint32m2_t vaux_3 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_1, 1), scale[is + 3], vl);
      vint32m2_t vaux_4 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_2, 0), scale[is + 4], vl);
      vint32m2_t vaux_5 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_2, 1), scale[is + 5], vl);
      vint32m2_t vaux_6 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_3, 0), scale[is + 6], vl);
      vint32m2_t vaux_7 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_3, 1), scale[is + 7], vl);

      vint32m1_t isum0 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(vaux_0, vaux_1, vl), vzero, vl);
      vint32m1_t isum1 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(vaux_2, vaux_3, vl), isum0, vl);
      vint32m1_t isum2 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(vaux_4, vaux_5, vl), isum1, vl);
      vint32m1_t isum3 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(vaux_6, vaux_7, vl), isum2, vl);

      sum_t += __riscv_vmv_x_s_i32m1_i32(isum3);

      q6 += 64;
      qh += 32;
      q8 += 128;
      is = 8;
    }

    sumf += d * sum_t;
  }

  *s = sumf;

#else

  int8_t aux8[QK_K];
  int16_t aux16[8];
  float sums[8];
  int32_t aux32[8];
  memset(sums, 0, 8 * sizeof(float));

  float sumf = 0;
  for (int i = 0; i < nb; ++i) {
    const uint8_t* restrict q4 = x[i].ql;
    const uint8_t* restrict qh = x[i].qh;
    const int8_t* restrict q8 = y[i].qs;
    memset(aux32, 0, 8 * sizeof(int32_t));
    int8_t* restrict a = aux8;
    for (int j = 0; j < QK_K; j += 128) {
      for (int l = 0; l < 32; ++l) {
        a[l + 0] = (int8_t)((q4[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
        a[l + 32] = (int8_t)((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
        a[l + 64] = (int8_t)((q4[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        a[l + 96] = (int8_t)((q4[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
      }
      a += 128;
      q4 += 64;
      qh += 32;
    }
    a = aux8;
    int is = 0;
    for (int j = 0; j < QK_K / 16; ++j) {
      int scale = x[i].scales[is++];
      for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
      for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
      q8 += 8;
      a += 8;
      for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
      for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
      q8 += 8;
      a += 8;
    }
    const float d = NE_FP16_TO_FP32(x[i].d) * y[i].d;
    for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
  }
  for (int l = 0; l < 8; ++l) sumf += sums[l];
  *s = sumf;
#endif
}

#else

static void ggml_vec_dot_q6_K_q8_K(const int n, float* restrict s, const void* restrict vx, const void* restrict vy) {
  assert(n % QK_K == 0);

  const block_q6_K* restrict x = vx;
  const block_q8_K* restrict y = vy;

  const int nb = n / QK_K;

#ifdef __ARM_NEON

  float sum = 0;

  const uint8x16_t m4b = vdupq_n_u8(0xF);
  const int8x16_t m32s = vdupq_n_s8(32);
#if defined(__ARM_FEATURE_DOTPROD)
  const int32x4_t vzero = vdupq_n_s32(0);
#endif

  const uint8x16_t mone = vdupq_n_u8(3);

  ggml_int8x16x4_t q6bytes;
  ggml_uint8x16x4_t q6h;

  for (int i = 0; i < nb; ++i) {
    const float d_all = (float)x[i].d;

    const uint8_t* restrict q6 = x[i].ql;
    const uint8_t* restrict qh = x[i].qh;
    const int8_t* restrict q8 = y[i].qs;

    const int8_t* restrict scale = x[i].scales;

    int32_t isum = 0;

    uint8x16_t qhbits = vld1q_u8(qh);
    ggml_uint8x16x2_t q6bits = ggml_vld1q_u8_x2(q6);
    ggml_int8x16x4_t q8bytes = ggml_vld1q_s8_x4(q8);

    q6h.val[0] = vshlq_n_u8(vandq_u8(mone, qhbits), 4);
    uint8x16_t shifted = vshrq_n_u8(qhbits, 2);
    q6h.val[1] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
    shifted = vshrq_n_u8(qhbits, 4);
    q6h.val[2] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
    shifted = vshrq_n_u8(qhbits, 6);
    q6h.val[3] = vshlq_n_u8(vandq_u8(mone, shifted), 4);

    q6bytes.val[0] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[0], m4b), q6h.val[0])), m32s);
    q6bytes.val[1] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[1], m4b), q6h.val[1])), m32s);
    q6bytes.val[2] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[0], 4), q6h.val[2])), m32s);
    q6bytes.val[3] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[1], 4), q6h.val[3])), m32s);

#if defined(__ARM_FEATURE_DOTPROD)

    isum += vaddvq_s32(vdotq_s32(vzero, q6bytes.val[0], q8bytes.val[0])) * scale[0] +
            vaddvq_s32(vdotq_s32(vzero, q6bytes.val[1], q8bytes.val[1])) * scale[1] +
            vaddvq_s32(vdotq_s32(vzero, q6bytes.val[2], q8bytes.val[2])) * scale[2] +
            vaddvq_s32(vdotq_s32(vzero, q6bytes.val[3], q8bytes.val[3])) * scale[3];
#else

    int16x8_t p0 = vaddq_s16(vmull_s8(vget_low_s8(q6bytes.val[0]), vget_low_s8(q8bytes.val[0])),
                             vmull_s8(vget_high_s8(q6bytes.val[0]), vget_high_s8(q8bytes.val[0])));
    int16x8_t p1 = vaddq_s16(vmull_s8(vget_low_s8(q6bytes.val[1]), vget_low_s8(q8bytes.val[1])),
                             vmull_s8(vget_high_s8(q6bytes.val[1]), vget_high_s8(q8bytes.val[1])));
    isum += vaddvq_s16(p0) * scale[0] + vaddvq_s16(p1) * scale[1];

    int16x8_t p2 = vaddq_s16(vmull_s8(vget_low_s8(q6bytes.val[2]), vget_low_s8(q8bytes.val[2])),
                             vmull_s8(vget_high_s8(q6bytes.val[2]), vget_high_s8(q8bytes.val[2])));
    int16x8_t p3 = vaddq_s16(vmull_s8(vget_low_s8(q6bytes.val[3]), vget_low_s8(q8bytes.val[3])),
                             vmull_s8(vget_high_s8(q6bytes.val[3]), vget_high_s8(q8bytes.val[3])));
    isum += vaddvq_s16(p2) * scale[2] + vaddvq_s16(p3) * scale[3];
#endif

    sum += isum * d_all * y[i].d;
  }
  *s = sum;

#elif defined __AVX2__

  const __m256i m4 = _mm256_set1_epi8(0xF);
  const __m256i m2 = _mm256_set1_epi8(3);
  const __m256i m32s = _mm256_set1_epi8(32);

  __m256 acc = _mm256_setzero_ps();

  for (int i = 0; i < nb; ++i) {
    const float d = y[i].d * NE_FP16_TO_FP32(x[i].d);

    const uint8_t* restrict q4 = x[i].ql;
    const uint8_t* restrict qh = x[i].qh;
    const int8_t* restrict q8 = y[i].qs;

    const __m64 scales_1 = _mm_set1_pi8(x[i].scales[0]);
    const __m64 scales_2 = _mm_set1_pi8(x[i].scales[1]);
    const __m64 scales_3 = _mm_set1_pi8(x[i].scales[2]);
    const __m64 scales_4 = _mm_set1_pi8(x[i].scales[3]);

    __m256i sumi = _mm256_setzero_si256();

    const __m128i scale_0 = _mm_set_epi64(scales_2, scales_1);
    const __m128i scale_1 = _mm_set_epi64(scales_4, scales_3);

    const __m256i q4bits1 = _mm256_loadu_si256((const __m256i*)q4);
    const __m128i q4bitsH = _mm_loadu_si128((const __m128i*)qh);

    const __m256i q4h_0 =
        _mm256_slli_epi16(_mm256_and_si256(MM256_SET_M128I(_mm_srli_epi16(q4bitsH, 2), q4bitsH), m2), 4);
    const __m256i q4h_1 = _mm256_slli_epi16(
        _mm256_and_si256(MM256_SET_M128I(_mm_srli_epi16(q4bitsH, 6), _mm_srli_epi16(q4bitsH, 4)), m2), 4);

    const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m4), q4h_0);
    const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m4), q4h_1);

    const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)(q8 + 0));
    const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)(q8 + 32));

    __m256i q8s_0 = _mm256_maddubs_epi16(m32s, q8_0);
    __m256i q8s_1 = _mm256_maddubs_epi16(m32s, q8_1);

    __m256i p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
    __m256i p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);

    p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
    p16_1 = _mm256_sub_epi16(p16_1, q8s_1);

    p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
    p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);

    sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));

    acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
  }

  *s = hsum_float_8(acc);

#elif defined __AVX__

  const __m128i m4 = _mm_set1_epi8(0xF);
  const __m128i m2 = _mm_set1_epi8(3);
  const __m128i m32s = _mm_set1_epi8(32);

  __m256 acc = _mm256_setzero_ps();

  for (int i = 0; i < nb; ++i) {
    const float d = y[i].d * NE_FP16_TO_FP32(x[i].d);

    const uint8_t* restrict q4 = x[i].ql;
    const uint8_t* restrict qh = x[i].qh;
    const int8_t* restrict q8 = y[i].qs;

    const __m64 scales_1 = _mm_set1_pi8(x[i].scales[0]);
    const __m64 scales_2 = _mm_set1_pi8(x[i].scales[1]);
    const __m64 scales_3 = _mm_set1_pi8(x[i].scales[2]);
    const __m64 scales_4 = _mm_set1_pi8(x[i].scales[3]);

    __m128i sumi_0 = _mm_setzero_si128();
    __m128i sumi_1 = _mm_setzero_si128();

    const __m128i scale_0 = _mm_set_epi64(scales_2, scales_1);
    const __m128i scale_1 = _mm_set_epi64(scales_4, scales_3);

    const __m256i q4bits1 = _mm256_loadu_si256((const __m256i*)q4);
    const __m128i q4bitsH = _mm_loadu_si128((const __m128i*)qh);

    const __m128i q4h_0 = _mm_slli_epi16(_mm_and_si128(q4bitsH, m2), 4);
    const __m128i q4h_1 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bitsH, 2), m2), 4);
    const __m128i q4h_2 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bitsH, 4), m2), 4);
    const __m128i q4h_3 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bitsH, 6), m2), 4);

    const __m128i q4_0 = _mm_or_si128(_mm_and_si128(_mm256_extractf128_si256(q4bits1, 0), m4), q4h_0);
    const __m128i q4_1 = _mm_or_si128(_mm_and_si128(_mm256_extractf128_si256(q4bits1, 1), m4), q4h_1);
    const __m128i q4_2 =
        _mm_or_si128(_mm_and_si128(_mm_srli_epi16(_mm256_extractf128_si256(q4bits1, 0), 4), m4), q4h_2);
    const __m128i q4_3 =
        _mm_or_si128(_mm_and_si128(_mm_srli_epi16(_mm256_extractf128_si256(q4bits1, 1), 4), m4), q4h_3);

    const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)(q8 + 0));
    const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)(q8 + 32));

    __m128i q8s_0 = _mm_maddubs_epi16(m32s, _mm256_extractf128_si256(q8_0, 0));
    __m128i q8s_1 = _mm_maddubs_epi16(m32s, _mm256_extractf128_si256(q8_0, 1));
    __m128i q8s_2 = _mm_maddubs_epi16(m32s, _mm256_extractf128_si256(q8_1, 0));
    __m128i q8s_3 = _mm_maddubs_epi16(m32s, _mm256_extractf128_si256(q8_1, 1));

    __m128i p16_0 = _mm_maddubs_epi16(q4_0, _mm256_extractf128_si256(q8_0, 0));
    __m128i p16_1 = _mm_maddubs_epi16(q4_1, _mm256_extractf128_si256(q8_0, 1));
    __m128i p16_2 = _mm_maddubs_epi16(q4_2, _mm256_extractf128_si256(q8_1, 0));
    __m128i p16_3 = _mm_maddubs_epi16(q4_3, _mm256_extractf128_si256(q8_1, 1));

    p16_0 = _mm_sub_epi16(p16_0, q8s_0);
    p16_1 = _mm_sub_epi16(p16_1, q8s_1);
    p16_2 = _mm_sub_epi16(p16_2, q8s_2);
    p16_3 = _mm_sub_epi16(p16_3, q8s_3);

    p16_0 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_0), p16_0);
    p16_1 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_unpackhi_epi64(scale_0, scale_0)), p16_1);
    p16_2 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_1), p16_2);
    p16_3 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_unpackhi_epi64(scale_1, scale_1)), p16_3);

    sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p16_0, p16_2));
    sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p16_1, p16_3));

    acc =
        _mm256_add_ps(_mm256_mul_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(MM256_SET_M128I(sumi_1, sumi_0))), acc);
  }

  *s = hsum_float_8(acc);

#elif defined __riscv_v_intrinsic

  float sumf = 0;

  for (int i = 0; i < nb; ++i) {
    const float d_all = (float)x[i].d;

    const uint8_t* restrict q6 = x[i].ql;
    const uint8_t* restrict qh = x[i].qh;
    const int8_t* restrict q8 = y[i].qs;

    const int8_t* restrict scale = x[i].scales;

    int32_t isum = 0;

    size_t vl = 16;

    vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);

    // load Q6
    vuint8mf2_t q6_0 = __riscv_vle8_v_u8mf2(q6, vl);
    vuint8mf2_t q6_1 = __riscv_vle8_v_u8mf2(q6 + 16, vl);

    // load qh
    vuint8mf2_t qh_x = __riscv_vle8_v_u8mf2(qh, vl);

    vuint8mf2_t qh0 = __riscv_vsll_vx_u8mf2(__riscv_vand_vx_u8mf2(qh_x, 0x3, vl), 0x4, vl);
    qh_x = __riscv_vsrl_vx_u8mf2(qh_x, 0x2, vl);
    vuint8mf2_t qh1 = __riscv_vsll_vx_u8mf2(__riscv_vand_vx_u8mf2(qh_x, 0x3, vl), 0x4, vl);
    qh_x = __riscv_vsrl_vx_u8mf2(qh_x, 0x2, vl);
    vuint8mf2_t qh2 = __riscv_vsll_vx_u8mf2(__riscv_vand_vx_u8mf2(qh_x, 0x3, vl), 0x4, vl);
    qh_x = __riscv_vsrl_vx_u8mf2(qh_x, 0x2, vl);
    vuint8mf2_t qh3 = __riscv_vsll_vx_u8mf2(__riscv_vand_vx_u8mf2(qh_x, 0x3, vl), 0x4, vl);

    vuint8mf2_t q6h_0 = __riscv_vor_vv_u8mf2(__riscv_vand_vx_u8mf2(q6_0, 0xF, vl), qh0, vl);
    vuint8mf2_t q6h_1 = __riscv_vor_vv_u8mf2(__riscv_vand_vx_u8mf2(q6_1, 0xF, vl), qh1, vl);
    vuint8mf2_t q6h_2 = __riscv_vor_vv_u8mf2(__riscv_vsrl_vx_u8mf2(q6_0, 0x4, vl), qh2, vl);
    vuint8mf2_t q6h_3 = __riscv_vor_vv_u8mf2(__riscv_vsrl_vx_u8mf2(q6_1, 0x4, vl), qh3, vl);

    vint8mf2_t q6v_0 = __riscv_vsub_vx_i8mf2(__riscv_vreinterpret_v_u8mf2_i8mf2(q6h_0), 32, vl);
    vint8mf2_t q6v_1 = __riscv_vsub_vx_i8mf2(__riscv_vreinterpret_v_u8mf2_i8mf2(q6h_1), 32, vl);
    vint8mf2_t q6v_2 = __riscv_vsub_vx_i8mf2(__riscv_vreinterpret_v_u8mf2_i8mf2(q6h_2), 32, vl);
    vint8mf2_t q6v_3 = __riscv_vsub_vx_i8mf2(__riscv_vreinterpret_v_u8mf2_i8mf2(q6h_3), 32, vl);

    // load Q8 and take product
    vint16m1_t p0 = __riscv_vwmul_vv_i16m1(q6v_0, __riscv_vle8_v_i8mf2(q8, vl), vl);
    vint16m1_t p1 = __riscv_vwmul_vv_i16m1(q6v_1, __riscv_vle8_v_i8mf2(q8 + 16, vl), vl);
    vint16m1_t p2 = __riscv_vwmul_vv_i16m1(q6v_2, __riscv_vle8_v_i8mf2(q8 + 32, vl), vl);
    vint16m1_t p3 = __riscv_vwmul_vv_i16m1(q6v_3, __riscv_vle8_v_i8mf2(q8 + 48, vl), vl);

    vint32m1_t vs_0 = __riscv_vwredsum_vs_i16m1_i32m1(p0, vzero, vl);
    vint32m1_t vs_1 = __riscv_vwredsum_vs_i16m1_i32m1(p1, vzero, vl);
    vint32m1_t vs_2 = __riscv_vwredsum_vs_i16m1_i32m1(p2, vzero, vl);
    vint32m1_t vs_3 = __riscv_vwredsum_vs_i16m1_i32m1(p3, vzero, vl);

    isum += __riscv_vmv_x_s_i32m1_i32(vs_0) * scale[0];
    isum += __riscv_vmv_x_s_i32m1_i32(vs_1) * scale[1];
    isum += __riscv_vmv_x_s_i32m1_i32(vs_2) * scale[2];
    isum += __riscv_vmv_x_s_i32m1_i32(vs_3) * scale[3];

    sumf += isum * d_all * y[i].d;
  }

  *s = sumf;

#else

  int8_t aux8[QK_K];
  int16_t aux16[8];
  float sums[8];
  int32_t aux32[8];
  memset(sums, 0, 8 * sizeof(float));

  float sumf = 0;
  for (int i = 0; i < nb; ++i) {
    const uint8_t* restrict q4 = x[i].ql;
    const uint8_t* restrict qh = x[i].qh;
    const int8_t* restrict q8 = y[i].qs;
    memset(aux32, 0, 8 * sizeof(int32_t));
    int8_t* restrict a = aux8;
    for (int l = 0; l < 16; ++l) {
      a[l + 0] = (int8_t)((q4[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
      a[l + 16] = (int8_t)((q4[l + 16] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
      a[l + 32] = (int8_t)((q4[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
      a[l + 48] = (int8_t)((q4[l + 16] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
    }
    int is = 0;
    for (int j = 0; j < QK_K / 16; ++j) {
      int scale = x[i].scales[is++];
      for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
      for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
      q8 += 8;
      a += 8;
      for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
      for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
      q8 += 8;
      a += 8;
    }
    const float d = NE_FP16_TO_FP32(x[i].d) * y[i].d;
    for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
  }
  for (int l = 0; l < 8; ++l) sumf += sums[l];
  *s = sumf;
#endif
}

#endif

#ifdef __cplusplus
}
#endif
