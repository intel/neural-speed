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
#include "layers/mha_dense.h"

#include <immintrin.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <random>
#include <vector>

#ifdef NS_TESTS
#include <memory>
#include <tuple>

#include "layers/ne_test_layers_utils.hpp"
#endif

#include "core/data_types.h"
#include "mha_dense_wrapper.h"

using namespace bestla;       // NOLINT
using namespace ne_bestla;    // NOLINT
using namespace custom::mha;  // NOLINT

void bestla_fusion_attn_bf16_forward(const attn_bf16_fwd_args_t* params) {
  return bestla_fusion_attn_forward(*reinterpret_cast<const attn_fwd_args_t<bf16, bf16, bf16, bf16>*>(params));
}

bool bestla_fusion_attn_fp32_fp16_fp16_fp32_support(const attn_shape_t* params) {
#if CompileBF16()
  GetCPUDevice();
  // TODO(Yi): check K V's layout
  return _cd->AMX_BF16();
#endif
  return false;
}
void bestla_fusion_attn_fp32_fp16_fp16_fp32_forward(const attn_fp32_fp16_fp16_fp32_fwd_args_t* params) {
  return bestla_fusion_attn_forward(*reinterpret_cast<const attn_fwd_args_t<float, fp16, fp16, float>*>(params));
  // return bestla_fusion_attn_forward_ref(*reinterpret_cast<const attn_fwd_args_t<float, fp16, fp16, float>*>(params));
}

bool bestla_fusion_attn_fp16_support(const attn_shape_t* params) {
#if CompileFP16()
  GetCPUDevice();
  // TODO(Yi): check K V's layout
  return _cd->AMX_BF16();
#endif
  return false;
}
void bestla_fusion_attn_fp16_forward(const attn_fp16_fwd_args_t* params) {
  return bestla_fusion_attn_forward<fp16, fp16, fp16, fp16>(
      *reinterpret_cast<const attn_fwd_args_t<fp16, fp16, fp16, fp16>*>(params));
}
void bestla_fusion_attn_int8_forward(const attn_int8_fwd_args_t* params) {
  return bestla_fusion_attn_forward<int8_t, int8_t, int8_t, int8_t>(
      *reinterpret_cast<const attn_fwd_args_t<int8_t, int8_t, int8_t, int8_t>*>(params));
}
size_t bestla_fusion_attn_workspace_size(const attn_shape_t* params) {
  const auto& p = *params;  // TODO(Yi): Better way to get tmp size?
  return size_t(ne_threading::get()->num_threads() * sizeof(float) * 16) * padto(padto(p.sl_kv, 48), 64);
}

bool bestla_reordered_attn_fp32_support(const attn_shape_t* params) {
  GetCPUDevice();
#if CompileBF16()
  // TODO(Yi): check K V's layout
  if (_cd->AMX_BF16()) return true;
#endif
  return _cd->AVX512F() || _cd->AVX2();  // use avx2 and f16c on avx2 platforms
}
// kv cache sizes in bytes per layer per batch per beam for;
void bestla_reordered_attn_fp32_batch_kv_info(const kv_shape_t* params, kv_cache_info_t* out) {
  GetCPUDevice();
  // use bf16 for kv-cache
  const auto p = *params;
  int n_tile = 0, row_pad = 0, elt_size = 0;
  if (_cd->AVX512F()) {
    out->k_layout = ATTN_FWD_LAYOUT_NTILE48_ROWPACK2;
    out->v_layout = ATTN_FWD_LAYOUT_NTILE48_ROWPACK2;
    n_tile = 48;
    row_pad = 32;
    elt_size = sizeof(bf16);
  } else if (_cd->AVX2()) {
    out->k_layout = ATTN_FWD_LAYOUT_NTILE24_ROWPACK1;
    out->v_layout = ATTN_FWD_LAYOUT_NTILE24_ROWPACK1;
    n_tile = 24;
    row_pad = 1;
    elt_size = sizeof(fp16);
  } else {
    assert(false);
  }

  out->stride_k_head_size = elt_size * n_tile;
  out->stride_k_sl = elt_size * padto(static_cast<int>(p.head_size), row_pad);
  out->stride_k_head_num = out->stride_k_sl * padto(static_cast<int>(p.sl_kv_max), n_tile);
  out->k_bytes = out->stride_k_head_num * static_cast<size_t>(p.heads_kv);

  out->stride_v_sl = elt_size * n_tile;
  out->stride_v_head_size = elt_size * padto(static_cast<int>(p.sl_kv_max), row_pad);
  out->stride_v_head_num = out->stride_v_head_size * padto(static_cast<int>(p.head_size), n_tile);
  out->v_bytes = out->stride_v_head_num * static_cast<size_t>(p.heads_kv);
}
template <ATTN_FWD_LAYOUT KV_LAYOUT>
void bestla_reordered_attn_fp32_forward_(const bestla_reordered_attn_fp32_fp32_fwd_args_t* params) {
  using kv_t = std::conditional_t<  //
      KV_LAYOUT == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2, bf16,
      std::conditional_t<  //
          KV_LAYOUT == ATTN_FWD_LAYOUT_NTILE48_ROWPACK4, int8_t,
          std::conditional_t<  //
              KV_LAYOUT == ATTN_FWD_LAYOUT_NTILE24_ROWPACK1, fp16, void>>>;
  const auto n_tile = KV_LAYOUT == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2   ? 48
                      : KV_LAYOUT == ATTN_FWD_LAYOUT_NTILE48_ROWPACK4 ? 48
                      : KV_LAYOUT == ATTN_FWD_LAYOUT_NTILE24_ROWPACK1 ? 24
                                                                      : 1;

  const attn_fwd_args_t<float, kv_t, kv_t, float> bestla_params = {
      /* .Q = */ params->Q,
      /* .K = */ reinterpret_cast<kv_t*>(params->K),
      /* .V = */ reinterpret_cast<kv_t*>(params->V),
      /* .dst = */ params->dst,
      /* .Q_sc = */ params->Q_sc,
      /* .K_sc = */ params->K_sc,
      /* .V_sc = */ params->V_sc,
      /* .dst_sc = */ params->dst_sc,
      /* .tmp = */ params->tmp,
      /* .QK_scale = */ params->QK_scale,
      /* .attn_flags = */ params->attn_flags,
      /* .batch_size = */ params->batch_size,
      /* .head_num = */ params->head_num,
      /* .heads_kv = */ params->heads_kv,
      /* .head_size = */ params->head_size,
      /* .sl_q = */ params->sl_q,
      /* .sl_kv = */ params->sl_kv,
      /* .Q_layout = */ params->Q_layout,
      /* .K_layout = */ params->K_layout,
      /* .V_layout = */ params->V_layout,
      /* .dst_layout = */ params->dst_layout,
      /* .step_q_bs = */ params->step_q_bs,
      /* .step_q_head_num = */ params->step_q_head_num,
      /* .step_q_sl = */ params->step_q_sl,
      /* .step_k_bs = */ static_cast<int>(params->stride_k_bs / sizeof(kv_t)),
      /* .step_k_head_num = */ static_cast<int>(params->stride_k_head_num / sizeof(kv_t)),
      /* .step_k_sl = */ static_cast<int>(params->stride_k_sl / sizeof(kv_t)),
      /* .step_k_head_size = */ n_tile,
      /* .step_v_bs = */ static_cast<int>(params->stride_v_bs / sizeof(kv_t)),
      /* .step_v_head_num = */ static_cast<int>(params->stride_v_head_num / sizeof(kv_t)),
      /* .step_v_sl = */ n_tile,
      /* .step_v_head_size = */ static_cast<int>(params->stride_v_head_size / sizeof(kv_t)),
      /* .step_dst_bs = */ params->step_dst_bs,
      /* .step_dst_head_num = */ params->step_dst_head_num,
      /* .step_dst_sl = */ params->step_dst_sl,
  };
  return bestla_fusion_attn_forward<float, kv_t, kv_t, float>(bestla_params);
}

void bestla_reordered_attn_fp32_forward(const bestla_reordered_attn_fp32_fp32_fwd_args_t* params) {
  assert(params->K_layout == params->V_layout);
  switch (params->K_layout) {
    case ATTN_FWD_LAYOUT_NTILE48_ROWPACK2:
      return bestla_reordered_attn_fp32_forward_<ATTN_FWD_LAYOUT_NTILE48_ROWPACK2>(params);
    case ATTN_FWD_LAYOUT_NTILE24_ROWPACK1:
      return bestla_reordered_attn_fp32_forward_<ATTN_FWD_LAYOUT_NTILE24_ROWPACK1>(params);
    // case ATTN_FWD_LAYOUT_NTILE48_ROWPACK4:
    //   return bestla_reordered_attn_fp32_forward_<ATTN_FWD_LAYOUT_NTILE48_ROWPACK4>(params);
    default:
      assert(false);
      break;
  }
}

template <bool zero_padding>
void bestla_reordered_attn_fp32_update_k_48x2(const bestla_fusion_attn_fp32_update_kv_args_t* params) {
  const auto p = *params;
  NE_ASSERT(p.step_head_size == 1);
  const auto pad_headsize = padto(p.head_size, 32);
  const auto pad_seq_max = padto(p.seq_max, 48);
  const auto cache_step_head_num = pad_headsize * pad_seq_max;
  const auto cache_step_bs = p.heads_kv * cache_step_head_num;
  GetCPUDevice();
  const bool use_jit = _cd->AVX512_BF16() && (p.seq_off == 0) && zero_padding;

#pragma omp parallel for collapse(2)
  for (int ibs = 0; ibs < p.batch_size; ++ibs) {
    for (int ihn = 0; ihn < p.heads_kv; ++ihn) {
      const auto dst = reinterpret_cast<bf16*>(p.cache) + ibs * cache_step_bs + ihn * cache_step_head_num;
      const auto src = p.src + ibs * p.step_bs + ihn * p.step_head_num;

      if (use_jit) {
        kernel::jit::PaddingTransInterleaveCvt::forward<48>(  //
            src, dst, p.seq_size, p.head_size, padto(p.seq_size, 48), padto(p.head_size, 32), p.step_seq, pad_headsize);
      } else {
        for (int i = 0; i < p.seq_size; ++i) {      // QK_GEMM should not require 0-padding on seq_kv (i.e. N-dim)
          for (int j = 0; j < pad_headsize; ++j) {  // K-dim padding for QK_GEMM
            const auto i_dst = p.seq_off + i;
            const auto ii = i_dst % 48;
            const auto i_blk = i_dst - ii;
            const auto jj = j % 2;
            const auto j_blk = j - jj;
            if constexpr (zero_padding) {
              dst[i_blk * pad_headsize + ii * 2 + j_blk * 48 + jj] =
                  j < p.head_size ? static_cast<bf16>(src[i * p.step_seq + j]) : bf16(0);
            } else {
              if (j < p.head_size)
                dst[i_blk * pad_headsize + ii * 2 + j_blk * 48 + jj] = static_cast<bf16>(src[i * p.step_seq + j]);
            }
          }
        }
      }
    }
  }
}

template <bool zero_padding>
void bestla_reordered_attn_fp32_update_k_24x1(const bestla_fusion_attn_fp32_update_kv_args_t* params) {
  const auto p = *params;
  NE_ASSERT(p.step_head_size == 1);
  const auto pad_headsize = padto(p.head_size, 1);
  const auto pad_seq_max = padto(p.seq_max, 24);
  const auto cache_step_head_num = pad_headsize * pad_seq_max;
  const auto cache_step_bs = p.heads_kv * cache_step_head_num;

  const int n_para = p.batch_size * p.heads_kv;
  // #pragma omp parallel
  for (int i_para = 0; i_para < n_para; ++i_para) {
    const int ibs = i_para / p.heads_kv;
    const int ihn = i_para % p.heads_kv;

    const auto dst = reinterpret_cast<fp16*>(p.cache) + ibs * cache_step_bs + ihn * cache_step_head_num;
    const auto src = p.src + ibs * p.step_bs + ihn * p.step_head_num;

    if (p.seq_off == 0 && p.head_size % 8 == 0 && zero_padding) {
      int i = 0;
      for (; i < padto_le(p.seq_size, 8); i += 8) {  // QK_GEMM should not require 0-padding on seq_kv (i.e. N-dim)
        for (int j = 0; j < pad_headsize; j += 8) {  // K-dim padding for QK_GEMM
          const auto i_dst = p.seq_off + i;
          const auto ii = i_dst % 24;
          const auto i_blk = i_dst - ii;
          const auto mm_dst = kernel::avx2::load_fp32_fp16_tr_x8_word<8>(src + i * p.step_seq + j, p.step_seq);
          for (int jj = 0; jj < 8; ++jj)
            _mm_store_si128(reinterpret_cast<__m128i*>(dst + i_blk * pad_headsize + ii + (j + jj) * 24), mm_dst[jj]);
        }
      }
      if (i < p.seq_size) {
        for (int j = 0; j < pad_headsize; j += 8) {  // K-dim padding for QK_GEMM
          const auto i_dst = p.seq_off + i;
          const auto ii = i_dst % 24;
          const auto i_blk = i_dst - ii;
          const auto mm_dst =
              kernel::avx2::load_fp32_fp16_tr_x8_word_tbl[p.seq_size - i](src + i * p.step_seq + j, p.step_seq);
          for (int jj = 0; jj < 8; ++jj)
            _mm_store_si128(reinterpret_cast<__m128i*>(dst + i_blk * pad_headsize + ii + (j + jj) * 24), mm_dst[jj]);
        }
      }
    } else {
      for (int i = 0; i < p.seq_size; ++i) {      // QK_GEMM should not require 0-padding on seq_kv (i.e. N-dim)
        for (int j = 0; j < pad_headsize; ++j) {  // K-dim padding for QK_GEMM
          const auto i_dst = p.seq_off + i;
          const auto ii = i_dst % 24;
          const auto i_blk = i_dst - ii;
          if constexpr (zero_padding) {
            dst[i_blk * pad_headsize + ii + j * 24] = utils::bit_cast<fp16, int16_t>(
                j < p.head_size ? NE_FP32_TO_FP16(src[i * p.step_seq + j]) : NE_FP32_TO_FP16(0.f));
          } else {
            if (j < p.head_size)
              dst[i_blk * pad_headsize + ii + j * 24] =
                  utils::bit_cast<fp16, int16_t>(NE_FP32_TO_FP16(src[i * p.step_seq + j]));
          }
        }
      }
    }
  }
}

template <bool zero_padding>
void bestla_reordered_attn_fp32_update_k_(const bestla_fusion_attn_fp32_update_kv_args_t* params) {
  GetCPUDevice();
  if (_cd->AVX512F()) {
    // ATTN_FWD_LAYOUT_NTILE48_ROWPACK2
    return bestla_reordered_attn_fp32_update_k_48x2<zero_padding>(params);
  } else if (_cd->AVX2()) {
    // ATTN_FWD_LAYOUT_NTILE24_ROWPACK1
    return bestla_reordered_attn_fp32_update_k_24x1<zero_padding>(params);
  } else {
    assert(false);
  }
}

void bestla_reordered_attn_fp32_update_k(const bestla_fusion_attn_fp32_update_kv_args_t* params) {
  return params->no_zeroing ? bestla_reordered_attn_fp32_update_k_<false>(params)
                            : bestla_reordered_attn_fp32_update_k_<true>(params);
}

template <bool zero_padding>
void bestla_reordered_attn_fp32_update_v_48x2(const bestla_fusion_attn_fp32_update_kv_args_t* params) {
  const auto p = *params;
  NE_ASSERT(p.step_head_size == 1);
  const auto pad_headsize = padto(p.head_size, 48);
  const auto pad_seq_max = padto(p.seq_max, 32);
  const auto step_cache_head_num = pad_headsize * pad_seq_max;
  const auto step_cache_bs = p.heads_kv * step_cache_head_num;
  GetCPUDevice();
  const bool use_jit = _cd->AVX512_BF16() && (p.seq_off == 0) && zero_padding;

#pragma omp parallel for collapse(2)
  for (int ibs = 0; ibs < p.batch_size; ++ibs) {
    for (int ihn = 0; ihn < p.heads_kv; ++ihn) {
      const auto dst = reinterpret_cast<bf16*>(p.cache) + ibs * step_cache_bs + ihn * step_cache_head_num;
      const auto src = p.src + ibs * p.step_bs + ihn * p.step_head_num;
      if (use_jit) {
        kernel::jit::PaddingInterleaveCvt::forward<48>(  //
            src, dst, p.seq_size, p.head_size, padto(p.seq_size, 32), padto(p.head_size, 48), p.step_seq, pad_seq_max);
      } else {
        for (int i = 0; i < padto(p.seq_off + p.seq_size, 32) - p.seq_off; ++i) {  // K-dim padding for PV_GEMM
          for (int j = 0; j < p.head_size; ++j) {  // PV_GEMM shouldn't require 0-padding on head_size (i.e. N-dim)
            const auto i_dst = p.seq_off + i;
            const auto ii = i_dst % 2;
            const auto i_blk = i_dst - ii;
            const auto jj = j % 48;
            const auto j_blk = j - jj;
            if constexpr (zero_padding) {
              dst[i_blk * 48 + ii + j_blk * pad_seq_max + jj * 2] =
                  i < p.seq_size ? static_cast<bf16>(src[i * p.step_seq + j]) : bf16(0);
            } else {
              if (i < p.seq_size)
                dst[i_blk * 48 + ii + j_blk * pad_seq_max + jj * 2] = static_cast<bf16>(src[i * p.step_seq + j]);
            }
          }
        }
      }
    }
  }
}
template <bool zero_padding>
void bestla_reordered_attn_fp32_update_v_24x1(const bestla_fusion_attn_fp32_update_kv_args_t* params) {
  const auto p = *params;
  NE_ASSERT(p.step_head_size == 1);
  const auto pad_headsize = padto(p.head_size, 24);
  const auto pad_seq_max = padto(p.seq_max, 1);
  const auto step_cache_head_num = pad_headsize * pad_seq_max;
  const auto step_cache_bs = p.heads_kv * step_cache_head_num;

  const int n_para = p.batch_size * p.heads_kv;
#pragma omp parallel
  for (int i_para = 0; i_para < n_para; ++i_para) {
    const int ibs = i_para / p.heads_kv;
    const int ihn = i_para % p.heads_kv;
    const auto dst = reinterpret_cast<fp16*>(p.cache) + ibs * step_cache_bs + ihn * step_cache_head_num;
    const auto src = p.src + ibs * p.step_bs + ihn * p.step_head_num;
    if (p.seq_off == 0 && p.head_size % 8 == 0) {
      for (int i = 0; i < p.seq_size; ++i) {        // K-dim padding for PV_GEMM
        for (int j = 0; j < p.head_size; j += 8) {  // PV_GEMM shouldn't require 0-padding on head_size (i.e. N-dim)
          const auto jj = j % 24;
          const auto j_blk = j - jj;
          const auto dst_m128 = _mm256_cvtps_ph(_mm256_load_ps(src + i * p.step_seq + j), _MM_FROUND_TO_NEAREST_INT);
          _mm_store_si128(reinterpret_cast<__m128i*>(dst + i * 24 + j_blk * pad_seq_max + jj), dst_m128);
        }
      }
    } else {
      for (int i = 0; i < p.seq_size; ++i) {     // K-dim padding for PV_GEMM
        for (int j = 0; j < p.head_size; ++j) {  // PV_GEMM shouldn't require 0-padding on head_size (i.e. N-dim)
          const auto i_dst = p.seq_off + i;
          const auto jj = j % 24;
          const auto j_blk = j - jj;
          if constexpr (zero_padding) {
            dst[i_dst * 24 + j_blk * pad_seq_max + jj] = utils::bit_cast<fp16, int16_t>(
                i < p.seq_size ? NE_FP32_TO_FP16(src[i * p.step_seq + j]) : NE_FP32_TO_FP16(0));
          } else {
            if (i < p.seq_size)
              dst[i_dst * 24 + j_blk * pad_seq_max + jj] =
                  utils::bit_cast<fp16, int16_t>(NE_FP32_TO_FP16(src[i * p.step_seq + j]));
          }
        }
      }
    }
  }
}
template <bool zero_padding>
void bestla_reordered_attn_fp32_update_v_(const bestla_fusion_attn_fp32_update_kv_args_t* params) {
  GetCPUDevice();
  if (_cd->AVX512F()) {
    // ATTN_FWD_LAYOUT_NTILE48_ROWPACK2
    return bestla_reordered_attn_fp32_update_v_48x2<zero_padding>(params);
  } else if (_cd->AVX2()) {
    // ATTN_FWD_LAYOUT_NTILE24_ROWPACK1
    return bestla_reordered_attn_fp32_update_v_24x1<zero_padding>(params);
  } else {
    assert(false);
  }
}
void bestla_reordered_attn_fp32_update_v(const bestla_fusion_attn_fp32_update_kv_args_t* params) {
  return params->no_zeroing ? bestla_reordered_attn_fp32_update_v_<false>(params)
                            : bestla_reordered_attn_fp32_update_v_<true>(params);
}

void bestla_reordered_attn_fp32_shift_rope_k(char* cache, const ne_fp16_t* cossin, int batch_size, int heads_kv,
                                             int head_size, int seq_max, int seq_keep) {
  const auto pad_headsize = padto(head_size, 32);
  const auto pad_seq_max = padto(seq_max, 48);
  const auto cache_step_head_num = pad_headsize * pad_seq_max;
  const auto cache_step_bs = heads_kv * cache_step_head_num;

#pragma omp parallel for collapse(2)
  for (int ibs = 0; ibs < batch_size; ++ibs)
    for (int ihn = 0; ihn < heads_kv; ++ihn) {
      const auto src = reinterpret_cast<bf16*>(cache) + ibs * cache_step_bs + ihn * cache_step_head_num;
      kernel::jit::CScaleInterleavedBF16FP16::forward<48>(  // NOLINT [build/include_what_you_use]
          src, reinterpret_cast<const fp16*>(cossin), head_size, pad_seq_max, pad_headsize, seq_keep);
    }
}

template <bool zero_padding>
void bestla_fusion_attn_fp32_batch_cpy_k_(const bestla_fusion_attn_fp32_batch_cpy_kv_args_t* params) {
  GetCPUDevice();
  assert(_cd->AVX512F());  // TODO(Yi): add avx2 implementation
  static constexpr auto N_TILE = 48;
  static constexpr auto K_TILE = 32;
  static constexpr auto K_PACK = 2;
  const auto p = *params;
  const auto pad_headsize = padto(p.head_size, K_TILE);
  const auto pad_seq_max = padto(p.seq_max, N_TILE);
  const auto step_head_num = pad_headsize * pad_seq_max;

  const auto seq_unaligned = std::min(padto(p.seq_off, N_TILE) - p.seq_off, p.seq_size);
  const auto size_aligned_cpy = pad_headsize * (padto(p.seq_off + p.seq_size, N_TILE) - padto(p.seq_off, N_TILE));
#pragma omp parallel for
  for (int ihn = 0; ihn < p.heads_kv; ++ihn) {
    const auto dst = reinterpret_cast<bf16*>(p.dst) + ihn * step_head_num;
    const auto src = reinterpret_cast<bf16*>(p.src) + ihn * step_head_num;

    if (seq_unaligned) {
      const auto ii = p.seq_off % N_TILE;
      const auto i_blk = p.seq_off - ii;
      const auto off = i_blk * pad_headsize + ii * K_PACK;
      for (int j = 0; j < pad_headsize; j += K_PACK) {  // K-dim padding for QK_GEMM
        memcpy(dst + off + j * N_TILE, src + off + j * N_TILE, sizeof(bf16) * K_PACK * seq_unaligned);
      }
    }
    if constexpr (zero_padding) {
      if (size_aligned_cpy) {
        const auto off = padto(p.seq_off, N_TILE) * pad_headsize;
        memcpy(dst + off, src + off, sizeof(bf16) * size_aligned_cpy);
      }
    } else {
      assert(("Unimplemented!", false));
    }
  }
}
void bestla_fusion_attn_fp32_batch_cpy_k(const bestla_fusion_attn_fp32_batch_cpy_kv_args_t* params) {
  return params->no_zeroing ? bestla_fusion_attn_fp32_batch_cpy_k_<false>(params)
                            : bestla_fusion_attn_fp32_batch_cpy_k_<true>(params);
}

template <bool zero_padding>
void bestla_fusion_attn_fp32_batch_cpy_v_(const bestla_fusion_attn_fp32_batch_cpy_kv_args_t* params) {
  GetCPUDevice();
  assert(_cd->AVX512F());  // TODO(Yi): add avx2 implementation
  static constexpr auto N_TILE = 48;
  static constexpr auto K_TILE = 32;
  static constexpr auto K_PACK = 2;
  const auto p = *params;
  const auto pad_headsize = padto(p.head_size, N_TILE);
  const auto pad_seq_max = padto(p.seq_max, K_TILE);
  const auto step_head_num = pad_headsize * pad_seq_max;

  const auto seq_off_aligned = padto(p.seq_off, K_PACK);
  const auto seq_end_aligned = padto(p.seq_off + p.seq_size, K_TILE);
  const auto seq_size_aligned = seq_end_aligned - seq_off_aligned;
#pragma omp parallel for collapse(2)
  for (int ihn = 0; ihn < p.heads_kv; ++ihn) {
    for (int j = 0; j < p.head_size; j += N_TILE) {
      const auto dst = reinterpret_cast<bf16*>(p.dst) + ihn * step_head_num + pad_seq_max * j;
      const auto src = reinterpret_cast<bf16*>(p.src) + ihn * step_head_num + pad_seq_max * j;
      if (p.seq_off != seq_off_aligned) {  // seq_size_unaligen must be 0 or 1 as K_PACK = 2
        const auto off = (seq_off_aligned - K_PACK) * N_TILE + 1;
        for (int jj = 0; jj < N_TILE; ++jj) dst[off + jj * K_PACK] = src[off + jj * K_PACK];
      }
      if constexpr (zero_padding) {
        if (seq_off_aligned != seq_end_aligned) {
          const auto off = seq_off_aligned * N_TILE;
          memcpy(dst + off, src + off, sizeof(bf16) * N_TILE * seq_size_aligned);
        }
      } else {
        assert(("Unimplemented!", false));
      }
    }
  }
}
void bestla_fusion_attn_fp32_batch_cpy_v(const bestla_fusion_attn_fp32_batch_cpy_kv_args_t* params) {
  return params->no_zeroing ? bestla_fusion_attn_fp32_batch_cpy_v_<false>(params)
                            : bestla_fusion_attn_fp32_batch_cpy_v_<true>(params);
}

// #ifdef __GNUC__
// #pragma GCC pop_options
// #endif

#ifdef NS_TESTS
#define CheckISA(ISA) \
  (bestla::device::CpuDevice::getInstance()->ISA() || (printf("Wrong Device ISA: " #ISA "\n"), false))

namespace {
bool ret_ok = true;

class TestMhaDese {
 public:
  TestMhaDese() {
    printf("Test suit: %s\n", __FUNCTION__);
    GetCPUDevice();
    ne_threading::get()->set_threads(std::min(_cd->getThreads(), omp_get_max_threads()));

#if CompileFP16()
    if (CheckISA(AMX_BF16)) {
      ret_ok &= test_case<float, fp16, fp16, float>({1, 1, 1, 32, 128, 64}, NE_ATTN_FLAG_NONE);
      ret_ok &= test_case<float, fp16, fp16, float>({2, 5, 5, 32, 64, 128}, NE_ATTN_FLAG_NONE);
      ret_ok &= test_case<float, fp16, fp16, float>({2, 5, 5, 80, 128, 77}, NE_ATTN_FLAG_NONE);
      ret_ok &= test_case<float, fp16, fp16, float>({1, 1, 1, 32, 63, 63}, NE_ATTN_FLAG_NONE);
      ret_ok &= test_case<float, fp16, fp16, float>({3, 4, 4, 256, 1, 384}, NE_ATTN_FLAG_NONE);
      ret_ok &= test_case<float, fp16, fp16, float>({1, 1, 1, 64, 64, 64}, NE_ATTN_FLAG_IS_CAUSAL);

      ret_ok &= test_case<fp16, fp16, fp16, fp16>({1, 1, 1, 32, 128, 64}, NE_ATTN_FLAG_NONE, true);
      ret_ok &= test_case<fp16, fp16, fp16, fp16>({2, 5, 5, 32, 64, 128}, NE_ATTN_FLAG_NONE, true);
      ret_ok &= test_case<fp16, fp16, fp16, fp16>({2, 5, 5, 80, 128, 77}, NE_ATTN_FLAG_NONE, true);
      ret_ok &= test_case<fp16, fp16, fp16, fp16>({1, 1, 1, 256, 63, 63}, NE_ATTN_FLAG_NONE, true);
      ret_ok &= test_case<fp16, fp16, fp16, fp16>({3, 4, 4, 256, 1, 384}, NE_ATTN_FLAG_NONE, true);
      ret_ok &= test_case<fp16, fp16, fp16, fp16>({1, 1, 1, 64, 64, 64}, NE_ATTN_FLAG_IS_CAUSAL, true);

      ret_ok &= test_case<float, fp16, fp16, float>({1, 1, 1, 32, 128, 64}, NE_ATTN_FLAG_NONE, true);
      ret_ok &= test_case<float, fp16, fp16, float>({2, 5, 5, 32, 64, 128}, NE_ATTN_FLAG_NONE, true);
      ret_ok &= test_case<float, fp16, fp16, float>({2, 5, 5, 80, 128, 77}, NE_ATTN_FLAG_NONE, true);
      ret_ok &= test_case<float, fp16, fp16, float>({1, 1, 1, 256, 63, 63}, NE_ATTN_FLAG_NONE, true);
      ret_ok &= test_case<float, fp16, fp16, float>({3, 4, 4, 256, 1, 384}, NE_ATTN_FLAG_NONE, true);
      ret_ok &= test_case<float, fp16, fp16, float>({1, 1, 1, 64, 64, 64}, NE_ATTN_FLAG_IS_CAUSAL, true);
    }
#endif

    if (CheckISA(AMX_BF16)) {
      const auto BA48b4a = ATTN_FWD_LAYOUT_NTILE48_ROWPACK4;
      ret_ok &= test_case<int8_t, int8_t, int8_t, int8_t>({1, 1, 1, 32, 128, 64}, NE_ATTN_FLAG_NONE, false, BA48b4a);
      ret_ok &= test_case<int8_t, int8_t, int8_t, int8_t>({2, 5, 5, 32, 64, 128}, NE_ATTN_FLAG_NONE, false, BA48b4a);
      ret_ok &= test_case<int8_t, int8_t, int8_t, int8_t>({2, 5, 5, 80, 128, 77}, NE_ATTN_FLAG_NONE, false, BA48b4a);
      ret_ok &= test_case<int8_t, int8_t, int8_t, int8_t>({1, 1, 1, 256, 63, 63}, NE_ATTN_FLAG_NONE, false, BA48b4a);
      ret_ok &= test_case<int8_t, int8_t, int8_t, int8_t>({3, 4, 4, 256, 1, 384}, NE_ATTN_FLAG_NONE, false, BA48b4a);
      ret_ok &=
          test_case<int8_t, int8_t, int8_t, int8_t>({1, 1, 1, 64, 64, 64}, NE_ATTN_FLAG_IS_CAUSAL, false, BA48b4a);
    }

    if (CheckISA(AMX_BF16)) {
      const auto BA48b2a = ATTN_FWD_LAYOUT_NTILE48_ROWPACK2;
      int flags = NE_ATTN_FLAG_NONE;
      ret_ok &= test_case<float, bf16, bf16, float>({1, 1, 1, 32, 128, 64}, flags, false, BA48b2a, 1e-3f);
      ret_ok &= test_case<float, bf16, bf16, float>({2, 5, 5, 32, 64, 128}, flags, false, BA48b2a, 1e-3f);
      ret_ok &= test_case<float, bf16, bf16, float>({2, 5, 5, 80, 128, 77}, flags, false, BA48b2a, 1e-3f);
      ret_ok &= test_case<float, bf16, bf16, float>({1, 1, 1, 256, 63, 63}, flags, false, BA48b2a, 1e-3f);
      ret_ok &= test_case<float, bf16, bf16, float>({3, 4, 4, 256, 1, 384}, flags, false, BA48b2a, 1e-3f);

      flags |= NE_ATTN_FLAG_IS_CAUSAL;
      ret_ok &= test_case<float, bf16, bf16, float>({1, 1, 1, 64, 64, 64}, flags, false, BA48b2a, 1e-3f);
    }

    if (CheckISA(AVX512F)) {  // PREFER_FP32
      const auto BA48b2a = ATTN_FWD_LAYOUT_NTILE48_ROWPACK2;
      int flags = NE_ATTN_FLAG_PREFER_FP32;
      ret_ok &= test_case<float, bf16, bf16, float>({1, 1, 1, 32, 128, 64}, flags, false, BA48b2a, 1e-3f);
      ret_ok &= test_case<float, bf16, bf16, float>({2, 5, 5, 32, 64, 128}, flags, false, BA48b2a, 1e-3f);
      ret_ok &= test_case<float, bf16, bf16, float>({2, 5, 5, 80, 128, 77}, flags, false, BA48b2a, 1e-3f);
      ret_ok &= test_case<float, bf16, bf16, float>({1, 1, 1, 256, 63, 63}, flags, false, BA48b2a, 1e-3f);
      ret_ok &= test_case<float, bf16, bf16, float>({3, 4, 4, 256, 1, 384}, flags, false, BA48b2a, 1e-3f);

      flags |= NE_ATTN_FLAG_IS_CAUSAL;
      ret_ok &= test_case<float, bf16, bf16, float>({1, 1, 1, 64, 64, 64}, flags, false, BA48b2a, 1e-3f);
    }
    if (CheckISA(AVX2)) {  // avx2
      const auto Ba24b = ATTN_FWD_LAYOUT_NTILE24_ROWPACK1;
      int flags = NE_ATTN_FLAG_PREFER_FP32;
      ret_ok &= test_case<float, fp16, fp16, float>({1, 1, 1, 32, 128, 64}, flags, false, Ba24b, 1e-3f);
      ret_ok &= test_case<float, fp16, fp16, float>({2, 5, 5, 32, 64, 128}, flags, false, Ba24b, 1e-3f);
      ret_ok &= test_case<float, fp16, fp16, float>({2, 5, 5, 80, 128, 77}, flags, false, Ba24b, 1e-3f);
      ret_ok &= test_case<float, fp16, fp16, float>({1, 1, 1, 256, 63, 63}, flags, false, Ba24b, 1e-3f);
      ret_ok &= test_case<float, fp16, fp16, float>({3, 4, 4, 256, 1, 384}, flags, false, Ba24b, 1e-3f);

      flags |= NE_ATTN_FLAG_IS_CAUSAL;
      ret_ok &= test_case<float, fp16, fp16, float>({1, 1, 1, 64, 64, 64}, flags, false, Ba24b, 1e-3f);
    }

    {  // amxbf16 => avx2 fallback
      int flags = NE_ATTN_FLAG_NONE;
      ret_ok &= test_reorder_pipe<float, float, float, float>({1, 1, 1, 32, 128, 64}, 64, flags);
      ret_ok &= test_reorder_pipe<float, float, float, float>({2, 5, 5, 32, 64, 128}, 256, flags);
      ret_ok &= test_reorder_pipe<float, float, float, float>({2, 5, 5, 80, 128, 77}, 256, flags);
      ret_ok &= test_reorder_pipe<float, float, float, float>({2, 5, 1, 80, 128, 77}, 256, flags);
      ret_ok &= test_reorder_pipe<float, float, float, float>({1, 1, 1, 256, 63, 63}, 256, flags);
      ret_ok &= test_reorder_pipe<float, float, float, float>({3, 4, 4, 256, 1, 384}, 384, flags);
      ret_ok &= test_reorder_pipe<float, float, float, float>({3, 4, 2, 256, 1, 384}, 384, flags);
      flags |= NE_ATTN_FLAG_IS_CAUSAL;
      ret_ok &= test_reorder_pipe<float, float, float, float>({1, 1, 1, 64, 64, 64}, 128, flags);
      flags |= NE_ATTN_FLAG_IS_ALIBI8;
      ret_ok &= test_reorder_pipe<float, float, float, float>({1, 8, 8, 64, 64, 64}, 128, flags);
    }
    printf("Test suit done: %s\n", __FUNCTION__);
  }

  template <class T>
  static constexpr float init_min_val = std::is_same<T, int8_t>::value    ? -127.f
                                        : std::is_same<T, uint8_t>::value ? 0.f
                                                                          : -1.f;
  template <class T>
  static constexpr float init_max_val = std::is_same<T, int8_t>::value    ? 127.f
                                        : std::is_same<T, uint8_t>::value ? 255.f
                                                                          : 1.f;
  template <class T>
  static constexpr float init_scale_val = 1.f / init_max_val<T>;

#ifdef _MSC_VER
#define __PRETTY_FUNCTION__ __FUNCSIG__
#endif

  template <class Q_T, class K_T, class V_T, class DST_T>
  bool test_case(const attn_shape_t& s, ne_attn_flags_t flags, bool k_trans = false,
                 ATTN_FWD_LAYOUT kv_layout = ATTN_FWD_LAYOUT_PLAIN, float eps = 1e-2f) {
    assert(kv_layout == ATTN_FWD_LAYOUT_PLAIN || !k_trans);
    const auto batch_size = s.batch_size;
    const auto head_num = s.head_num;
    const auto heads_kv = s.heads_kv;
    const auto head_size = s.head_size;
    const auto sl_q = s.sl_q;
    const auto sl_kv = s.sl_kv;
    assert(("GQA not supported!", s.head_num == s.heads_kv));

    const auto is_causal = flags & NE_ATTN_FLAG_IS_CAUSAL ? "maksed" : "unmask";
    const auto is_alibi8 = flags & NE_ATTN_FLAG_IS_ALIBI8 ? "alibi8" : "";
    const auto prefer_fp32 = flags & NE_ATTN_FLAG_PREFER_FP32 ? "FP32" : "";
    printf("\ntest_case: %s\t", __PRETTY_FUNCTION__);
    printf("bs_%d hn_%d hkv_%d hs_%d sl_q_%d sk_kv_%d %s %s %s\n", batch_size, head_num, heads_kv, head_size, sl_q,
           sl_kv, is_causal, is_alibi8, prefer_fp32);

    const auto NTILE = kv_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK4   ? 48
                       : kv_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2 ? 48
                       : kv_layout == ATTN_FWD_LAYOUT_NTILE24_ROWPACK1 ? 24
                                                                       : 0;
    const auto ROWPACK = kv_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK4   ? 4
                         : kv_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2 ? 2
                         : kv_layout == ATTN_FWD_LAYOUT_NTILE24_ROWPACK1 ? 1
                                                                         : 0;
    const auto ROWPAD = ROWPACK > 1 ? ROWPACK * 16 : 1;
    const auto k_rows_pad = kv_layout != ATTN_FWD_LAYOUT_PLAIN ? padto(head_size, ROWPAD) : head_size;
    const auto k_cols_pad = kv_layout != ATTN_FWD_LAYOUT_PLAIN ? padto(sl_kv, NTILE) : sl_kv;
    const auto v_rows_pad = kv_layout != ATTN_FWD_LAYOUT_PLAIN ? padto(sl_kv, ROWPAD) : sl_kv;
    const auto v_cols_pad = kv_layout != ATTN_FWD_LAYOUT_PLAIN ? padto(head_size, NTILE) : head_size;

    std::vector<Q_T> src_q(batch_size * head_num * sl_q * head_size);
    std::vector<K_T> src_k(batch_size * heads_kv * k_rows_pad * k_cols_pad);
    std::vector<V_T> src_v(batch_size * heads_kv * v_rows_pad * v_cols_pad);
    std::vector<DST_T> dst(batch_size * head_num * sl_q * head_size);
    std::vector<DST_T> ref(batch_size * head_num * sl_q * head_size);  // reference result
    std::vector<char> tmp(bestla_fusion_attn_workspace_size(&s));

    // init vector
    static std::mt19937 rng(1);
    std::uniform_int_distribution<> dist;
    init_vector(&src_q, init_min_val<Q_T>, init_max_val<Q_T>, dist(rng));
    init_vector(&src_k, init_min_val<K_T>, init_max_val<K_T>, dist(rng));
    init_vector(&src_v, init_min_val<V_T>, init_max_val<V_T>, dist(rng));

    // pad0 for padded layouts
    if (kv_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK4 || kv_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2 ||
        kv_layout == ATTN_FWD_LAYOUT_NTILE24_ROWPACK1) {
#pragma omp parallel for collapse(2)
      for (int ibs = 0; ibs < batch_size; ++ibs) {
        for (int ihn = 0; ihn < heads_kv; ++ihn) {
          // K
          const auto k_off = (ibs * heads_kv + ihn) * k_rows_pad * k_cols_pad;
          for (int i = 0; i < k_rows_pad; ++i) {
            for (int j = 0; j < k_cols_pad; ++j) {
              if (i < head_size && j < sl_kv) continue;

              const auto j_remain = j % NTILE;
              const auto j_block = j - j_remain;
              const auto i_remain = i % ROWPACK;
              const auto i_block = i - i_remain;
              src_k[k_off + j_block * k_rows_pad + i_block * NTILE + j_remain * ROWPACK + i_remain] = K_T(0);
            }
          }
          // V
          const auto v_off = (ibs * heads_kv + ihn) * v_rows_pad * v_cols_pad;
          for (int i = 0; i < v_rows_pad; ++i) {
            for (int j = 0; j < v_cols_pad; ++j) {
              if (i < sl_kv && j < head_size) continue;

              const auto j_remain = j % NTILE;
              const auto j_block = j - j_remain;
              const auto i_remain = i % ROWPACK;
              const auto i_block = i - i_remain;
              src_v[v_off + j_block * v_rows_pad + i_block * NTILE + j_remain * ROWPACK + i_remain] = V_T(0);
            }
          }
        }
      }
    }

    attn_fwd_args_t<Q_T, K_T, V_T, DST_T> args{
        /* .Q = */ src_q.data(),
        /* .K = */ src_k.data(),
        /* .V = */ src_v.data(),
        /* .dst = */ ref.data(),
        /* .Q_sc = */ init_scale_val<Q_T>,
        /* .K_sc = */ init_scale_val<K_T>,
        /* .V_sc = */ init_scale_val<V_T>,
        /* .dst_sc = */ init_scale_val<V_T>,
        /* .tmp = */ tmp.data(),
        /* .QK_scale = */ 1.f / sqrtf(static_cast<float>(head_size)),
        /* .attn_flags = */ flags,
        /* .batch_size = */ batch_size,
        /* .head_num = */ head_num,
        /* .heads_kv = */ heads_kv,
        /* .head_size = */ head_size,
        /* .sl_q = */ sl_q,
        /* .sl_kv = */ sl_kv,
        /* .Q_layout = */ ATTN_FWD_LAYOUT_PLAIN,
        /* .K_layout = */ kv_layout,
        /* .V_layout = */ kv_layout,
        /* .dst_layout = */ ATTN_FWD_LAYOUT_PLAIN,
        /* .step_q_bs = */ sl_q * head_num * head_size,
        /* .step_q_head_num = */ head_size,
        /* .step_q_sl = */ head_num * head_size,
        /* .step_k_bs = */ sl_kv * heads_kv * head_size,
        /* .step_k_head_num = */ k_trans ? head_size * sl_kv : head_size,
        /* .step_k_sl = */ k_trans ? 1 : heads_kv * head_size,
        /* .step_k_head_size = */ k_trans ? sl_kv : 1,
        /* .step_v_bs = */ sl_kv * heads_kv * head_size,
        /* .step_v_head_num = */ head_size,
        /* .step_v_sl = */ heads_kv * head_size,
        /* .step_v_head_size = */ 1,
        /* .step_dst_bs = */ sl_q * head_num * head_size,
        /* .step_dst_head_num = */ head_size,
        /* .step_dst_sl = */ head_num * head_size,
    };
    if (kv_layout != ATTN_FWD_LAYOUT_PLAIN) {
      args.step_k_bs = heads_kv * k_rows_pad * k_cols_pad;
      args.step_k_head_num = k_rows_pad * k_cols_pad;
      args.step_k_sl = k_rows_pad;
      args.step_k_head_size = NTILE;
      args.step_v_bs = heads_kv * v_rows_pad * v_cols_pad;
      args.step_v_head_num = v_rows_pad * v_cols_pad;
      args.step_v_sl = NTILE;
      args.step_v_head_size = v_rows_pad;
    }

    bestla_fusion_attn_forward_ref(args);

    args.dst = dst.data();
    bestla_fusion_attn_forward(args);

    // Check result
    return compare_data(dst.data(), ref.data(), dst.size(), eps);
  }

  template <class Q_T, class K_T, class V_T, class DST_T>
  bool test_reorder_pipe(const attn_shape_t& s, int sl_kv_max, ne_attn_flags_t flags) {
    const auto batch_size = s.batch_size;
    const auto head_num = s.head_num;
    const auto heads_kv = s.heads_kv;
    const auto head_size = s.head_size;
    const auto sl_q = s.sl_q;
    const auto sl_kv = s.sl_kv;
    assert(("head_num must be a multiple of heads_kv!", head_num % heads_kv == 0));

    const auto is_causal = flags & NE_ATTN_FLAG_IS_CAUSAL ? "maksed" : "unmask";
    const auto is_alibi8 = flags & NE_ATTN_FLAG_IS_ALIBI8 ? "alibi8" : "";
    const auto prefer_fp32 = flags & NE_ATTN_FLAG_PREFER_FP32 ? "FP32" : "";
    printf("\ntest_case: %s\t", __PRETTY_FUNCTION__);
    printf("bs_%d hn_%d hkv_%d hs_%d sl_q_%d sk_kv_%d %s %s %s\n", batch_size, head_num, heads_kv, head_size, sl_q,
           sl_kv, is_causal, is_alibi8, prefer_fp32);

    assert(sl_kv_max >= sl_kv);

    kv_shape_t kv_shape = {
        /* .heads_kv */ static_cast<uint32_t>(heads_kv),
        /* .head_size */ static_cast<uint32_t>(head_size),
        /* .sl_kv_max */ static_cast<uint32_t>(sl_kv_max),
    };
    kv_cache_info_t kv_cache_info;
    bestla_reordered_attn_fp32_batch_kv_info(&kv_shape, &kv_cache_info);
    assert(kv_cache_info.k_layout >= kv_cache_info.v_layout);
    const auto kv_layout = kv_cache_info.k_layout;
    const auto NTILE = kv_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK4   ? 48
                       : kv_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2 ? 48
                       : kv_layout == ATTN_FWD_LAYOUT_NTILE24_ROWPACK1 ? 24
                                                                       : 0;
    const auto ROWPACK = kv_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK4   ? 4
                         : kv_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2 ? 2
                         : kv_layout == ATTN_FWD_LAYOUT_NTILE24_ROWPACK1 ? 1
                                                                         : 0;
    const auto ROWPAD = ROWPACK > 1 ? ROWPACK * 16 : 1;
    const auto k_rows_pad = kv_layout != ATTN_FWD_LAYOUT_PLAIN ? padto(head_size, ROWPAD) : head_size;
    const auto k_cols_pad = kv_layout != ATTN_FWD_LAYOUT_PLAIN ? padto(sl_kv, NTILE) : sl_kv;
    const auto v_rows_pad = kv_layout != ATTN_FWD_LAYOUT_PLAIN ? padto(sl_kv, ROWPAD) : sl_kv;
    const auto v_cols_pad = kv_layout != ATTN_FWD_LAYOUT_PLAIN ? padto(head_size, NTILE) : head_size;

    std::vector<Q_T> src_q(batch_size * head_num * sl_q * head_size);
    std::vector<K_T> src_k(batch_size * heads_kv * sl_kv * head_size);
    std::vector<V_T> src_v(batch_size * heads_kv * sl_kv * head_size);
    std::vector<char> k_cache(batch_size * kv_cache_info.k_bytes);
    std::vector<char> v_cache(batch_size * kv_cache_info.v_bytes);
    std::vector<DST_T> dst(batch_size * head_num * sl_q * head_size);
    std::vector<DST_T> ref(batch_size * head_num * sl_q * head_size);  // reference result
    std::vector<char> tmp(bestla_fusion_attn_workspace_size(&s));

    // init vector
    static std::mt19937 rng(1);
    std::uniform_int_distribution<> dist;
    init_vector(&src_q, init_min_val<Q_T>, init_max_val<Q_T>, dist(rng));
    init_vector(&src_k, init_min_val<K_T>, init_max_val<K_T>, dist(rng));
    init_vector(&src_v, init_min_val<V_T>, init_max_val<V_T>, dist(rng));

    // undefined values
    init_vector(&k_cache, INT8_MIN, INT8_MAX, dist(rng));
    init_vector(&v_cache, INT8_MIN, INT8_MAX, dist(rng));

    int step_src_k_bs = sl_kv * heads_kv * head_size;
    int step_src_k_head_num = head_size;
    int step_src_k_sl = heads_kv * head_size;
    int step_src_k_head_size = 1;
    int step_src_v_bs = sl_kv * heads_kv * head_size;
    int step_src_v_head_num = head_size;
    int step_src_v_sl = heads_kv * head_size;
    int step_src_v_head_size = 1;
    attn_fwd_args_t<Q_T, K_T, V_T, DST_T> ref_args{
        /* .Q = */ src_q.data(),
        /* .K = */ src_k.data(),
        /* .V = */ src_v.data(),
        /* .dst = */ ref.data(),
        /* .Q_sc = */ init_scale_val<Q_T>,
        /* .K_sc = */ init_scale_val<K_T>,
        /* .V_sc = */ init_scale_val<V_T>,
        /* .dst_sc = */ init_scale_val<V_T>,
        /* .tmp = */ tmp.data(),
        /* .QK_scale = */ 1.f / sqrtf(static_cast<float>(head_size)),
        /* .attn_flags = */ flags,
        /* .batch_size = */ batch_size,
        /* .head_num = */ head_num,
        /* .heads_kv = */ heads_kv,
        /* .head_size = */ head_size,
        /* .sl_q = */ sl_q,
        /* .sl_kv = */ sl_kv,
        /* .Q_layout = */ ATTN_FWD_LAYOUT_PLAIN,
        /* .K_layout = */ ATTN_FWD_LAYOUT_PLAIN,
        /* .V_layout = */ ATTN_FWD_LAYOUT_PLAIN,
        /* .dst_layout = */ ATTN_FWD_LAYOUT_PLAIN,
        /* .step_q_bs = */ sl_q * head_num * head_size,
        /* .step_q_head_num = */ head_size,
        /* .step_q_sl = */ head_num * head_size,

        /* .step_k_bs = */ step_src_k_bs,
        /* .step_k_head_num = */ step_src_k_head_num,
        /* .step_k_sl = */ step_src_k_sl,
        /* .step_k_head_size = */ step_src_k_head_size,
        /* .step_v_bs = */ step_src_v_bs,
        /* .step_v_head_num = */ step_src_v_head_num,
        /* .step_v_sl = */ step_src_v_sl,
        /* .step_v_head_size = */ step_src_v_head_size,

        /* .step_dst_bs = */ sl_q * head_num * head_size,
        /* .step_dst_head_num = */ head_size,
        /* .step_dst_sl = */ head_num * head_size,
    };
    bestla_fusion_attn_forward_ref(ref_args);

    if (std::is_same<std::tuple<Q_T, K_T, V_T, DST_T>, std::tuple<float, float, float, float>>::value) {
      assert(kv_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2 || kv_layout == ATTN_FWD_LAYOUT_NTILE24_ROWPACK1);
      // for testing, first reorder sl_kv - 1 and than concat the last 1 line
      const auto seq_size_first = sl_kv - 1;
      const auto seq_size_next = 1;
      bestla_fusion_attn_fp32_update_kv_args_t update_k_args = {
          /* .src = */ src_k.data(),
          /* .cache = */ k_cache.data(),
          /* .batch_size = */ batch_size,
          /* .heads_kv = */ heads_kv,
          /* .head_size = */ head_size,
          /* .seq_off = */ 0,
          /* .seq_size = */ seq_size_first,
          /* .seq_max = */ sl_kv_max,
          /* .step_bs = */ step_src_k_bs,
          /* .step_head_num = */ step_src_k_head_num,
          /* .step_seq = */ step_src_k_sl,
          /* .step_head_size = */ step_src_k_head_size,
      };
      bestla_reordered_attn_fp32_update_k(&update_k_args);

      bestla_fusion_attn_fp32_update_kv_args_t update_v_args = {
          /* .src = */ src_v.data(),
          /* .cache = */ v_cache.data(),
          /* .batch_size = */ batch_size,
          /* .heads_kv = */ heads_kv,
          /* .head_size = */ head_size,
          /* .seq_off = */ 0,
          /* .seq_size = */ seq_size_first,
          /* .seq_max = */ sl_kv_max,
          /* .step_bs = */ step_src_v_bs,
          /* .step_head_num = */ step_src_v_head_num,
          /* .step_seq = */ step_src_v_sl,
          /* .step_head_size = */ step_src_v_head_size,
      };
      bestla_reordered_attn_fp32_update_v(&update_v_args);

      update_k_args.seq_off = seq_size_first;
      update_k_args.seq_size = seq_size_next;
      update_k_args.src = src_k.data() + seq_size_first * step_src_k_sl;
      bestla_reordered_attn_fp32_update_k(&update_k_args);

      update_v_args.seq_off = seq_size_first;
      update_v_args.seq_size = seq_size_next;
      update_v_args.src = src_v.data() + seq_size_first * step_src_v_sl;
      bestla_reordered_attn_fp32_update_v(&update_v_args);

      bestla_reordered_attn_fp32_fp32_fwd_args_t kern_args{
          /* .Q = */ reinterpret_cast<float*>(src_q.data()),
          /* .K = */ k_cache.data(),
          /* .V = */ v_cache.data(),
          /* .dst = */ reinterpret_cast<float*>(dst.data()),
          /* .Q_sc = */ init_scale_val<Q_T>,
          /* .K_sc = */ init_scale_val<K_T>,
          /* .V_sc = */ init_scale_val<V_T>,
          /* .dst_sc = */ init_scale_val<V_T>,
          /* .tmp = */ tmp.data(),
          /* .QK_scale = */ 1.f / sqrtf(static_cast<float>(head_size)),
          /* .attn_flags = */ flags,
          /* .batch_size = */ batch_size,
          /* .head_num = */ head_num,
          /* .heads_kv = */ heads_kv,
          /* .head_size = */ head_size,
          /* .sl_q = */ sl_q,
          /* .sl_kv = */ sl_kv,
          /* .Q_layout = */ ATTN_FWD_LAYOUT_PLAIN,
          /* .K_layout = */ kv_layout,
          /* .V_layout = */ kv_layout,
          /* .dst_layout = */ ATTN_FWD_LAYOUT_PLAIN,
          /* .step_q_bs = */ sl_q * head_num * head_size,
          /* .step_q_head_num = */ head_size,
          /* .step_q_sl = */ head_num * head_size,

          /* .stride_k_bs = */ static_cast<int>(kv_cache_info.k_bytes),
          /* .stride_k_head_num = */ kv_cache_info.stride_k_head_num,
          /* .stride_k_sl = */ kv_cache_info.stride_k_sl,
          /* .stride_k_head_size = */ kv_cache_info.stride_k_head_size,
          /* .stride_v_bs = */ static_cast<int>(kv_cache_info.v_bytes),
          /* .stride_v_head_num = */ kv_cache_info.stride_v_head_num,
          /* .stride_v_sl = */ kv_cache_info.stride_v_sl,
          /* .stride_v_head_size = */ kv_cache_info.stride_v_head_size,

          /* .step_dst_bs = */ sl_q * head_num * head_size,
          /* .step_dst_head_num = */ head_size,
          /* .step_dst_sl = */ head_num * head_size,
      };
      bestla_reordered_attn_fp32_forward(&kern_args);
    }

    // Check result
    return compare_data(dst.data(), ref.data(), dst.size(), 1e-2f);
  }
};
static const TestMhaDese inst_;

}  // namespace

int main() {
  printf("NS_TESTS: mha_dense ");
  printf(ret_ok ? "OK\n" : "FAILED\n");
  return ret_ok ? 0 : -1;
}
#endif
