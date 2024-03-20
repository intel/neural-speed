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
  // use avx2 and f16c on avx2 platforms
  // todo: check avx2 mha on sever
  return !_cd->AVX512F() && _cd->AVX2();
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
