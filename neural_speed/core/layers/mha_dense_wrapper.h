//  Copyright (c) 2024 Intel Corporation
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
#ifndef NE_CORE_GRAPH_MHA_DENSE_WRAPPER_H
#define NE_CORE_GRAPH_MHA_DENSE_WRAPPER_H

#include <immintrin.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <random>
#include <vector>

#include "bestla/bestla.h"
#include "bestla/bestla_epilogue.h"
#include "bestla/bestla_gemm.h"
#include "bestla/bestla_parallel.h"
#include "bestla/bestla_prologue_a.h"
#include "bestla/bestla_prologue_b.h"
#include "bestla/bestla_storage.h"
#include "bestla/bestla_wrapper.h"
#include "core/data_types.h"
#include "layers/bestla_common.hpp"

#ifdef NS_TESTS
#include <memory>
#include <tuple>

#include "layers/ne_test_layers_utils.hpp"
#endif

#define MHA_2ND_EXP 1
constexpr bool MHA_PREFER_AVX512FP16 = true;

#if defined(__GNUC__) && !defined(__clang_major__)  // clang cannot understand target("t1", "t2", "t3" ...)
#define ADD_TARGET(T, ...) __VA_ARGS__, T
#define TARGETS_512_0() "avx512f", "avx512bw", "avx512vl"
#if CompileBF16()
#define TARGETS_512_1() ADD_TARGET("avx512bf16", TARGETS_512_0())
#else
#define TARGETS_512_1() TARGETS_512_0()
#endif
#if CompileFP16()
#define TARGETS_512_2() ADD_TARGET("avx512fp16", TARGETS_512_1())
#else
#define TARGETS_512_2() TARGETS_512_1()
#endif
#define TARGET_512 __attribute__((target(TARGETS_512_2())))
#else
#define TARGET_512
#endif

namespace ne_bestla {
namespace custom {
namespace mha {
using namespace bestla;     // NOLINT
using namespace ne_bestla;  // NOLINT
using bestla::utils::bf16;
using bestla::utils::fp16;
using bestla::utils::padto;
using bestla::utils::padto_le;
using bestla::utils::remainsize;
using bestla::utils::updiv;
namespace utils = bestla::utils;

template <typename Q_T, typename K_T, typename V_T, typename DST_T>
struct attn_fwd_args_t {
  Q_T* Q;
  K_T* K;
  V_T* V;
  DST_T* dst;
  float Q_sc, K_sc, V_sc, dst_sc;
  char* tmp;
  float QK_scale;
  ne_attn_flags_t attn_flags;
  int batch_size, head_num, heads_kv, head_size, sl_q, sl_kv;
  ATTN_FWD_LAYOUT Q_layout, K_layout, V_layout, dst_layout;
  int step_q_bs, step_q_head_num, step_q_sl;
  int step_k_bs, step_k_head_num, step_k_sl, step_k_head_size;
  int step_v_bs, step_v_head_num, step_v_sl, step_v_head_size;
  int step_dst_bs, step_dst_head_num, step_dst_sl;
};

struct mha_problem_t {
  int batch_size, head_num, heads_kv, head_size, sl_q, sl_kv;
};

inline float mha_exp_ref(float x) {
#if MHA_2ND_EXP
  return kernel::ref::exp_ps_0_1(x);
#else
  return expf(x);
#endif
}

#ifdef NOT_CURRENTLY_USED
TARGET_512 inline __m512 exp_2nd_ph(const __m512 z, const __m512 f, const __m512 c0, const __m512 c1, const __m512 c2) {
  const auto y = _mm512_fmadd_ph(_mm512_fmadd_ph(f, c0, c1), f, c2);  // auto y = (f * c0 + c1) * f + c2;
  const auto exp = _mm512_scalef_ph(y, z);
  return exp;
}

TARGET_512 inline __m512 exp_ph_0_1(const __m512 x) {
  static const auto c0 = _mm512_castsi512_ph(_mm512_set1_epi16(fp16(0.240226507f).x));
  static const auto c1 = _mm512_castsi512_ph(_mm512_set1_epi16(fp16(0.452920674f).x));
  static const auto c2 = _mm512_castsi512_ph(_mm512_set1_epi16(fp16(0.713483036f).x));
  static const float v_log2e = std::log2(std::exp(1.f));
  static const auto log2e = _mm512_castsi512_ph(_mm512_set1_epi16(fp16(v_log2e).x));
  static const auto half = _mm512_castsi512_ph(_mm512_set1_epi16(fp16(.5f).x));

  const auto x1 = _mm512_fmadd_ph(x, log2e, half);  // auto x1 = x * log2e + _mm512_set1_ph(.5f);
  const auto z = _mm512_floor_ph(x1);
  const auto f = _mm512_sub_ph(x1, z);  // auto f = x1 - z;

  return exp_2nd_ph(z, f, c0, c1, c2);
}
#endif

alignas(32) const uint32_t mask8[9][8]{
    {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000},
    {0xffffffff, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000},
    {0xffffffff, 0xffffffff, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000},
    {0xffffffff, 0xffffffff, 0xffffffff, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000},
    {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x00000000, 0x00000000, 0x00000000, 0x00000000},
    {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x00000000, 0x00000000, 0x00000000},
    {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x00000000, 0x00000000},
    {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x00000000},
    {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff},
};

/**
 * @brief An Epilogue that optionally apply a casual mask and scale the fp32 result, performing exp, accumulating sum of
 * each line of exp, and storing exp as bf16 results
 */
template <BTLA_ISA ISA_T, typename T_DST>
class scale_exp_acc_sum_fp32_t {
 public:
  struct Param {  // NOLINT(readability-identifier-naming): align with bestla name
    T_DST* dst;
    float* dst_sum;
    int ld_dst;  // #elements
    float scale;
    int causal_offset;  // offset for causal mask; negative value for disabling causal mask
    float alibi_slope;  // m-factor in the alibi paper for current head: https://arxiv.org/abs/2108.12409
  };

  TARGET_512 BTLA_CODE forward(const float* src, const int src_step, const int M_offset, const int N_offset,
                               const int M, const int N, const Param& p, void* /* tmpcache */,
                               size_t /* cachesize */) const {
    assert(("alibi not supported!", p.alibi_slope == 0.f));
    const auto dst = p.dst + M_offset * p.ld_dst + N_offset;
    const auto dst_sum = p.dst_sum + M_offset;
#if MHA_2ND_EXP && CompileBF16()
    static_assert(std::is_same<T_DST, bf16>::value, "bf16 support only");
    const auto v_scale = _mm512_set1_ps(p.scale);
    for (int i = 0; i < M; ++i) {
      const auto N_unmasked =
          std::min(N, p.causal_offset < 0 ? INT32_MAX : i + M_offset - N_offset + p.causal_offset + 1);

      const auto v_mask = _cvtu32_mask16((1U << (N_unmasked % 16)) - 1);
      int j = 0;
      auto v_sum = _mm512_setzero_ps();
      for (; j < N_unmasked - 15; j += 16) {
        const auto v_exp = kernel::avx512f::exp_ps_0_1(_mm512_mul_ps(v_scale, _mm512_loadu_ps(src + i * src_step + j)));
        v_sum = _mm512_add_ps(v_sum, v_exp);
        _mm256_storeu_epi16(dst + i * p.ld_dst + j, (__m256i)_mm512_cvtneps_pbh(v_exp));
      }
      if (j < N_unmasked) {
        const auto v_exp =
            kernel::avx512f::exp_ps_0_1(_mm512_mul_ps(v_scale, _mm512_maskz_loadu_ps(v_mask, src + i * src_step + j)));
        v_sum = _mm512_mask_add_ps(v_sum, v_mask, v_sum, v_exp);
        _mm256_storeu_epi16(dst + i * p.ld_dst + j, (__m256i)_mm512_maskz_cvtneps_pbh(v_mask, v_exp));
        j += 16;
      }
      dst_sum[i] += _mm512_reduce_add_ps(v_sum);

      if (j < utils::padto(N, 64)) memset(dst + i * p.ld_dst + j, 0, sizeof(*dst) * (utils::padto(N, 64) - j));
    }
#else
    for (int i = 0; i < M; ++i) {
      const auto N_unmasked =
          std::min(N, p.causal_offset < 0 ? INT32_MAX : i + M_offset - N_offset + p.causal_offset + 1);
      for (int j = 0; j < N_unmasked; ++j) {
        const auto exp_ = expf(src[i * src_step + j] * p.scale);
        dst[i * p.ld_dst + j] = static_cast<T_DST>(exp_);
        dst_sum[i] += exp_;
      }
      if (N_unmasked < utils::padto(N, 64))
        memset(dst + i * p.ld_dst + N_unmasked, 0, sizeof(*dst) * (utils::padto(N, 64) - N_unmasked));
    }
#endif

    return BTLA_CODE::Success;
  }
};
template <BTLA_ISA ISA_T>
using ScaleExpAccSumFp32Bf16 = scale_exp_acc_sum_fp32_t<ISA_T, bf16>;

/**
 * @brief An Epilogue that scale the fp32 result, convert to bf16 and write back to memory
 */
template <BTLA_ISA ISA_T, typename T_SRC, typename T_DST>
class scale_write_back_t {
 public:
  using SType = T_SRC;
  using DType = T_DST;
  struct Param {  // NOLINT(readability-identifier-naming): align with bestla name
    const float* scale;
    DType* dst;
    int ld_dst;
  };

  BTLA_CODE forward(const SType* src, const int src_step, const int M_offset, const int N_offset, const int M,
                    const int N, const Param& p, void* /* tmpcache */, size_t /* cachesize */) {
    const auto dst = p.dst + M_offset * p.ld_dst + N_offset;
    const auto scale = p.scale + M_offset;
    // TODO(Yi): high performance implementation
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < N; ++j)  //
        dst[i * p.ld_dst + j] = static_cast<DType>(scale[i] * src[i * src_step + j]);

    return BTLA_CODE::Success;
  }
};
template <BTLA_ISA ISA_T>
using ScaleWriteBackFp32Bf16 = scale_write_back_t<ISA_T, float, bf16>;
template <BTLA_ISA ISA_T>
using ScaleWriteBackFp32Fp32 = scale_write_back_t<ISA_T, float, float>;
template <BTLA_ISA ISA_T>
using ScaleWriteBackS32S8 = scale_write_back_t<ISA_T, int32_t, int8_t>;

/**
 * @brief PackedWeight(Default) with batch
 */
class storage_packed_weight_batch_t : public storage::gemm::IWeightBase {
  using Base = storage::gemm::IWeightBase;

 public:
  int mBatch;
  storage::ObjectAlignedBuffer<NE_ALIGNMENT> mWBuf;
  // size_t mWSize;

  explicit storage_packed_weight_batch_t(uint64_t _core_id) : Base(_core_id), mBatch(0) {}
  size_t resize(int NPad, int KPad, int N, int K, int num_batch, BTLA_DTYPE dtype) {
    IWeightBase::resize(NPad, KPad, N, K, dtype);
    mBatch = num_batch;
    auto bsize = static_cast<size_t>(mBatch) * NPad * KPad * utils::bestla_dtype_size(dtype);
    mWBuf.resize(bsize);
    mSize = utils::padto(IWeightBase::getSerializedSize() + mWBuf.getSerializedSize(), NE_ALIGNMENT);
    return mSize;
  }

  template <typename T>
  inline constexpr T* WPtr() const {
    return mWBuf.get<T>();
  }

  void assign(int8_t* buf) override {
    deserializeBuffer(buf, true);
    mWBuf.deserializeBuffer(buf, true);
  }

  void serialize(int8_t* wptr) override {
    serializeToBuffer(wptr);
    mWBuf.serializeToBuffer(wptr);
  }

  void deserialize(int8_t* rptr) override {
    deserializeBuffer(rptr, false);
    mWBuf.deserializeBuffer(rptr, false);
  }

 protected:
  size_t getSerializedSize() override { return Base::getSerializedSize() + sizeof(mBatch); }

  void serializeToBuffer(int8_t*& wptr) override {
    Base::serializeToBuffer(wptr);
    utils::serialize(wptr, mBatch);
  }
  void deserializeBuffer(int8_t*& rptr, bool map_buf) override {
    Base::deserializeBuffer(rptr, map_buf);
    if (!map_buf) {
      mBatch = utils::deserialize<int>(rptr);
    } else {
      utils::serialize<int>(rptr, mBatch);
    }
  }
};

/**
 * @brief An weight Prologue that Packs transposed Bf16 weight; optimized for runtime packing. It is the base type of
 * that for transposed / non-transposed source
 */
template <class GemmCore_T, BTLA_ISA ISA_T, bool IsTrans, typename T_SRC = typename GemmCore_T::BType>
class weight_pack_batch_bf16_base_t {
 public:
  using WType = typename GemmCore_T::BType;           // weight type
  using SType = T_SRC;                                // source type (before packed)
  using StorageType = storage_packed_weight_batch_t;  // packed weight type

  struct Param {  // NOLINT(readability-identifier-naming): align with bestla name
    const SType* B;
    const int ldb;
    const StorageType* packedW;
  };

  BTLA_CODE getWeight(...) = delete;

  BTLA_CODE getWeight(WType** dstptr, int* dststep, int /* b_size */, int /* k_size */, int /* n_size */, int b_offset,
                      int k_offset, int n_offset, const Param& param, void* /* tmpcache */, size_t /* cachesize */) {
    const auto wptr = param.packedW;
    if (!wptr) return BTLA_CODE::InvalidParam;
    assert(k_offset % GemmCore_T::KTILE == 0);
    assert(n_offset % GemmCore_T::NTILE == 0);
    auto KPad = wptr->mKPad;
    auto NPad = wptr->mNPad;
    *dstptr = wptr->template WPtr<WType>() + n_offset * KPad + k_offset * GemmCore_T::NTILE;
    *dststep = KPad;
    return BTLA_CODE::Success;
  }

  BTLA_CODE getWeight(WType** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                      const Param& param, void* tmpcache, size_t cachesize) {
    return getWeight(dstptr, dststep, 1, k_size, n_size, 0, k_offset, n_offset, param, tmpcache, cachesize);
  }

  BTLA_CODE packWeight(...) = delete;
};

template <class GemmCore_T, BTLA_ISA ISA_T, typename T_SRC = typename GemmCore_T::BType>
class weight_pack_batch_bf16_trans_t : public weight_pack_batch_bf16_base_t<GemmCore_T, ISA_T, true, T_SRC> {
  using Base = weight_pack_batch_bf16_base_t<GemmCore_T, ISA_T, true, T_SRC>;

 public:
  using typename Base::Param;
  using typename Base::StorageType;
  using typename Base::SType;
  using typename Base::WType;

  /// Reorder job of a thread
  void run(const Param& p, const parallel::ThreadProblem2D& thdp, const std::function<int(int)>& step_batch) {
    if (!thdp.valid) return;
    const auto pw = dynamic_cast<const StorageType*>(p.packedW);
    assert(pw != nullptr);
    const int KPad = pw->mKPad;  // K size after transpose & padding
    const int NPad = pw->mNPad;  // N size after transpose & padding
    assert(pw->mK <= KPad);
    assert(pw->mN <= NPad);

    // y for batch; x for major-dim of the source data (N-dim of the packed weight)
    const auto [y, x] = thdp.loc;
    const auto [ny, nx] = thdp.size;
    const auto nx_pad = utils::padto(nx, GemmCore_T::NTILE);

    assert(padto(pw->mK, GemmCore_T::KTILE) == KPad);

    using KernInterleave = typename kernel::wrapper::PaddingTransInterleaveMN<  //
        GemmCore_T::NTILE, GemmCore_T::PACK_ROW>;

    for (int ibat = y; ibat < y + ny; ++ibat) {
      const auto forward_stat = KernInterleave::template forward<ISA_T, T_SRC, WType>(  //
          p.B + step_batch(ibat) + x * p.ldb,                                           //
          pw->template WPtr<WType>() + ibat * KPad * NPad + x * KPad,                   //
          nx, pw->mK,                                                                   // size
          nx_pad, KPad,                                                                 // padded size
          p.ldb, KPad);                                                                 // step
      assert(forward_stat == BTLA_CODE::Success);
    }
  }
};

template <class GemmCore_T, BTLA_ISA ISA_T, typename T_SRC = typename GemmCore_T::BType>
class weight_pack_batch_bf16_non_tr_t : public weight_pack_batch_bf16_base_t<GemmCore_T, ISA_T, false, T_SRC> {
  using Base = weight_pack_batch_bf16_base_t<GemmCore_T, ISA_T, false, T_SRC>;

 public:
  using typename Base::Param;
  using typename Base::StorageType;
  using typename Base::SType;
  using typename Base::WType;

  /// Reorder job of a thread
  void run(const Param& p, const parallel::ThreadProblem2D& thdp, const std::function<int(int)>& step_batch) {
    if (!thdp.valid) return;
    const auto pw = dynamic_cast<const StorageType*>(p.packedW);
    assert(pw != nullptr);
    const int KPad = pw->mKPad;  // K size after padding
    const int NPad = pw->mNPad;  // N size after padding
    assert(pw->mK <= KPad);
    assert(pw->mN <= NPad);
    assert(padto(pw->mN, GemmCore_T::NTILE) == NPad);

    auto [y, x] = thdp.loc;
    auto [ny, nx] = thdp.size;
    const auto nx_pad = utils::padto(nx, GemmCore_T::KTILE);

    using KernInterleave = typename kernel::wrapper::PaddingInterleaveMN<  //
        GemmCore_T::NTILE, GemmCore_T::PACK_ROW>;

    for (int ibat = y; ibat < y + ny; ++ibat) {
      const auto forward_stat = KernInterleave::template forward<ISA_T, T_SRC, WType>(  //
          p.B + step_batch(ibat) + x * p.ldb,                                           //
          pw->template WPtr<WType>() + ibat * KPad * NPad + x * GemmCore_T::NTILE,      //
          nx, pw->mN,                                                                   // size
          nx_pad, NPad,                                                                 // padded size
          p.ldb, KPad);                                                                 // stride
      assert(forward_stat == BTLA_CODE::Success);
    }
  }
};

template <class _GemmCore_T, BTLA_ISA ISA_T>
class activation_identity_t {
 public:
  using AType = typename _GemmCore_T::AType;
  struct Param {  // NOLINT(readability-identifier-naming): align with bestla name
    const AType* A;
    int lda;
  };
  activation_identity_t() = default;

  BTLA_CODE getActivation(AType** dstptr, int* dststep, const Param& _param, int m_size, int k_size, int m_offset,
                          int k_offset, void* /* tmpcache */, size_t /* cachesize */) {
    auto aptr = const_cast<AType*>(_param.A);
    *dstptr = aptr + m_offset * _param.lda + k_offset;
    *dststep = _param.lda;
    return BTLA_CODE::Success;
  }
};

/**
 * @brief LauncherBase with addition input as packed weight offset
 */
template <BTLA_ISA RT_ISA_, class _GemmCore_T, template <class, BTLA_ISA> class _PrologueA_T,
          template <class, BTLA_ISA> class _PrologueB_T, template <BTLA_ISA> class _Epilogue_T>
class launcher_base_off_t                  //
    : public wrapper::gemm::LauncherBase<  //
          RT_ISA_, _GemmCore_T, _PrologueA_T, _PrologueB_T, _Epilogue_T> {
  using Base = wrapper::gemm::LauncherBase<  //
      RT_ISA_, _GemmCore_T, _PrologueA_T, _PrologueB_T, _Epilogue_T>;

 public:
  using typename Base::GemmCore;
  using Param = typename Base::Param;
  using AType = typename Base::AType;
  using BType = typename Base::BType;
  using CType = typename Base::CType;
  static constexpr auto RT_ISA = RT_ISA_;

  void run(const Param& _param, const parallel::gemm::ThreadProblemBase& _config,
           int w_offset /* weight offset for batching */) {
    // TO(Yi) temporarily configure to max tiling size
    this->mGemmCore.configure(16, 16, 16);  // Need 'this->' here; See：https://stackoverflow.com/questions/11405
    auto StackTmp = alloca(_config.stacksize);
    auto tmpB = reinterpret_cast<BType*>(StackTmp);
    tmpB = utils::cpu_pointer_align(tmpB);
    auto tmpA = reinterpret_cast<AType*>(tmpB + static_cast<size_t>(_config.block[1]) * _config.block[2]);
    tmpA = utils::cpu_pointer_align(tmpA);
    auto tmpC = reinterpret_cast<CType*>(tmpA + static_cast<size_t>(GemmCore::MTILE) * _config.block[2]);
    tmpC = utils::cpu_pointer_align(tmpC);
    auto tmpCache = tmpC + _config.block[0] * _config.block[1];
    tmpCache = utils::cpu_pointer_align(tmpCache);

    for (int itern = 0; itern < _config.size[1]; itern += _config.block[1]) {
      int n_remain = utils::remainsize(itern, _config.size[1], _config.block[1]);
      for (int iterm = 0; iterm < _config.size[0]; iterm += _config.block[0]) {
        int m_remain = utils::remainsize(iterm, _config.size[0], _config.block[0]);
        run_block(_param, _config, w_offset, iterm, itern, m_remain, n_remain, tmpA, tmpB, tmpC, tmpCache);
      }
    }
  }

 protected:
  void run_block(const Param& _param, const parallel::gemm::ThreadProblemBase& _config,
                 int w_offset /* weight offset for batching */, int blk_m, int blk_n, int blk_msize, int blk_nsize,
                 AType* tmpA, BType* /*tmpB*/, CType* tmpC, void* tmpcache) {
    int n_padded = padto(blk_nsize, GemmCore::NTILE);
    for (int iterk = 0; iterk < _param.problem.dims[3]; iterk += _config.block[2]) {
      int k_remain = utils::remainsize(iterk, _param.problem.dims[3], _config.block[2]);
      int k_padded = padto(k_remain, GemmCore::KTILE);
      int k_paddedle = padto_le(k_remain, GemmCore::KTILE);
      BType* bptr_cache = nullptr;
      int bcache_step = 0;
      this->mProB.getWeight(&bptr_cache, &bcache_step,      // See：https://stackoverflow.com/questions/11405
                            k_padded, n_padded,             //
                            iterk, _config.loc[1] + blk_n,  //
                            _param.paramB, tmpcache, _config.tmpcachesize);
      bptr_cache += w_offset;
      int bcache_stride = bcache_step * sizeof(BType);
      for (int i = 0; i < blk_msize; i += GemmCore::MTILE) {
        int m_remain = remainsize(i, blk_msize, GemmCore::MTILE);
        auto cptr_cache = tmpC + i * _config.block[1];
        int ccache_stride = _config.block[1] * sizeof(CType);

        int acache_step = 0;
        if (k_paddedle) {
          AType* aptr_cache = tmpA;
          this->mProA.getActivation(&aptr_cache, &acache_step, _param.paramA, m_remain, k_paddedle,
                                    blk_m + i + _config.loc[0], iterk, tmpcache, _config.tmpcachesize);
          this->mGemmCore.forward(aptr_cache, bptr_cache, cptr_cache, m_remain, n_padded, k_paddedle,
                                  acache_step * sizeof(AType), bcache_stride, ccache_stride, iterk, tmpcache,
                                  _config.tmpcachesize);
        }
        int k_tail = k_remain - k_paddedle;
        if (k_tail) {
          AType* aptr_cache = tmpA;
          this->mProA.getActivation(&aptr_cache, &acache_step, _param.paramA, m_remain, k_tail,
                                    blk_m + i + _config.loc[0], iterk + k_paddedle, tmpcache, _config.tmpcachesize);
          this->mGemmCore.forward(aptr_cache, bptr_cache + k_paddedle * GemmCore::NTILE, cptr_cache, m_remain, n_padded,
                                  GemmCore::KTILE, acache_step * sizeof(AType), bcache_stride, ccache_stride,
                                  iterk + k_paddedle, tmpcache, _config.tmpcachesize);
        }
      }
    }
    this->mEpilogue.forward(tmpC, _config.block[1], _config.loc[0] + blk_m, _config.loc[1] + blk_n, blk_msize,
                            blk_nsize, _param.paramC, tmpcache, _config.tmpcachesize);
  }
};

/**
 * @brief LauncherBase with addition input as packed weight offset
 */
template <BTLA_ISA RT_ISA_, class _GemmCore_T, template <class, BTLA_ISA> class _PrologueA_T,
          template <class, BTLA_ISA> class _PrologueB_T, template <BTLA_ISA> class _Epilogue_T>
class launcher_base_weight_t               //
    : public wrapper::gemm::LauncherBase<  //
          RT_ISA_, _GemmCore_T, _PrologueA_T, _PrologueB_T, _Epilogue_T> {
  using Base = wrapper::gemm::LauncherBase<  //
      RT_ISA_, _GemmCore_T, _PrologueA_T, _PrologueB_T, _Epilogue_T>;

 public:
  using typename Base::GemmCore;
  using Param = typename Base::Param;
  using AType = typename Base::AType;
  using BType = typename Base::BType;
  using CType = typename Base::CType;
  static constexpr auto RT_ISA = RT_ISA_;

  void run(const Param& _param, const parallel::gemm::ThreadProblemBase& _config) {
    this->mGemmCore.configure(16, 16, 16);  // Need 'this->' here; See：https://stackoverflow.com/questions/11405
    auto StackTmp = alloca(_config.stacksize);
    auto tmpB = reinterpret_cast<BType*>(StackTmp);
    tmpB = utils::cpu_pointer_align(tmpB);
    auto tmpA = reinterpret_cast<AType*>(tmpB + static_cast<size_t>(_config.block[1]) * _config.block[2]);
    tmpA = utils::cpu_pointer_align(tmpA);
    auto tmpC = reinterpret_cast<CType*>(tmpA + static_cast<size_t>(GemmCore::MTILE) * _config.block[2]);
    tmpC = utils::cpu_pointer_align(tmpC);
    auto tmpCache = tmpC + _config.block[0] * _config.block[1];
    tmpCache = utils::cpu_pointer_align(tmpCache);

    for (int itern = 0; itern < _config.size[1]; itern += _config.block[1]) {
      int n_remain = utils::remainsize(itern, _config.size[1], _config.block[1]);
      for (int iterm = 0; iterm < _config.size[0]; iterm += _config.block[0]) {
        int m_remain = utils::remainsize(iterm, _config.size[0], _config.block[0]);
        run_block(_param, _config, iterm, itern, m_remain, n_remain, tmpA, tmpB, tmpC, tmpCache);
      }
    }
  }

 protected:
  void run_block(const Param& _param, const parallel::gemm::ThreadProblemBase& _config, int blk_m, int blk_n,
                 int blk_msize, int blk_nsize, AType* tmpA, BType* tmpB, CType* tmpC, void* tmpcache) {
    int n_padded = padto(blk_nsize, GemmCore::NTILE);
    for (int iterk = 0; iterk < _param.problem.dims[3]; iterk += _config.block[2]) {
      int k_remain = remainsize(iterk, _param.problem.dims[3], _config.block[2]);
      int k_padded = padto(k_remain, GemmCore::KTILE);
      int k_paddedle = padto_le(k_remain, GemmCore::KTILE);
      auto bptr_cache = tmpB;
      int bcache_step = 0;

      this->mProB.getWeight(&bptr_cache, &bcache_step, _param.paramB, k_padded, blk_nsize, iterk,
                            _config.loc[1] + blk_n, tmpcache, _config.tmpcachesize);
      int bcache_stride = bcache_step * sizeof(BType);
      for (int i = 0; i < blk_msize; i += GemmCore::MTILE) {
        int m_remain = remainsize(i, blk_msize, GemmCore::MTILE);
        auto cptr_cache = tmpC + i * _config.block[1];
        int ccache_stride = _config.block[1] * sizeof(CType);

        int acache_step = 0;
        if (k_paddedle) {
          AType* aptr_cache = tmpA;
          this->mProA.getActivation(&aptr_cache, &acache_step, _param.paramA, m_remain, k_paddedle,
                                    (blk_m + i + _config.loc[0]), iterk, tmpcache, _config.tmpcachesize);
          this->mGemmCore.forward(aptr_cache, bptr_cache, cptr_cache, m_remain, n_padded, k_paddedle,
                                  acache_step * sizeof(AType), bcache_stride, ccache_stride, iterk, tmpcache,
                                  _config.tmpcachesize);
        }
        int k_tail = k_remain - k_paddedle;
        if (k_tail) {
          AType* aptr_cache = tmpA;
          this->mProA.getActivation(&aptr_cache, &acache_step, _param.paramA, m_remain, k_tail,
                                    (blk_m + i + _config.loc[0]), iterk + k_paddedle, tmpcache, _config.tmpcachesize);
          this->mGemmCore.forward(aptr_cache, bptr_cache + k_paddedle * GemmCore::NTILE, cptr_cache, m_remain, n_padded,
                                  GemmCore::KTILE, acache_step * sizeof(AType), bcache_stride, ccache_stride,
                                  iterk + k_paddedle, tmpcache, _config.tmpcachesize);
        }
      }
    }
    this->mEpilogue.forward(tmpC, _config.block[1], (_config.loc[0] + blk_m), _config.loc[1] + blk_n, blk_msize,
                            blk_nsize, _param.paramC, tmpcache, _config.tmpcachesize);
  }
};

/**
 * @brief MHA interface
 *
 * @tparam L_ExpSum Launcher type of the QK exp sum matmul
 * @tparam L_Scale Launcher type of the PV scale matmul (S for that in the flash-attn paper)
 */
template </* class Parallel_T, */ class L_ExpSum, class L_Scale>
class mha_interface_t {
 public:
  using PrologueQ = typename L_ExpSum::PrologueA;
  using PrologueK = typename L_ExpSum::PrologueB;
  using QKProQArgs = typename PrologueQ::Param;
  using QKProKArgs = typename PrologueK::Param;
  using QKArgs = typename L_ExpSum::Param;
  using QKEpiArgs = typename L_ExpSum::EpiParam;

  using PrologueS = typename L_Scale::PrologueA;
  using PrologueV = typename L_Scale::PrologueB;
  using PVProPArgs = typename PrologueS::Param;
  using PVProVArgs = typename PrologueV::Param;
  using PVArgs = typename L_Scale::Param;
  using PVEpiArgs = typename L_Scale::EpiParam;

  using GemmQK = typename L_ExpSum::GemmCore;
  using GemmPV = typename L_Scale::GemmCore;
  using Q_T = typename std::remove_const<typename std::remove_pointer<decltype(QKProQArgs::A)>::type>::type;
  using K_T = typename PrologueK::SType;
  using V_T = typename PrologueV::SType;
  using DST_T = typename std::remove_const<typename std::remove_pointer<decltype(PVEpiArgs::dst)>::type>::type;

  static_assert(GemmQK::MTILE == GemmPV::MTILE, "2 GEMM should have the same M_TILE.");

  BTLA_CODE compute(const attn_fwd_args_t<Q_T, K_T, V_T, DST_T>& p, parallel::IThreading& th) {
    static constexpr auto M_TILE = GemmQK::MTILE;
    assert(p.Q_sc == 1 && p.K_sc == 1 && p.V_sc == 1 && p.dst_sc == 1);
    assert(p.Q_layout == ATTN_FWD_LAYOUT_PLAIN && p.K_layout == ATTN_FWD_LAYOUT_PLAIN &&
           p.V_layout == ATTN_FWD_LAYOUT_PLAIN && p.dst_layout == ATTN_FWD_LAYOUT_PLAIN);
    assert(p.step_v_head_size == 1);
    assert(p.step_k_head_size == 1 || p.step_k_sl == 1);
    const auto num_heads = p.batch_size * p.head_num;  // Total number of heads
    GetCPUDevice();

    const bool is_causal = (p.attn_flags & NE_ATTN_FLAG_IS_CAUSAL) != 0;
    const bool is_alibi = (p.attn_flags & NE_ATTN_FLAG_IS_ALIBI8) != 0;
    const bool prefer_fp32 = (p.attn_flags & NE_ATTN_FLAG_PREFER_FP32) != 0;

    assert(!is_causal || p.sl_q <= p.sl_kv);
    assert(("qlen should be no greater then klen/vlen!", !is_causal || p.sl_q <= p.sl_kv));
    assert(("prefer_fp32 not implemented!", !prefer_fp32));
    assert(("alibi not supported!", !is_alibi));
    assert(("GQA not supported!", p.head_num == p.heads_kv));
    const auto sl_diff = p.sl_kv - p.sl_q;

    // prepare memory for packed weight
    // TODO(Yi): init packed weight with p.tmp
    storage_packed_weight_batch_t /*<typename GemmQK::BType>*/ K_pack(GemmQK::ID);  // packed K
    K_pack.resize(padto(p.sl_kv, GemmQK::NTILE), padto(p.head_size, GemmQK::KTILE), p.sl_kv, p.head_size, num_heads,
                  utils::bestla_dtype<typename GemmQK::BType>);
    auto bufferK = utils::amalloc<int8_t>(K_pack.mSize);
    K_pack.assign(bufferK);
    storage_packed_weight_batch_t /*<typename GemmPV::BType>*/ V_pack(GemmPV::ID);  // packed V
    V_pack.resize(padto(p.head_size, GemmPV::NTILE), padto(p.sl_kv, GemmPV::KTILE), p.head_size, p.sl_kv, num_heads,
                  utils::bestla_dtype<typename GemmPV::BType>);
    auto bufferV = utils::amalloc<int8_t>(V_pack.mSize);
    V_pack.assign(bufferV);
    const auto K_pack_batch_off = K_pack.mKPad * K_pack.mNPad;
    const auto V_pack_batch_off = V_pack.mKPad * V_pack.mNPad;

    const auto step_batch_k = [step_bs = p.step_k_bs, step_hn = p.step_k_head_num, hn = p.heads_kv](int ibat) {
      return (ibat / hn) * step_bs + (ibat % hn) * step_hn;
    };
    const auto step_batch_v = [step_bs = p.step_v_bs, step_hn = p.step_v_head_num, hn = p.heads_kv](int ibat) {
      return (ibat / hn) * step_bs + (ibat % hn) * step_hn;
    };

    // prepare parallel scheduler for packed weight
    using Scheduler2D = bestla::parallel::Scheduler2D;
    using ThreadProblem2D = bestla::parallel::ThreadProblem2D;
    const auto schK = p.step_k_head_size == 1
                          ? Scheduler2D({th.num_threads(), {num_heads, p.sl_kv}, {1, GemmQK::NTILE}})
                          : Scheduler2D({th.num_threads(), {num_heads, p.head_size}, {1, GemmQK::KTILE}});
    const auto schV = Scheduler2D({th.num_threads(), {num_heads, p.sl_kv}, {1, GemmPV::KTILE}});

    const mha_problem_t problem = {p.batch_size, p.head_num, p.heads_kv, p.head_size, p.sl_q, p.sl_kv};
    const auto m_tiles = updiv(p.sl_q, M_TILE);
    const auto num_tasks = num_heads * m_tiles;
    const Scheduler2D parl({th.num_threads(), {num_tasks, 1}, {1, 1}, {0, 0}});

    th.parallel_for([&](int tid) {
      {  // reorder K & V
        ThreadProblem2D thdpK{tid};
        schK.getIndex(thdpK);
        l_expsum.mProB.run(  // pack K
            QKProKArgs{
                /* .B = */ p.K,
                /* .ldb = */ p.step_k_sl * p.step_k_head_size,  //  use the non-one step
                /* .StorageType = */ &K_pack,
            },
            thdpK, step_batch_k);

        ThreadProblem2D thdpV{tid};
        schV.getIndex(thdpV);
        l_scale.mProB.run(  // pack V
            PVProVArgs{
                /* .B = */ p.V,
                /* .ldb = */ p.step_v_sl,
                /* .StorageType = */ &V_pack,
            },
            thdpV, step_batch_v);
      }

      th.sync(tid);

      // calculate mm + softmax + mm
      {
        const int tmp_exp_size = M_TILE * padto(p.sl_kv, GemmQK::NTILE) * sizeof(ne_bf16_t);  // TODO(Yi): alignment?
        const auto tmp = p.tmp + tid * tmp_exp_size;
        ThreadProblem2D thdp{tid};
        parl.getIndex(thdp);
        const auto [task_start, _assert0] = thdp.loc;
        auto [task_size, _assert_max1] = thdp.size;
        assert(task_size == 0 || _assert0 == 0);
        assert(task_size == 0 || _assert_max1 == 1 || _assert_max1 == 0);
        if (_assert_max1 == 0 || !thdp.valid) task_size = 0;

        for (int task_id = task_start; task_id < task_start + task_size; ++task_id) {
          const int ibat = task_id / m_tiles;
          const int i_m = task_id % m_tiles * M_TILE;
          const int ibs = ibat / p.head_num;
          const int ihn = ibat % p.head_num;
          const int m_size = std::min(M_TILE, p.sl_q - i_m);
          // TODO(Yi): heads_kv

          float exp_sum[M_TILE]{};
          memset(exp_sum, 0, sizeof(exp_sum));

          // ptr to Q / dst matrix of the current head
          const auto head_q = p.Q + ibs * p.step_q_bs + ihn * p.step_q_head_num;
          const auto head_dst = p.dst + ibs * p.step_dst_bs + ihn * p.step_dst_head_num;
          const auto unmasked_size = is_causal ? std::min(p.sl_kv, p.sl_kv - p.sl_q + i_m + M_TILE - 1 + 1) : p.sl_kv;

          const auto unmasked_size_pad_qk = std::min(p.sl_kv, padto(unmasked_size, GemmQK::NTILE));
          const auto unmasked_size_pad_pv = std::min(p.sl_kv, padto(unmasked_size, GemmPV::KTILE));
          const auto ld_tmp_exp = padto(padto(unmasked_size_pad_pv, GemmQK::NTILE), GemmPV::KTILE);

          typename parallel::gemm::ThreadProblemBase tpQK{
              /* ThreadProblem2D */ {tid, {}, {i_m, 0}, {m_size, unmasked_size_pad_qk}, true},
              /* .block = */ {M_TILE, GemmQK::NTILE, p.head_size},
              /* .stacksize = */ _cd->getL2CacheSize(),
              /* .tmpcachesize = */ _cd->getL2CacheSize(),
          };
          const auto bf16_tmp = reinterpret_cast<bf16*>(tmp);
          l_expsum.run(  // QxK => S ==exp==> P
              QKArgs{
                  utils::GemmProblem{
                      /* .batch */ 1,
                      /* .M = */ p.sl_q,
                      /* .N = */ unmasked_size_pad_qk,
                      /* .K = */ p.head_size,
                  },
                  /* .paramA = */ QKProQArgs{head_q, p.step_q_sl},
                  /* .paramB = */ QKProKArgs{nullptr, 0, &K_pack},
                  /* .paramC = */
                  QKEpiArgs{
                      /* .dst = */ bf16_tmp - i_m * ld_tmp_exp,  // pretend that there is a whole exp mat
                      /* .dst_sum = */ exp_sum - i_m,            // pretend that there is a whole exp sum
                      /* .ld_dst = */ ld_tmp_exp,
                      /* .scale = */ p.QK_scale,
                      /* .causal_offset = */ is_causal ? sl_diff : -1,
                      /* .alibi_slope = */ 0.f,
                  },
                  // /* .workspace = */ nullptr,
              },
              tpQK, /* w_offset */ ibat * K_pack_batch_off);
          for (int ii = 0; ii < M_TILE; ++ii) exp_sum[ii] = 1.f / exp_sum[ii];

          typename parallel::gemm::ThreadProblemBase tpPV{
              /* ThreadProblem2D */ {tid, {}, {0, 0}, {m_size, p.head_size}, true},
              /* .block = */ {M_TILE, GemmPV::NTILE, unmasked_size_pad_qk},
              /* .stacksize = */ _cd->getL2CacheSize(),
              /* .tmpcachesize = */ _cd->getL2CacheSize(),
          };
          l_scale.run(  // PxV => O
              PVArgs{
                  utils::GemmProblem{
                      /* .batch */ 1,
                      /* .M = */ std::min(p.sl_q - i_m, M_TILE),
                      /* .N = */ p.head_size,
                      /* .K = */ unmasked_size_pad_qk,
                  },
                  /* .paramA = */ PVProPArgs{(utils::bf16*)tmp, ld_tmp_exp},
                  /* .paramB = */ PVProVArgs{nullptr, 0, &V_pack},
                  /* .paramC = */
                  PVEpiArgs{
                      /* .scale = */ exp_sum,
                      /* .dst = */ head_dst + i_m * p.step_dst_sl,
                      /* .ld_dst = */ p.step_dst_sl,
                  },
                  // /* .workspace = */ nullptr,
              },
              tpPV, /* w_offset */ ibat * V_pack_batch_off);
        }
      }
    });
    utils::afree(bufferK);
    utils::afree(bufferV);
    return BTLA_CODE::Success;
  }

 protected:
  L_ExpSum l_expsum;
  L_Scale l_scale;
};

/**
 * @brief An Epilogue that optionally apply a casual mask (but may not filling zero) and scale the fp32 result, update
 * the maximum of each line of the result, and storing exp as bf16 results
 */
template <BTLA_ISA ISA_T, typename T_SRC, typename T_DST>
class scale_track_max_t {
 public:
  using DType = T_DST;
  using SType = T_SRC;
  struct Param;

  BTLA_CODE forward(const SType* src, const int src_step, const int M_offset, const int N_offset, const int M,
                    const int N, const Param& p) const {
    assert(false);
    return BTLA_CODE::NotSupport;
  }
};
template <BTLA_ISA ISA_T>
class scale_track_max_t<ISA_T, fp16, float> {
 public:
  using DType = float;
  using SType = fp16;
  struct Param {  // NOLINT(readability-identifier-naming): align with bestla name
    DType* dst;
    DType* dst_max;
    int ld_dst;  // #elements
    float scale;
    int causal_offset;  // offset for causal mask; negative value for disabling causal mask
    float alibi_slope;  // m-factor in the alibi paper for current head: https://arxiv.org/abs/2108.12409
  };

  TARGET_512 BTLA_CODE forward(const SType* src, const int src_step, const int M_offset, const int N_offset,
                               const int M, const int N, const Param& p, void* /* tmpcache */,
                               size_t /* cachesize */) const {
    assert(("alibi not supported!", p.alibi_slope == 0.f));
    const auto dst = p.dst + M_offset * p.ld_dst + N_offset;
    const auto dst_max = p.dst_max + M_offset;
#if CompileFP16()
#if MHA_2ND_EXP
    const auto v_scale = _mm512_set1_ps(p.scale);

    for (int i = 0; i < M; ++i) {
      const auto N_unmasked =
          std::min(N, p.causal_offset < 0 ? INT32_MAX : i + M_offset - N_offset + p.causal_offset + 1);

      const auto v_mask = _cvtu32_mask16((1U << (N_unmasked % 16)) - 1);
      int j = 0;
      auto v_max = _mm512_set1_ps(-INFINITY);
      for (; j < N_unmasked - 15; j += 16) {
        const auto xs = _mm512_mul_ps(v_scale, _mm512_cvtxph_ps(_mm256_loadu_ph(src + i * src_step + j)));
        v_max = _mm512_max_ps(v_max, xs);
        _mm512_storeu_ps(dst + i * p.ld_dst + j, xs);
      }
      if (j < N_unmasked) {
        const auto xs = _mm512_mul_ps(
            v_scale, _mm512_cvtxph_ps(_mm256_castsi256_ph(_mm256_maskz_loadu_epi16(v_mask, src + i * src_step + j))));
        v_max = _mm512_mask_max_ps(v_max, v_mask, v_max, xs);
        _mm512_storeu_ps(dst + i * p.ld_dst + j, xs);
        j += 16;
      }
      dst_max[i] = std::max(dst_max[i], _mm512_reduce_max_ps(v_max));

      // if (j < utils::padto(N, 64))
      //   memset(dst + i * p.ld_dst + j, 0, sizeof(*dst) * (utils::padto(N, 64) - j));
    }
#else
    for (int i = 0; i < M; ++i) {
      const auto N_unmasked =
          std::min(N, p.causal_offset < 0 ? INT32_MAX : i + M_offset - N_offset + p.causal_offset + 1);
      for (int j = 0; j < N_unmasked; ++j) {
        const auto val_ = src[i * src_step + j] * p.scale;
        dst[i * p.ld_dst + j] = static_cast<T_DST>(val_);
        dst_max[i] = std::max(dst_max[i], val_);
      }
      if (N_unmasked < utils::padto(N, 64))
        memset(dst + i * p.ld_dst + N_unmasked, 0, sizeof(*dst) * (utils::padto(N, 64) - N_unmasked));
    }
#endif

    return BTLA_CODE::Success;
#else
    return BTLA_CODE::NotSupport;
#endif
  }
};
template <BTLA_ISA ISA_T>
using ScaleTrackMaxFp16Fp32 = scale_track_max_t<ISA_T, fp16, float>;

template <BTLA_ISA ISA_T>
class scale_track_max_t<ISA_T, float, float> {
 public:
  using DType = float;
  using SType = float;
  struct Param {  // NOLINT(readability-identifier-naming): align with bestla name
    DType* dst;
    DType* dst_max;
    int ld_dst;  // #elements
    float scale;
    int causal_offset;  // offset for causal mask; negative value for disabling causal mask
    float alibi_slope;  // m-factor in the alibi paper for current head: https://arxiv.org/abs/2108.12409
  };
  static constexpr float seq15[16]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

  BTLA_CODE forward(const SType* src, const int src_step, const int M_offset, const int N_offset, const int M,
                    const int N, const Param& p, void* /* tmpcache */, size_t /* cachesize */) const {
    return p.alibi_slope == 0 ? forward_<false>(src, src_step, M_offset, N_offset, M, N, p)
                              : forward_<true>(src, src_step, M_offset, N_offset, M, N, p);
  }

#if CompileAVX512F()
  template <bool HAS_ALIBI>
  TARGET_512 BTLA_CODE forward_512(const SType* src, const int src_step, const int M_offset, const int N_offset,
                                   const int M, const int N, const Param& p) const {
    const auto dst = p.dst + M_offset * p.ld_dst + N_offset;
    const auto dst_max = p.dst_max + M_offset;
    const auto v_scale = _mm512_set1_ps(p.scale);
    const auto v_seq15 = _mm512_loadu_ps(seq15);
    const auto alibi_slope = _mm512_set1_ps(p.alibi_slope);
    const auto alibi_base = _mm512_mul_ps(alibi_slope, _mm512_add_ps(v_seq15, _mm512_set1_ps(N_offset)));
    const auto alibi_step = _mm512_set1_ps(p.alibi_slope * 16);

    for (int i = 0; i < M; ++i) {
      auto alibi_curr = alibi_base;
      const auto N_unmasked =
          std::min(N, p.causal_offset < 0 ? INT32_MAX : i + M_offset - N_offset + p.causal_offset + 1);

      const auto v_mask = _cvtu32_mask16((1U << (N_unmasked % 16)) - 1);
      int j = 0;
      auto v_max = _mm512_set1_ps(-INFINITY);
      for (; j < N_unmasked - 15; j += 16) {
        const auto xs = _mm512_fmadd_ps(v_scale, _mm512_loadu_ps(src + i * src_step + j), alibi_curr);
        v_max = _mm512_max_ps(v_max, xs);
        _mm512_storeu_ps(dst + i * p.ld_dst + j, xs);
        if constexpr (HAS_ALIBI) alibi_curr = _mm512_add_ps(alibi_curr, alibi_step);
      }
      if (j < N_unmasked) {
        const auto xs = _mm512_fmadd_ps(v_scale, _mm512_maskz_loadu_ps(v_mask, src + i * src_step + j), alibi_curr);
        v_max = _mm512_mask_max_ps(v_max, v_mask, v_max, xs);
        _mm512_storeu_ps(dst + i * p.ld_dst + j, xs);
        if constexpr (HAS_ALIBI) alibi_curr = _mm512_add_ps(alibi_curr, alibi_step);
        j += 16;
      }
      dst_max[i] = std::max(dst_max[i], _mm512_reduce_max_ps(v_max));

      // if (j < utils::padto(N, 64))
      //   memset(dst + i * p.ld_dst + j, 0, sizeof(*dst) * (utils::padto(N, 64) - j));
    }
    return BTLA_CODE::Success;
  }
#endif
#if CompileAVX2()
  template <bool HAS_ALIBI>
  BTLA_CODE forward_avx2(const SType* src, const int src_step, const int M_offset, const int N_offset, const int M,
                         const int N, const Param& p) const {
    const auto dst = p.dst + M_offset * p.ld_dst + N_offset;
    const auto dst_max = p.dst_max + M_offset;
    const auto v_scale = _mm256_set1_ps(p.scale);
    const auto v_seq7 = _mm256_loadu_ps(seq15);
    const auto alibi_slope = _mm256_set1_ps(p.alibi_slope);
    const auto alibi_base = _mm256_mul_ps(alibi_slope, _mm256_add_ps(v_seq7, _mm256_set1_ps(N_offset)));
    const auto alibi_step = _mm256_set1_ps(p.alibi_slope * 8);
    const auto infinity_neg = _mm256_set1_ps(-INFINITY);
    for (int i = 0; i < M; ++i) {
      auto alibi_curr = alibi_base;
      const auto N_unmasked =
          std::min(N, p.causal_offset < 0 ? INT32_MAX : i + M_offset - N_offset + p.causal_offset + 1);

      const auto v_mask = _mm256_load_si256(reinterpret_cast<const __m256i*>(mask8[N_unmasked % 8]));
      int j = 0;
      auto v_max = infinity_neg;
      for (; j < N_unmasked - 7; j += 8) {
        const auto xs = _mm256_fmadd_ps(v_scale, _mm256_loadu_ps(src + i * src_step + j), alibi_curr);
        v_max = _mm256_max_ps(v_max, xs);
        _mm256_storeu_ps(dst + i * p.ld_dst + j, xs);
        if constexpr (HAS_ALIBI) alibi_curr = _mm256_add_ps(alibi_curr, alibi_step);
      }
      if (j < N_unmasked) {
        const auto xs = _mm256_fmadd_ps(v_scale, _mm256_maskload_ps(src + i * src_step + j, v_mask), alibi_curr);
        const auto masked_xs = _mm256_blendv_ps(infinity_neg, xs, _mm256_castsi256_ps(v_mask));
        v_max = _mm256_max_ps(v_max, masked_xs);
        _mm256_storeu_ps(dst + i * p.ld_dst + j, xs);
        if constexpr (HAS_ALIBI) alibi_curr = _mm256_add_ps(alibi_curr, alibi_step);
        j += 8;
      }
      alignas(32) float dst_tmp[8];
      _mm256_store_ps(dst_tmp, v_max);
      for (int ii = 0; ii < 8; ++ii) dst_max[i] = std::max(dst_max[i], dst_tmp[ii]);
      // if (j < bestla::utils::padto(N, 64))
      //   memset(dst + i * p.ld_dst + j, 0, sizeof(*dst) * (bestla::utils::padto(N, 64) - j));
    }
    return BTLA_CODE::Success;
  }
#endif
  template <bool HAS_ALIBI>
  BTLA_CODE forward_(const SType* src, const int src_step, const int M_offset, const int N_offset, const int M,
                     const int N, const Param& p) const {
    const auto dst = p.dst + M_offset * p.ld_dst + N_offset;
    const auto dst_max = p.dst_max + M_offset;
#if MHA_2ND_EXP
#if CompileAVX512F()
    if constexpr (ISA_T >= BTLA_ISA::AVX512F) {
      return forward_512<HAS_ALIBI>(src, src_step, M_offset, N_offset, M, N, p);
    }
#endif
#if CompileAVX2()
    if constexpr (ISA_T >= BTLA_ISA::AVX2) {
      return forward_avx2<HAS_ALIBI>(src, src_step, M_offset, N_offset, M, N, p);
    }
#endif
#endif

    // reference
    for (int i = 0; i < M; ++i) {
      const auto N_unmasked =
          std::min(N, p.causal_offset < 0 ? INT32_MAX : i + M_offset - N_offset + p.causal_offset + 1);
      for (int j = 0; j < N_unmasked; ++j) {
        const auto val_ = src[i * src_step + j] * p.scale;
        dst[i * p.ld_dst + j] = static_cast<DType>(val_);
        dst_max[i] = std::max(dst_max[i], val_);
      }
      if (N_unmasked < utils::padto(N, 64))
        memset(dst + i * p.ld_dst + N_unmasked, 0, sizeof(*dst) * (utils::padto(N, 64) - N_unmasked));
    }
    return BTLA_CODE::Success;
  }
};
template <BTLA_ISA ISA_T>
using ScaleTrackMaxFp32Fp32 = scale_track_max_t<ISA_T, float, float>;

template <BTLA_ISA ISA_T>
class scale_track_max_t<ISA_T, int32_t, float> {
 public:
  using DType = float;
  using SType = int32_t;
  struct Param {  // NOLINT(readability-identifier-naming): align with bestla name
    DType* dst;
    DType* dst_max;
    int ld_dst;  // #elements
    float scale;
    int causal_offset;  // offset for causal mask; negative value for disabling causal mask
    float alibi_slope;  // m-factor in the alibi paper for current head: https://arxiv.org/abs/2108.12409
  };

  TARGET_512 BTLA_CODE forward(const SType* src, const int src_step, const int M_offset, const int N_offset,
                               const int M, const int N, const Param& p, void* /* tmpcache */,
                               size_t /* cachesize */) const {
    assert(("alibi not supported!", p.alibi_slope == 0.f));
    const auto dst = p.dst + M_offset * p.ld_dst + N_offset;
    const auto dst_max = p.dst_max + M_offset;
#if CompileAVX512F()
    const auto v_scale = _mm512_set1_ps(p.scale);

    for (int i = 0; i < M; ++i) {
      const auto N_unmasked =
          std::min(N, p.causal_offset < 0 ? INT32_MAX : i + M_offset - N_offset + p.causal_offset + 1);

      const auto v_mask = _cvtu32_mask16((1U << (N_unmasked % 16)) - 1);
      int j = 0;
      auto v_max = _mm512_set1_ps(-INFINITY);
      for (; j < N_unmasked - 15; j += 16) {
        const auto xs = _mm512_mul_ps(v_scale, _mm512_cvtepi32_ps(_mm512_loadu_si512(src + i * src_step + j)));
        v_max = _mm512_max_ps(v_max, xs);
        _mm512_storeu_ps(dst + i * p.ld_dst + j, xs);
      }
      if (j < N_unmasked) {
        const auto xs =
            _mm512_mul_ps(v_scale, _mm512_cvtepi32_ps(_mm512_maskz_loadu_epi32(v_mask, src + i * src_step + j)));
        v_max = _mm512_mask_max_ps(v_max, v_mask, v_max, xs);
        _mm512_storeu_ps(dst + i * p.ld_dst + j, xs);
        j += 16;
      }
      dst_max[i] = std::max(dst_max[i], _mm512_reduce_max_ps(v_max));
      // if (j < utils::padto(N, 64))
      //   memset(dst + i * p.ld_dst + j, 0, sizeof(*dst) * (utils::padto(N, 64) - j));
    }
    return BTLA_CODE::Success;
#else
    return BTLA_CODE::NotSupport;
#endif
  }
};
template <BTLA_ISA ISA_T>
using ScaleTrackMaxS32Fp32 = scale_track_max_t<ISA_T, int32_t, float>;

template <class _GemmCore_T, BTLA_ISA ISA_T>
class weight_base_t {
 public:
  using BType = typename _GemmCore_T::BType;
  using SType = BType;
  struct Param {  // NOLINT(readability-identifier-naming): align with bestla name
    const SType* B;
    int ldb;
    bool is_padded;
  };
  weight_base_t() = default;
  BTLA_CODE getWeight(BType** dst_ptr, int* dst_step, const Param& p, int k_size, int n_size, int k_offset,
                      int n_offset, void* /* tmpcache */, size_t /* cachesize */) {
    if ((n_size % _GemmCore_T::NTILE == 0) && std::is_same<SType, BType>::value &&
        0) {  // TODO(Yi) : use a gemm core accept step for K or reorder at runtime
      *dst_ptr = const_cast<SType*>(p.B) + k_offset * p.ldb + n_offset;
      *dst_step = p.ldb;
      return BTLA_CODE::Success;
    } else if (*dst_ptr != nullptr && std::is_same<SType, BType>::value) {
      const auto src = const_cast<SType*>(p.B) + k_offset * p.ldb + n_offset;
      const auto npad = padto(n_size, _GemmCore_T::NTILE);
      *dst_step = npad;
      for (int k = 0; k < k_size; ++k) {
        memcpy(*dst_ptr + k * npad, src + k * p.ldb, sizeof(BType) * n_size);
        memset(*dst_ptr + k * npad + n_size, 0, sizeof(BType) * (npad - n_size));
      }
      return BTLA_CODE::Success;
    } else {
      assert(false);
      return BTLA_CODE::NotSupport;
    }
  }
};
template <class _GemmCore_T, BTLA_ISA ISA_T>
class weight_forward_n_tile48_t {
 public:
  using BType = typename _GemmCore_T::BType;
  using SType = BType;
  struct Param {  // NOLINT(readability-identifier-naming): align with bestla name
    const SType* B;
    int ldb;
    bool is_padded;
  };
  weight_forward_n_tile48_t() = default;
  BTLA_CODE getWeight(BType** dst_ptr, int* dst_step, const Param& p, int k_size, int n_size, int k_offset,
                      int n_offset, void* /* tmpcache */, size_t /* cachesize */) {
    assert(p.is_padded);
    *dst_ptr = const_cast<SType*>(p.B) + k_offset * 48 + n_offset * p.ldb;
    *dst_step = p.ldb;
    return BTLA_CODE::Success;
  }
};
#if CompileAVX512F()
template <class _GemmCore_T, BTLA_ISA ISA_T>
class weight_cvt_bf16_ntile48_t {
 public:
  using BType = typename _GemmCore_T::BType;
  using SType = bf16;
  struct Param {  // NOLINT(readability-identifier-naming): align with bestla name
    const SType* B;
    int ldb;
    bool is_padded;
  };
  weight_cvt_bf16_ntile48_t() = default;
  TARGET_512 BTLA_CODE getWeight(BType** dst_ptr, int* dst_step, const Param& p, int k_size, int n_size, int k_offset,
                                 int n_offset, void* /* tmpcache */, size_t /* cachesize */) {
    assert(p.is_padded);
    const auto src = const_cast<SType*>(p.B) + k_offset * 48 + n_offset * p.ldb;
    const auto dst = *dst_ptr;
    *dst_step = _GemmCore_T::NTILE;
    if constexpr (std::is_same_v<BType, float> && std::is_same_v<SType, utils::bf16>) {
      assert(n_size <= _GemmCore_T::NTILE);
      assert(n_size <= 48);
      assert(n_offset % 2 == 0 && k_offset % 2 == 0);
      // static const auto mask_lo = _cvtu32_mask32(0x55555555U);
      static const auto mask_hi = _cvtu32_mask32(0xaaaaaaaaU);
      for (int i = 0; i < k_size; i += 2) {
        for (int j = 0; j < n_size; j += 16) {
          const auto cur_src = src + i * 48 + j * 2;
          const auto cur_dst = dst + i * 48 + j;
          const auto src_lo = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_loadu_si512(cur_src), 16U));
          const auto src_hi = _mm512_castsi512_ps(_mm512_maskz_loadu_epi16(mask_hi, cur_src));
          _mm512_store_ps(cur_dst + 0, src_lo);
          _mm512_store_ps(cur_dst + 48, src_hi);
        }
      }
    } else {
      assert(0);
    }
    return BTLA_CODE::Success;
  }
};
#endif

#if CompileAVX2()
template <class _GemmCore_T, BTLA_ISA ISA_T>
class weight_cvt_f16_n_tile24_t {  // convert fp16 weight to fp32 using F16C
 public:
  using BType = typename _GemmCore_T::BType;
  using SType = fp16;
  struct Param {  // NOLINT(readability-identifier-naming): align with bestla name
    const SType* B;
    int ldb;
    bool is_padded;
  };
  weight_cvt_f16_n_tile24_t() = default;
  BTLA_CODE getWeight(BType** dst_ptr, int* dst_step, const Param& p, int k_size, int n_size, int k_offset,
                      int n_offset, void* /* tmpcache */, size_t /* cachesize */) {
    assert(p.is_padded);
    const auto src = const_cast<SType*>(p.B) + k_offset * 24 + n_offset * p.ldb;
    const auto dst = *dst_ptr;
    *dst_step = _GemmCore_T::NTILE;
    if constexpr (std::is_same_v<BType, float> && std::is_same_v<SType, utils::fp16>) {
      assert(n_size <= _GemmCore_T::NTILE);
      assert(n_size <= 24);
      assert(n_offset % 24 == 0);
      if (n_size == 24) {
        constexpr auto n_size = 24;
        for (int i = 0; i < k_size; ++i) {
          for (int j = 0; j < n_size; j += 8) {
            const auto cur_src = src + i * 24 + j;
            const auto cur_dst = dst + i * 24 + j;
            const auto src = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(cur_src)));
            _mm256_store_ps(cur_dst, src);
          }
        }
      } else {
        for (int i = 0; i < k_size; ++i) {
          for (int j = 0; j < n_size; j += 8) {
            const auto cur_src = src + i * 24 + j;
            const auto cur_dst = dst + i * 24 + j;
            const auto src = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(cur_src)));
            _mm256_store_ps(cur_dst, src);
          }
        }
      }

    } else {
      assert(0);
    }
    return BTLA_CODE::Success;
  }
};
#endif
template <class SRC_T, class DST_T, BTLA_ISA ISA_T>
struct inplace_precompute_max_softmax_t {
  // nsize is the staring n-size when causal mask enabled
  // src and dst cam be on the same address if sizeof(SRC_T) >= sizeof(DST_T) and ld is correctly set
  // s_max and expsum cam be on the same address
  static void forward(int m_size, int n_size, int n_pad_size, bool is_causal, SRC_T* src, DST_T* dst,
                      const SRC_T* s_max, float* expsum, int ld_src, int ld_dst) {
    assert(false);
  }
};
#if CompileFP16()
template <BTLA_ISA ISA_T>
struct inplace_precompute_max_softmax_t<float, fp16, ISA_T> {
  TARGET_512 static void forward(int m_size, int n_size, int n_pad_size, bool is_causal, float* src, fp16* dst,
                                 const float* s_max, float* expsum, int ld_src, int ld_dst) {
    for (int ii = 0; ii < m_size; ++ii) {
      const auto i_src = src + ii * ld_src;
      const auto i_dst = dst + ii * ld_dst;
      const auto curr_n_size = n_size + (is_causal ? ii : 0);
      const uint16_t v_mask = _cvtu32_mask16((1U << (curr_n_size % 16)) - 1);
      {  // subtract max
        const auto row_max = _mm512_set1_ps(s_max[ii]);
        for (int jj = 0; jj < curr_n_size; jj += 16) {  // should be fine to do extra work on idx >= curr_n_size
          _mm512_storeu_ps(i_src + jj, _mm512_sub_ps(_mm512_loadu_ps(i_src + jj), row_max));
        }
      }
      auto v_sum = _mm512_setzero_ps();
      {  // exp & sum
        int jj = 0;
        for (; jj < curr_n_size / 16 * 16; jj += 16) {
          const auto v_exp = kernel::avx512f::exp_ps_0_1(_mm512_loadu_ps(i_src + jj));
          v_sum = _mm512_add_ps(v_sum, v_exp);
          _mm512_storeu_ps(i_src + jj, v_exp);
        }
        if (jj < curr_n_size) {
          const auto v_exp =
              kernel::avx512f::exp_ps_0_1(_mm512_loadu_ps(i_src + jj));  // should be fine to load some extra
          v_sum = _mm512_mask_add_ps(v_sum, v_mask, v_sum, v_exp);
          _mm512_storeu_ps(i_src + jj, v_exp);  // should be fine to store some extra
        }
        expsum[ii] = _mm512_reduce_add_ps(v_sum);
        v_sum = _mm512_set1_ps(expsum[ii]);
      }
      {  // scale & fp16
        int jj = 0;
        for (; jj < curr_n_size / 16 * 16; jj += 16) {
          const auto v_softmax = _mm512_div_ps(_mm512_loadu_ps(i_src + jj), v_sum);
          _mm256_storeu_ph(i_dst + jj, _mm512_cvtxps_ph(v_softmax));
        }
        if (jj < curr_n_size) {
          const auto v_softmax = _mm512_div_ps(_mm512_loadu_ps(i_src + jj), v_sum);
          _mm256_storeu_ph(i_dst + jj, _mm512_maskz_cvtxps_ph(v_mask, v_softmax));
          jj += 16;
        }
        if (jj < n_pad_size) memset(i_dst + jj, 0, sizeof(fp16) * (n_pad_size - jj));
      }
    }
  }
};
#endif

#if CompileBF16()
template <BTLA_ISA ISA_T>
struct inplace_precompute_max_softmax_t<float, bf16, ISA_T> {
  TARGET_512 static void forward(int m_size, int n_size, int n_pad_size, bool is_causal, float* src, bf16* dst,
                                 const float* s_max, float* expsum, int ld_src, int ld_dst) {
    for (int ii = 0; ii < m_size; ++ii) {
      const auto i_src = src + ii * ld_src;
      const auto i_dst = dst + ii * ld_dst;
      const auto curr_n_size = n_size + (is_causal ? ii : 0);
      const auto v_mask = _cvtu32_mask16((1U << (curr_n_size % 16)) - 1);
      const auto v_mask32 = _cvtu32_mask32((1U << (curr_n_size % 32)) - 1);
      {  // subtract max
        const auto row_max = _mm512_set1_ps(s_max[ii]);
        for (int jj = 0; jj < curr_n_size; jj += 16) {  // should be fine to do extra work on idx >= curr_n_size
          _mm512_storeu_ps(i_src + jj, _mm512_sub_ps(_mm512_loadu_ps(i_src + jj), row_max));
        }
      }
      auto v_sum = _mm512_setzero_ps();
      {  // exp & sum
        int jj = 0;
        for (; jj < curr_n_size / 16 * 16; jj += 16) {
          const auto v_exp = kernel::avx512f::exp_ps_0_1(_mm512_loadu_ps(i_src + jj));
          v_sum = _mm512_add_ps(v_sum, v_exp);
          _mm512_storeu_ps(i_src + jj, v_exp);
        }
        if (jj < curr_n_size) {
          const auto v_exp =
              kernel::avx512f::exp_ps_0_1(_mm512_loadu_ps(i_src + jj));  // should be fine to load some extra
          v_sum = _mm512_mask_add_ps(v_sum, v_mask, v_sum, v_exp);
          _mm512_storeu_ps(i_src + jj, v_exp);  // should be fine to store some extra
        }
        expsum[ii] = _mm512_reduce_add_ps(v_sum);
        v_sum = _mm512_set1_ps(expsum[ii]);
      }
      {  // scale & bf16
        int jj = 0;
        for (; jj < curr_n_size / 32 * 32; jj += 32) {
          const auto v_softmax0 = _mm512_div_ps(_mm512_loadu_ps(i_src + jj), v_sum);
          const auto v_softmax1 = _mm512_div_ps(_mm512_loadu_ps(i_src + jj + 16), v_sum);
          _mm512_storeu_epi16(i_dst + jj, (__m512i)_mm512_cvtne2ps_pbh(v_softmax1, v_softmax0));
        }
        if (jj < curr_n_size) {
          const auto v_softmax0 = _mm512_div_ps(_mm512_loadu_ps(i_src + jj), v_sum);
          const auto v_softmax1 = _mm512_div_ps(_mm512_loadu_ps(i_src + jj + 16), v_sum);
#if defined(__GNUC__) && (__GNUC__ == 13) && (__GNUC_MINOR__ <= 2)
          // There is a bug on gcc 13.1/13.2 what reverse the parameter order;
          // A GUN team member said that it will befixed in GCC 13.3
          _mm512_storeu_epi16(i_dst + jj, (__m512i)_mm512_maskz_cvtne2ps_pbh(v_mask32, v_softmax0, v_softmax1));
#else
          _mm512_storeu_epi16(i_dst + jj, (__m512i)_mm512_maskz_cvtne2ps_pbh(v_mask32, v_softmax1, v_softmax0));
#endif
          jj += 32;
        }
        if (jj < n_pad_size) memset(i_dst + jj, 0, sizeof(bf16) * (n_pad_size - jj));
      }
    }
  }
};
#endif

#if CompileAVX512F()
template <BTLA_ISA ISA_T>
struct inplace_precompute_max_softmax_t<std::enable_if_t<ISA_T >= BTLA_ISA::AVX512F, float>, float, ISA_T> {
  TARGET_512 static void forward(  // NOLINT [build/include_what_you_use]
      int m_size, int n_size, int n_pad_size, bool is_causal, float* src, float* dst, const float* s_max, float* expsum,
      int ld_src, int ld_dst) {
    for (int ii = 0; ii < m_size; ++ii) {
      const auto i_src = src + ii * ld_src;
      const auto i_dst = dst + ii * ld_dst;
      const auto curr_n_size = n_size + (is_causal ? ii : 0);
      const auto v_mask = _cvtu32_mask16((1U << (curr_n_size % 16)) - 1);
      {  // subtract max
        const auto row_max = _mm512_set1_ps(s_max[ii]);
        for (int jj = 0; jj < curr_n_size; jj += 16) {  // should be fine to do extra work on idx >= curr_n_size
          _mm512_storeu_ps(i_src + jj, _mm512_sub_ps(_mm512_loadu_ps(i_src + jj), row_max));
        }
      }
      auto v_sum = _mm512_setzero_ps();
      {  // exp & sum
        int jj = 0;
        for (; jj < curr_n_size / 16 * 16; jj += 16) {
          const auto v_exp = kernel::avx512f::exp_ps_0_1(_mm512_loadu_ps(i_src + jj));
          v_sum = _mm512_add_ps(v_sum, v_exp);
          _mm512_storeu_ps(i_src + jj, v_exp);
        }
        if (jj < curr_n_size) {
          const auto v_exp =
              kernel::avx512f::exp_ps_0_1(_mm512_loadu_ps(i_src + jj));  // should be fine to load some extra
          v_sum = _mm512_mask_add_ps(v_sum, v_mask, v_sum, v_exp);
          _mm512_storeu_ps(i_src + jj, v_exp);  // should be fine to store some extra
        }
        expsum[ii] = _mm512_reduce_add_ps(v_sum);
        v_sum = _mm512_set1_ps(expsum[ii]);
      }
      {  // scale & store
        int jj = 0;
        for (; jj < padto_le(curr_n_size, 16); jj += 16) {
          _mm512_store_ps(i_dst + jj, _mm512_div_ps(_mm512_loadu_ps(i_src + jj), v_sum));
        }
        if (jj < curr_n_size) {
          _mm512_store_ps(i_dst + jj, _mm512_maskz_div_ps(v_mask, _mm512_loadu_ps(i_src + jj), v_sum));
          jj += 16;
        }
        if (jj < n_pad_size) memset(i_dst + jj, 0, sizeof(bf16) * (n_pad_size - jj));
      }
    }
  }
};
#endif

#if CompileAVX2()
template <BTLA_ISA ISA_T>
struct inplace_precompute_max_softmax_t<std::enable_if_t<(ISA_T < BTLA_ISA::AVX512F && ISA_T >= BTLA_ISA::AVX2), float>,
                                        float, ISA_T> {
  static void forward(  // NOLINT [build/include_what_you_use]
      int m_size, int n_size, int n_pad_size, bool is_causal, float* src, float* dst, const float* s_max, float* expsum,
      int ld_src, int ld_dst) {
    for (int ii = 0; ii < m_size; ++ii) {
      const auto i_src = src + ii * ld_src;
      const auto i_dst = dst + ii * ld_dst;
      const auto curr_n_size = n_size + (is_causal ? ii : 0);
      const auto v_mask = _mm256_load_si256(reinterpret_cast<const __m256i*>(mask8[curr_n_size % 8]));
      {  // subtract max
        const auto row_max = _mm256_set1_ps(s_max[ii]);
        for (int jj = 0; jj < curr_n_size; jj += 8) {  // should be fine to do extra work on idx >= curr_n_size
          _mm256_storeu_ps(i_src + jj, _mm256_sub_ps(_mm256_loadu_ps(i_src + jj), row_max));
        }
      }
      auto v_sum = _mm256_setzero_ps();
      {  // exp & sum
        int jj = 0;
        for (; jj < padto_le(curr_n_size, 8); jj += 8) {
          const auto v_exp = kernel::avx2::exp_ps_0_1(_mm256_loadu_ps(i_src + jj));
          v_sum = _mm256_add_ps(v_sum, v_exp);
          _mm256_storeu_ps(i_src + jj, v_exp);
        }
        if (jj < curr_n_size) {
          const auto v_exp = kernel::avx2::exp_ps_0_1(_mm256_loadu_ps(i_src + jj));  // should be fine to load extra
          v_sum = _mm256_add_ps(v_sum, _mm256_and_ps(v_exp, _mm256_castsi256_ps(v_mask)));
          _mm256_storeu_ps(i_src + jj, v_exp);  // should be fine to store some extra
        }

        alignas(32) float sum_tmp[8];
        _mm256_store_ps(sum_tmp, v_sum);
        expsum[ii] = 0.f;
        for (int iii = 0; iii < 8; ++iii) expsum[ii] += sum_tmp[iii];
        v_sum = _mm256_set1_ps(expsum[ii]);
      }
      {  // scale & store
        int jj = 0;
        for (; jj < padto_le(curr_n_size, 8); jj += 8) {
          _mm256_store_ps(i_dst + jj, _mm256_div_ps(_mm256_loadu_ps(i_src + jj), v_sum));
        }
        if (jj < curr_n_size) {
          const auto quotient = _mm256_div_ps(_mm256_loadu_ps(i_src + jj), v_sum);
          _mm256_store_ps(i_dst + jj, _mm256_and_ps(quotient, _mm256_castsi256_ps(v_mask)));
          jj += 8;
        }
        if (jj < n_pad_size) memset(i_dst + jj, 0, sizeof(float) * (n_pad_size - jj));
      }
    }
  }
};
#endif
template <BTLA_ISA ISA_T>
struct inplace_precompute_max_softmax_t<float, uint8_t, ISA_T> {
  TARGET_512 static void forward(int m_size, int n_size, int n_pad_size, bool is_causal, float* src, uint8_t* dst,
                                 float* s_max, float* expsum, int ld_src, int ld_dst) {
    for (int ii = 0; ii < m_size; ++ii) {
      const auto i_src = src + ii * ld_src;
      const auto i_dst = dst + ii * ld_dst;
      const auto curr_n_size = n_size + (is_causal ? ii : 0);
      const uint16_t v_mask = _cvtu32_mask16((1U << (curr_n_size % 16)) - 1);
      {  // subtract max
        const auto row_max = _mm512_set1_ps(s_max[ii]);
        for (int jj = 0; jj < curr_n_size; jj += 16) {  // should be fine to do extra work on idx >= curr_n_size
          _mm512_storeu_ps(i_src + jj, _mm512_sub_ps(_mm512_loadu_ps(i_src + jj), row_max));
        }
      }
      {  // exp & sum
        auto v_sum = _mm512_setzero_ps();
        int jj = 0;
        for (; jj < curr_n_size / 16 * 16; jj += 16) {
          const auto v_exp = kernel::avx512f::exp_ps_0_1(_mm512_loadu_ps(i_src + jj));
          v_sum = _mm512_add_ps(v_sum, v_exp);
          _mm512_storeu_ps(i_src + jj, v_exp);
        }
        if (jj < curr_n_size) {
          const auto v_exp =
              kernel::avx512f::exp_ps_0_1(_mm512_loadu_ps(i_src + jj));  // should be fine to load some extra
          v_sum = _mm512_mask_add_ps(v_sum, v_mask, v_sum, v_exp);
          _mm512_storeu_ps(i_src + jj, v_exp);  // should be fine to store some extra
        }
        expsum[ii] = _mm512_reduce_add_ps(v_sum);
      }
      {  // scale & int8
        const auto v_scale = _mm512_set1_ps(UINT8_MAX);
        int jj = 0;
        for (; jj < curr_n_size / 16 * 16; jj += 16) {
          const auto v_softmax = _mm512_mul_ps(_mm512_loadu_ps(i_src + jj), v_scale);
          _mm_storeu_si128(reinterpret_cast<__m128i*>(i_dst + jj),
                           _mm512_cvtusepi32_epi8(_mm512_cvtps_epu32(v_softmax)));
        }
        if (jj < curr_n_size) {
          const auto v_softmax = _mm512_mul_ps(_mm512_loadu_ps(i_src + jj), v_scale);
          _mm_storeu_si128(reinterpret_cast<__m128i*>(i_dst + jj),
                           _mm512_maskz_cvtusepi32_epi8(v_mask, _mm512_cvtps_epu32(v_softmax)));
          jj += 16;
        }
        if (jj < n_pad_size) memset(i_dst + jj, 0, sizeof(uint8_t) * (n_pad_size - jj));
      }
    }
  }
};

/**
 * @brief MHA interface with N-dim parallelism & stable softmax
 *
 * @tparam L_Max Launcher type of the QK matmul; tracking the dst max value of each row
 * @tparam L_Scale Launcher type of the PV scale matmul (S for that in the flash-attn paper)
 */
template </* class Parallel_T, */ class L_Max, class L_Scale>
class mha_stable_interface_t {
  template <class EpiArgs, bool HAS_SCALE, class T>
  static inline typename std::enable_if<!HAS_SCALE, EpiArgs>::type composeEpiArgs(float*, T* dst, int ld_dst) {
    return {dst, ld_dst};
  }
  template <class EpiArgs, bool HAS_SCALE, class T>
  static inline typename std::enable_if<HAS_SCALE, EpiArgs>::type composeEpiArgs(float* scale, T* dst, int ld_dst) {
    return {scale, dst, ld_dst};
  }

 public:
  using PrologueQ = typename L_Max::PrologueA;
  using PrologueK = typename L_Max::PrologueB;
  using QKProQArgs = typename PrologueQ::Param;
  using QKProKArgs = typename PrologueK::Param;
  using QKArgs = typename L_Max::Param;
  using QKEpiArgs = typename L_Max::EpiParam;

  using PrologueS = typename L_Scale::PrologueA;
  using PrologueV = typename L_Scale::PrologueB;
  using PVProPArgs = typename PrologueS::Param;
  using PVProVArgs = typename PrologueV::Param;
  using PVArgs = typename L_Scale::Param;
  using PVEpiArgs = typename L_Scale::EpiParam;

  using GemmQK = typename L_Max::GemmCore;
  using GemmPV = typename L_Scale::GemmCore;
  using Q_T = typename std::remove_const<typename std::remove_pointer<decltype(QKProQArgs::A)>::type>::type;
  using K_T = typename PrologueK::SType;
  using V_T = typename PrologueV::SType;
  using DST_T = typename L_Scale::Epilogue::DType;

  static constexpr auto RT_ISA = std::max(L_Max::RT_ISA, L_Scale::RT_ISA);

  static_assert(GemmQK::MTILE == GemmPV::MTILE, "2 GEMM should have the same M_TILE.");
  static constexpr auto M_TILE = GemmQK::MTILE;

  BTLA_CODE compute(const attn_fwd_args_t<Q_T, K_T, V_T, DST_T>& p, parallel::IThreading& th) {
    assert((std::is_same<Q_T, int8_t>::value || p.Q_sc == 1));
    assert((std::is_same<K_T, int8_t>::value || p.K_sc == 1));
    assert((std::is_same<V_T, int8_t>::value || p.V_sc == 1));
    assert((std::is_same<DST_T, int8_t>::value || p.dst_sc == 1));

    assert((p.Q_layout == ATTN_FWD_LAYOUT_PLAIN && p.dst_layout == ATTN_FWD_LAYOUT_PLAIN));
    assert((p.K_layout == ATTN_FWD_LAYOUT_PLAIN ||
            (std::is_same<K_T, int8_t>::value && p.K_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK4) ||
            (std::is_same<K_T, bf16>::value && p.K_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2) ||
            (std::is_same<K_T, fp16>::value && p.K_layout == ATTN_FWD_LAYOUT_NTILE24_ROWPACK1)));
    assert((p.V_layout == ATTN_FWD_LAYOUT_PLAIN ||
            (std::is_same<V_T, int8_t>::value && p.V_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK4) ||
            (std::is_same<V_T, bf16>::value && p.V_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2) ||
            (std::is_same<V_T, fp16>::value && p.V_layout == ATTN_FWD_LAYOUT_NTILE24_ROWPACK1)));

    assert((!std::is_same<  //
               PrologueK, mha::weight_forward_n_tile48_t<typename L_Max::GemmCore, L_Max::RT_ISA>>::value) ||
           p.K_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK4 ||
           p.K_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2);  // WeightForward can only be used with preprocessed layout

    assert((!std::is_same<  //
               PrologueV, mha::weight_forward_n_tile48_t<typename L_Scale::GemmCore, L_Scale::RT_ISA>>::value) ||
           p.V_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK4 ||
           p.V_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2);  // WeightForward can only be used with preprocessed layout

    assert((p.K_layout != ATTN_FWD_LAYOUT_PLAIN || p.step_v_head_size == 1));
    assert((p.V_layout != ATTN_FWD_LAYOUT_PLAIN || p.step_k_sl == 1));
    const auto num_heads = p.batch_size * p.head_num;  // Total number of heads
    GetCPUDevice();
    const bool is_causal = (p.attn_flags & NE_ATTN_FLAG_IS_CAUSAL) != 0;
    const bool is_alibi = (p.attn_flags & NE_ATTN_FLAG_IS_ALIBI8) != 0;
    const bool prefer_fp32 = (p.attn_flags & NE_ATTN_FLAG_PREFER_FP32) != 0;

    assert(("prefer_fp32 not followed!",  //
            !prefer_fp32 || (GemmQK::COMP == bestla::gemm::CompType::COMP_FP32 &&
                             GemmPV::COMP == bestla::gemm::CompType::COMP_FP32)));
    assert(("qlen should be no greater then klen/vlen!", !is_causal || p.sl_q <= p.sl_kv));
    assert(!is_causal || p.sl_q <= p.sl_kv);
    assert(("head_num must be a multiple of heads_kv!", p.head_num % p.heads_kv == 0));
    const auto group_heads = p.head_num / p.heads_kv;
    const auto sl_diff = p.sl_kv - p.sl_q;

    // TP will need the real rank order of k
    int32_t k_offset = 0;
    int32_t log_head_num = p.head_num;
#ifdef NS_TP_MODEL
    parallel_context* p_ctx = init_parallel_context();
    int32_t world_size = get_tp_size(p_ctx);
    int32_t rank = get_tp_rank(p_ctx);
    if (world_size > 1) k_offset += rank * p.head_num;
    log_head_num *= world_size;
#endif

    // alibi slope
    const int n_heads_log2_floor = 1 << static_cast<int>(floor(log2(log_head_num)));
    const float m0 = powf(2.0f, -(8.f) / n_heads_log2_floor);
    const float m1 = powf(2.0f, -(8.f / 2.0f) / n_heads_log2_floor);

    const auto m_tiles = updiv(p.sl_q, M_TILE);
    const auto num_tasks = num_heads * m_tiles;

    using Scheduler2D = bestla::parallel::Scheduler2D;
    const Scheduler2D parl({th.num_threads(), {num_tasks, 1}, {1, 1}, {0, 0}});  // main parallel scheduler

    th.parallel_for([&](int tid) {
      const int tmp_s_size = M_TILE * padto(padto(p.sl_kv, GemmQK::NTILE), GemmPV::KTILE);
      const int tmp_p_size = tmp_s_size;
      const int tmp_bytes = tmp_s_size * sizeof(float);  // S & exp
      const auto tmp_s = reinterpret_cast<float*>(p.tmp + tid * tmp_bytes);
      using PType = typename GemmPV::AType;
      const auto tmp_p = reinterpret_cast<PType*>(tmp_s);  // overwrite tmp_s row-wisely

      // calculate mm + softmax + mm
      {
        typename parallel::ThreadProblem2D thdp{tid};
        parl.getIndex(thdp);
        const auto [task_start, _assert0] = thdp.loc;
        auto [task_size, _assert_max1] = thdp.size;
        assert(task_size == 0 || _assert0 == 0);
        assert(task_size == 0 || _assert_max1 == 1 || _assert_max1 == 0);
        if (_assert_max1 == 0 || !thdp.valid) task_size = 0;

        for (int task_id = task_start; task_id < task_start + task_size; ++task_id) {
          const int ibat = task_id / m_tiles;
          const int i_m = task_id % m_tiles * M_TILE;
          const int ibs = ibat / p.head_num;
          const int ihn = ibat % p.head_num;
          const int ihkv = ihn / group_heads;
          const int m_size = std::min(M_TILE, p.sl_q - i_m);

          const auto alibi_ihn_m = !is_alibi ? 0.f
                                   : (ihn + k_offset < n_heads_log2_floor)
                                       ? powf(m0, ihn + k_offset + 1)
                                       : powf(m1, 2 * (ihn + k_offset - n_heads_log2_floor) + 1);

          float s_max[M_TILE]{};  // maximum for each row of the S matrix
          std::fill_n(s_max, M_TILE, -INFINITY);

          // ptr to Q / dst matrix of the current head
          const auto head_q = p.Q + ibs * p.step_q_bs + ihn * p.step_q_head_num;
          const auto head_k = p.K + ibs * p.step_k_bs + ihkv * p.step_k_head_num;
          const auto head_v = p.V + ibs * p.step_v_bs + ihkv * p.step_v_head_num;
          const auto head_dst = p.dst + ibs * p.step_dst_bs + ihn * p.step_dst_head_num;
          const auto unmasked_size = is_causal ? std::min(p.sl_kv, sl_diff + i_m + M_TILE - 1 + 1) : p.sl_kv;

          const auto unmasked_size_pad_qk = std::min(p.sl_kv, padto(unmasked_size, GemmQK::NTILE));
          const auto unmasked_size_pad_pv = std::min(p.sl_kv, padto(unmasked_size, GemmPV::KTILE));
          const int ld_tmp_s = padto(padto(unmasked_size_pad_pv, GemmQK::NTILE), GemmPV::KTILE);
          static_assert(sizeof(float) >= sizeof(PType), "PType exceeded float size!");
          const int ld_tmp_p = ld_tmp_s * sizeof(float) / sizeof(PType);
          const auto qk_prok_ldb = p.step_k_sl == 1                                 ? p.step_k_head_size
                                   : p.K_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK4 ? p.step_k_sl
                                   : p.K_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2 ? p.step_k_sl
                                   : p.K_layout == ATTN_FWD_LAYOUT_NTILE24_ROWPACK1 ? p.step_k_sl
                                                                                    : (assert(0), 0);

          typename parallel::gemm::ThreadProblemBase tpQK{
              /* ThreadProblem2D */ {tid, {}, {i_m, 0}, {m_size, unmasked_size_pad_qk}, true},
              /* .block = */ {M_TILE, GemmQK::NTILE, p.head_size},
              /* .stacksize = */ _cd->getL2CacheSize(),
              /* .tmpcachesize = */ _cd->getL2CacheSize(),
          };
          l_qk.run(  // QxK => S ==exp==> P
              QKArgs{
                  utils::GemmProblem{
                      /* .batch */ 1,
                      /* .M = */ p.sl_q,
                      /* .N = */ unmasked_size_pad_qk,
                      /* .K = */ p.head_size,
                  },
                  /* .paramA = */
                  QKProQArgs{
                      head_q,
                      p.step_q_sl,
                  },
                  /* .paramB = */
                  QKProKArgs{
                      /* .B = */ head_k,
                      /* .ldb = */ qk_prok_ldb,
                      /* .is_padded = */ true,
                  },  // K should be pre-transposed
                  /* .paramC = */
                  QKEpiArgs{
                      /* .dst = */ tmp_s - i_m * ld_tmp_s,  // pretend that there is a whole S mat
                      /* .dst_sum = */ s_max - i_m,         // pretend that there is a whole S mat
                      /* .ld_dst = */ ld_tmp_s,
                      /* .scale = */ p.QK_scale * p.Q_sc * p.K_sc,
                      /* .causal_offset = */ is_causal ? sl_diff : -1,
                      /* .alibi_slope = */ alibi_ihn_m,
                  },
                  // /* .workspace = */ nullptr,
              },
              tpQK);

          // softmax (with pre-computed row_max)
          const auto unmasked_size_start = is_causal ? std::min(sl_diff + i_m + 1, p.sl_kv) : p.sl_kv;
          float expsum[M_TILE]{};  // maximum for each row of the S matrix
          const auto softmax_npad_size = padto(unmasked_size_pad_pv, GemmPV::KTILE);
          inplace_precompute_max_softmax_t<float, PType, RT_ISA>::forward(  //
              m_size, unmasked_size_start, softmax_npad_size,               // m / n
              is_causal, tmp_s, tmp_p, s_max, expsum, ld_tmp_s, ld_tmp_p);  //

          const auto pv_scale = expsum;
          for (int i = 0; i < M_TILE; ++i) pv_scale[i] = p.V_sc / UINT8_MAX / expsum[i] / p.dst_sc;

          const auto pv_prov_ldb = p.step_v_head_size == 1                          ? p.step_v_sl
                                   : p.V_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK4 ? p.step_v_head_size
                                   : p.V_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2 ? p.step_v_head_size
                                   : p.V_layout == ATTN_FWD_LAYOUT_NTILE24_ROWPACK1 ? p.step_v_head_size
                                                                                    : (assert(0), 0);

          typename parallel::gemm::ThreadProblemBase tpPV{
              /* ThreadProblem2D */ {tid, {}, {0, 0}, {m_size, p.head_size}, true},
              /* .block = */ {M_TILE, GemmPV::NTILE, unmasked_size_pad_pv},
              /* .stacksize = */ _cd->getL2CacheSize(),
              /* .tmpcachesize = */ _cd->getL2CacheSize(),
          };
          l_pv.run(  // PxV => O
              PVArgs{
                  utils::GemmProblem{
                      /* .batch */ 1,
                      /* .M = */ std::min(p.sl_q - i_m, M_TILE),
                      /* .N = */ p.head_size,
                      /* .K = */ unmasked_size_pad_pv,
                  },
                  /* .paramA = */ PVProPArgs{tmp_p, ld_tmp_p},
                  /* .paramB = */
                  PVProVArgs{
                      /* .B = */ head_v,
                      /* .ldb = */ pv_prov_ldb,
                      /* .is_padded = */ true,
                  },
                  /* .paramC = */
                  composeEpiArgs<PVEpiArgs, std::is_same<V_T, int8_t>::value>(  //
                      pv_scale, head_dst + i_m * p.step_dst_sl, p.step_dst_sl),
                  // /* .workspace = */ nullptr,
              },
              tpPV);
        }
      }
    });
    return BTLA_CODE::Success;
  }

 protected:
  L_Max l_qk;
  L_Scale l_pv;
};

template <typename Q_T, typename K_T, typename V_T, typename DST_T>
inline void bestla_fusion_attn_forward(const attn_fwd_args_t<Q_T, K_T, V_T, DST_T>& params) = delete;

template <class GEMM_T, BTLA_ISA ISA_T>
using WeightPackBatchBf16Bf16NonTr = weight_pack_batch_bf16_non_tr_t<GEMM_T, ISA_T, bf16>;
template <class GEMM_T, BTLA_ISA ISA_T>
using WeightPackBatchBf16Bf16Trans = weight_pack_batch_bf16_trans_t<GEMM_T, ISA_T, bf16>;
template <>
inline void bestla_fusion_attn_forward<bf16, bf16, bf16, bf16>(const attn_fwd_args_t<bf16, bf16, bf16, bf16>& p) {
  using GemmKernelBF16ExpSum = mha::launcher_base_off_t<  //
      BTLA_ISA::AMX_BF16,                                 //
      gemm::HCoreRowNAmxbf16<64, 16>,                     //
      prologue_a::gemm::ActivationBase,                   //
      WeightPackBatchBf16Bf16Trans,                       //
      mha::ScaleExpAccSumFp32Bf16>;                       //
  using GemmKernelBF16 = mha::launcher_base_off_t<        //
      BTLA_ISA::AMX_BF16,                                 //
      gemm::HCoreRowNAmxbf16<64, 16>,                     //
      prologue_a::gemm::ActivationBase,                   //
      WeightPackBatchBf16Bf16NonTr,                       //
      mha::ScaleWriteBackFp32Bf16>;
  static mha_interface_t<GemmKernelBF16ExpSum, GemmKernelBF16> kernel;
  const auto pth = ne_threading::get();
  [[maybe_unused]] const auto ret = kernel.compute(p, *pth);
  assert(ret == BTLA_CODE::Success);
}

template <class GEMM_T, BTLA_ISA ISA_T>
using WeightPackBatchFp16Bf16NonTr = weight_pack_batch_bf16_non_tr_t<GEMM_T, ISA_T, fp16>;
template <class GEMM_T, BTLA_ISA ISA_T>
using WeightPackBatchFp16Bf16Trans = weight_pack_batch_bf16_trans_t<GEMM_T, ISA_T, fp16>;
template <>
inline void bestla_fusion_attn_forward<float, fp16, fp16, float>(
    const attn_fwd_args_t<float, fp16, fp16, float>& params) {
  GetCPUDevice();
  const auto pth = ne_threading::get();
  if (MHA_PREFER_AVX512FP16 && _cd->AVX512_FP16() && params.step_k_sl == 1) {
    using GemmKernelFP16TrackMax = mha::launcher_base_weight_t<  //
        BTLA_ISA::AVX512_FP16,                                   //
        gemm::HCoreRowNAvx512fp16<64, 8>,                        //
        prologue_a::gemm::ActivationConverterFp32,               //
        mha::weight_base_t,                                      //
        mha::ScaleTrackMaxFp16Fp32>;                             //
    using GemmKernelFP16 = mha::launcher_base_weight_t<          //
        BTLA_ISA::AVX512_FP16,                                   //
        gemm::HCoreRowNAvx512fp16<64, 8>,                        //
        prologue_a::gemm::ActivationBase,                        //
        mha::weight_base_t,                                      //
        bestla::epilogue::gemm::AccumulatorWriteBackFp16Fp32>;
    static mha_stable_interface_t<GemmKernelFP16TrackMax, GemmKernelFP16> kernel;
    [[maybe_unused]] const auto ret = kernel.compute(params, *pth);
    assert(ret == BTLA_CODE::Success);
  } else if (_cd->AMX_BF16() &&                           //
             params.K_layout == ATTN_FWD_LAYOUT_PLAIN &&  //
             params.V_layout == ATTN_FWD_LAYOUT_PLAIN) {
    if (params.step_k_head_size == 1) {
      using GemmKernelFP32FP16BF16ExpSum = mha::launcher_base_off_t<  //
          BTLA_ISA::AMX_BF16,                                         //
          gemm::HCoreRowNAmxbf16<64, 16>,                             //
          prologue_a::gemm::ActivationConverterFp32,                  //
          WeightPackBatchFp16Bf16Trans,                               //
          mha::ScaleExpAccSumFp32Bf16>;                               //
      using GemmKernelBF16FP16FP32 = mha::launcher_base_off_t<        //
          BTLA_ISA::AMX_BF16,                                         //
          gemm::HCoreRowNAmxbf16<64, 16>,                             //
          prologue_a::gemm::ActivationBase,                           //
          WeightPackBatchFp16Bf16NonTr,                               //
          mha::ScaleWriteBackFp32Fp32>;
      static mha_interface_t<GemmKernelFP32FP16BF16ExpSum, GemmKernelBF16FP16FP32> kernel;
      [[maybe_unused]] const auto ret = kernel.compute(params, *pth);
      assert(ret == BTLA_CODE::Success);
    } else if (params.step_k_sl == 1) {
      using GemmKernelFP32FP16BF16ExpSum = mha::launcher_base_off_t<  //
          BTLA_ISA::AMX_BF16,                                         //
          gemm::HCoreRowNAmxbf16<64, 16>,                             //
          prologue_a::gemm::ActivationConverterFp32,                  //
          WeightPackBatchFp16Bf16NonTr,                               //
          mha::ScaleExpAccSumFp32Bf16>;                               //
      using GemmKernelBF16FP16FP32 = mha::launcher_base_off_t<        //
          BTLA_ISA::AMX_BF16,                                         //
          gemm::HCoreRowNAmxbf16<64, 16>,                             //
          prologue_a::gemm::ActivationBase,                           //
          WeightPackBatchFp16Bf16NonTr,                               //
          mha::ScaleWriteBackFp32Fp32>;
      static mha_interface_t<GemmKernelFP32FP16BF16ExpSum, GemmKernelBF16FP16FP32> kernel;
      [[maybe_unused]] const auto ret = kernel.compute(params, *pth);
      assert(ret == BTLA_CODE::Success);
    }
  } else if (_cd->AVX2() &&  //
             params.K_layout == ATTN_FWD_LAYOUT_NTILE24_ROWPACK1 &&
             params.V_layout == ATTN_FWD_LAYOUT_NTILE24_ROWPACK1) {
    using GemmKernelTrackMax = mha::launcher_base_weight_t<  //
        BTLA_ISA::AVX2,                                      //
        gemm::SCoreRowNAvx2<24, 4>,                          //
        prologue_a::gemm::ActivationBase,                    //
        mha::weight_cvt_f16_n_tile24_t,                      //
        mha::ScaleTrackMaxFp32Fp32>;                         //
    using GemmKernelId = mha::launcher_base_weight_t<        //
        BTLA_ISA::AVX2,                                      //
        gemm::SCoreRowNAvx2<24, 4>,                          //
        mha::activation_identity_t,                          // pretty sure we have enough paddings for P-matrix
        mha::weight_cvt_f16_n_tile24_t,                      //
        bestla::epilogue::gemm::AccumulatorWriteBackFp32>;   //
    static mha_stable_interface_t<GemmKernelTrackMax, GemmKernelId> mha;
    [[maybe_unused]] const auto ret = mha.compute(params, *pth);
    assert(ret == BTLA_CODE::Success);
  } else {
    assert(false);  // no suitbale launcher
  }
}

template <>
inline void bestla_fusion_attn_forward<fp16, fp16, fp16, fp16>(const attn_fwd_args_t<fp16, fp16, fp16, fp16>& params) {
  GetCPUDevice();
  const auto pth = ne_threading::get();
  if (_cd->AMX_BF16()) {
    using GemmKernelFP16TrackMax = mha::launcher_base_weight_t<  //
        BTLA_ISA::AVX512_FP16,                                   //
        gemm::HCoreRowNAvx512fp16<64, 8>,                        //
        prologue_a::gemm::ActivationBase,                        //
        mha::weight_base_t,                                      //
        mha::ScaleTrackMaxFp16Fp32>;                             //
    using GemmKernelFP16 = mha::launcher_base_weight_t<          //
        BTLA_ISA::AVX512_FP16,                                   //
        gemm::HCoreRowNAvx512fp16<64, 8>,                        //
        prologue_a::gemm::ActivationBase,                        //
        mha::weight_base_t,                                      //
        bestla::epilogue::gemm::AccumulatorWriteBackFp16>;
    static mha_stable_interface_t<GemmKernelFP16TrackMax, GemmKernelFP16> kernel;
    [[maybe_unused]] const auto ret = kernel.compute(params, *pth);
    assert(ret == BTLA_CODE::Success);
  } else {
    assert(0);
  }
}

template <>
inline void bestla_fusion_attn_forward<int8_t, int8_t, int8_t, int8_t>(
    const attn_fwd_args_t<int8_t, int8_t, int8_t, int8_t>& params) {
  GetCPUDevice();
  const auto pth = ne_threading::get();
  if (/* params.sl_q > 4 &&  */ _cd->AMX_INT8()) {                // TODO(Yi): add vnni impl
    using GemmKernelInt32TrackMax = mha::launcher_base_weight_t<  //
        BTLA_ISA::AMX_INT8,                                       //
        gemm::ICoreRowNAmxint8SS<48, 16>,                         //
        prologue_a::gemm::ActivationBase,                         //
        mha::weight_forward_n_tile48_t,                           //
        mha::ScaleTrackMaxS32Fp32>;                               //
    using GemmKernelInt32 = mha::launcher_base_weight_t<          //
        BTLA_ISA::AMX_INT8,                                       //
        gemm::ICoreRowNAmxint8<48, 16>,                           //
        prologue_a::gemm::ActivationBase,                         //
        mha::weight_forward_n_tile48_t,                           //
        mha::ScaleWriteBackS32S8>;                                //
    static mha_stable_interface_t<GemmKernelInt32TrackMax, GemmKernelInt32> mha;
    [[maybe_unused]] const auto ret = mha.compute(params, *pth);
    assert(ret == BTLA_CODE::Success);
  } else if (_cd->AVX512_VNNI()) {
    // using GemmKernelInt32TrackMax = mha::launcher_base_weight_t<  //
    //     BTLA_ISA::AMX_INT8,                                       // TODO(Yi): s8s8 vnni kernel?
    //     gemm::GemmCore_Row_NN_16x48_AMX_S8S8,                     //
    //     prologue::gemm::ActivationBase,                           //
    //     mha::weight_forward_n_tile48_t,                           //
    //     mha::ScaleTrackMaxS32Fp32>;                               //
    // using GemmKernelInt32 = mha::launcher_base_weight_t<          //
    //     BTLA_ISA::AVX512_VNNI,                                    //
    //     gemm::GemmCore_Row_NN_8x48_AVX512_VNNI,                   //
    //     prologue::gemm::ActivationBase,                           //
    //     mha::weight_forward_n_tile48_t,                           //
    //     mha::ScaleWriteBackS32S8>;                                //
    // static mha_stable_interface_t<GemmKernelInt32TrackMax, GemmKernelInt32> mha;
    // [[maybe_unused]] const auto ret = mha.compute(params);
    // assert(ret == BTLA_CODE::Success);
    assert(0);
  } else {
    assert(0);
  }
}

template <>
inline void bestla_fusion_attn_forward<float, bf16, bf16, float>(
    const attn_fwd_args_t<float, bf16, bf16, float>& params) {
  GetCPUDevice();
  const auto pth = ne_threading::get();
  if (_cd->AVX512F() && (params.attn_flags & NE_ATTN_FLAG_PREFER_FP32) != 0) {
    using GemmKernelBF16TrackMax = mha::launcher_base_weight_t<  //
        BTLA_ISA::AMX_BF16,                                      //
        gemm::SCoreRowNAvx512f<48, 8>,                           //
        prologue_a::gemm::ActivationBase,                        //
        mha::weight_cvt_bf16_ntile48_t,                          //
        mha::ScaleTrackMaxFp32Fp32>;                             //
    using GemmKernelBF16 = mha::launcher_base_weight_t<          //
        BTLA_ISA::AMX_BF16,                                      //
        gemm::SCoreRowNAvx512f<48, 8>,                           //
        mha::activation_identity_t,                              // pretty sure we have enough paddings for P-matrix
        mha::weight_cvt_bf16_ntile48_t,                          //
        bestla::epilogue::gemm::AccumulatorWriteBackFp32>;       //
    static mha_stable_interface_t<GemmKernelBF16TrackMax, GemmKernelBF16> mha;
    [[maybe_unused]] const auto ret = mha.compute(params, *pth);
    assert(ret == BTLA_CODE::Success);
  } else if (/* params.sl_q > 4 &&  */ _cd->AMX_BF16()) {        // TODO(Yi): add vdpbf16ps impl
    using GemmKernelBF16TrackMax = mha::launcher_base_weight_t<  //
        BTLA_ISA::AMX_BF16,                                      //
        gemm::HCoreRowNAmxbf16<48, 16>,                          //
        prologue_a::gemm::ActivationConverterFp32,               //
        mha::weight_forward_n_tile48_t,                          //
        mha::ScaleTrackMaxFp32Fp32>;                             //
    using GemmKernelBF16 = mha::launcher_base_weight_t<          //
        BTLA_ISA::AMX_BF16,                                      //
        gemm::HCoreRowNAmxbf16<48, 16>,                          //
        mha::activation_identity_t,                              // pretty sure we have enough paddings for P-matrix
        mha::weight_forward_n_tile48_t,                          //
        bestla::epilogue::gemm::AccumulatorWriteBackFp32>;       //
    static mha_stable_interface_t<GemmKernelBF16TrackMax, GemmKernelBF16> mha;
    [[maybe_unused]] const auto ret = mha.compute(params, *pth);
    assert(ret == BTLA_CODE::Success);
  } else {
    assert(0);
  }
}

template <typename Q_T, typename K_T, typename V_T, typename DST_T>
inline void bestla_fusion_attn_forward_ref(const attn_fwd_args_t<Q_T, K_T, V_T, DST_T>& p) {
  const bool is_causal = (p.attn_flags & NE_ATTN_FLAG_IS_CAUSAL) != 0;
  const bool is_alibi = (p.attn_flags & NE_ATTN_FLAG_IS_ALIBI8) != 0;
  const bool prefer_fp32 = (p.attn_flags & NE_ATTN_FLAG_PREFER_FP32) != 0;
  assert(!is_causal || p.sl_q <= p.sl_kv);
  assert(("head_num must be a multiple of heads_kv!", p.head_num % p.heads_kv == 0));
  const auto group_heads = p.head_num / p.heads_kv;
  attn_shape_t attn_shape{
      p.batch_size, p.head_num, p.heads_kv, p.head_size, p.sl_q, p.sl_kv,
  };
  const auto workspace_size = bestla_fusion_attn_workspace_size(&attn_shape);
  static std::mt19937 rng;
  static std::uniform_int_distribution<> dist;
#ifdef NS_TESTS
  init_vector(p.tmp, workspace_size, INT8_MIN - 1, INT8_MAX + 1, dist(rng));
#else
  std::fill_n(p.tmp, workspace_size, 'f');
#endif
  const bool IS_BF16_GEMM =
      !prefer_fp32 &&  //
      ((std::is_same<Q_T, float>::value && std::is_same<K_T, fp16>::value && std::is_same<V_T, fp16>::value &&
        std::is_same<DST_T, float>::value && (!MHA_PREFER_AVX512FP16 || (p.step_k_head_size == 1))) ||
       (std::is_same<Q_T, float>::value && std::is_same<K_T, bf16>::value && std::is_same<V_T, bf16>::value &&
        std::is_same<DST_T, float>::value));
  assert(p.Q_layout == ATTN_FWD_LAYOUT_PLAIN);
  assert(p.dst_layout == ATTN_FWD_LAYOUT_PLAIN);
  assert((p.K_layout == ATTN_FWD_LAYOUT_PLAIN ||
          (std::is_same<K_T, int8_t>::value && p.K_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK4) ||
          (std::is_same<K_T, bf16>::value && p.K_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2) ||
          (std::is_same<K_T, fp16>::value && p.K_layout == ATTN_FWD_LAYOUT_NTILE24_ROWPACK1)));
  assert((p.V_layout == ATTN_FWD_LAYOUT_PLAIN ||
          (std::is_same<V_T, int8_t>::value && p.V_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK4) ||
          (std::is_same<V_T, bf16>::value && p.V_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2) ||
          (std::is_same<V_T, fp16>::value && p.V_layout == ATTN_FWD_LAYOUT_NTILE24_ROWPACK1)));

  const auto NTILE = p.K_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK4   ? 48
                     : p.K_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2 ? 48
                     : p.K_layout == ATTN_FWD_LAYOUT_NTILE24_ROWPACK1 ? 24
                                                                      : 0;
  const auto ROWPACK = p.V_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK4   ? 4
                       : p.V_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2 ? 2
                       : p.V_layout == ATTN_FWD_LAYOUT_NTILE24_ROWPACK1 ? 1
                                                                        : 0;
  // TP will need the real rank order of k
  int32_t k_offset = 0;
  int32_t log_head_num = p.head_num;
#ifdef NS_TP_MODEL
  parallel_context* p_ctx = init_parallel_context();
  int32_t world_size = get_tp_size(p_ctx);
  int32_t rank = get_tp_rank(p_ctx);
  if (world_size > 1) k_offset += rank * p.head_num;
  log_head_num = p.head_num * world_size;
#endif
  const int n_heads_log2_floor = 1 << static_cast<int>(floor(log2(log_head_num)));
  const float m0 = powf(2.0f, -(8.f) / n_heads_log2_floor);
  const float m1 = powf(2.0f, -(8.f / 2.0f) / n_heads_log2_floor);

#pragma omp parallel for collapse(3)
  for (int ibs = 0; ibs < p.batch_size; ++ibs)
    for (int ihn = 0; ihn < p.head_num; ++ihn)
      for (int i = 0; i < p.sl_q; ++i) {
        const auto ihkv = ihn / group_heads;
        const auto q_curr = p.Q + ibs * p.step_q_bs + ihn * p.step_q_head_num + i * p.step_q_sl;
        const auto dst_curr = p.dst + ibs * p.step_dst_bs + ihn * p.step_dst_head_num + i * p.step_dst_sl;

        const auto k_curr = p.K + ibs * p.step_k_bs + ihkv * p.step_k_head_num;
        const auto v_curr = p.V + ibs * p.step_v_bs + ihkv * p.step_v_head_num;

        const auto sl_diff = p.sl_kv - p.sl_q;
        const auto unmasked = is_causal ? sl_diff + i + 1 : p.sl_kv;
        const auto curr_row = std::unique_ptr<float[]>(new float[unmasked]);

        const auto alibi_ihn_m = !is_alibi ? 0.f
                                 : (ihn + k_offset < n_heads_log2_floor)
                                     ? powf(m0, ihn + k_offset + 1)
                                     : powf(m1, 2 * (ihn + k_offset - n_heads_log2_floor) + 1);

        // Q x K
        float row_max = -INFINITY;
        for (int j = 0; j < unmasked; ++j) {
          curr_row[j] = 0.f;
          for (int k = 0; k < p.head_size; ++k) {
            if (p.K_layout != ATTN_FWD_LAYOUT_PLAIN) {
              const auto j_remain = j % NTILE;
              const auto j_block = j - j_remain;
              const auto k_remain = k % ROWPACK;
              const auto k_block = k - k_remain;
              const auto k_value =
                  static_cast<float>(k_curr[j_block * p.step_k_sl + k_block * NTILE + j_remain * ROWPACK + k_remain]);
              const auto q_value =
                  IS_BF16_GEMM ? static_cast<float>(static_cast<bf16>(q_curr[k])) : static_cast<float>(q_curr[k]);
              curr_row[j] += k_value * q_value;
            } else if (IS_BF16_GEMM) {
              curr_row[j] += static_cast<float>(static_cast<bf16>(q_curr[k])) *  // TODO(Yi) fp16 acc
                             static_cast<float>(static_cast<bf16>(k_curr[j * p.step_k_sl + k * p.step_k_head_size]));
            } else {
              curr_row[j] += static_cast<float>(q_curr[k]) *  // TODO(Yi) fp16 acc
                             static_cast<float>(k_curr[j * p.step_k_sl + k * p.step_k_head_size]);
            }
          }
          curr_row[j] = curr_row[j] * p.QK_scale * p.Q_sc * p.K_sc + j * alibi_ihn_m;
          row_max = std::max(row_max, curr_row[j]);
        }

        // exp
        float exp_sum = 0.f;
        for (int j = 0; j < unmasked; ++j) {
          curr_row[j] = mha_exp_ref(curr_row[j] - row_max);
          exp_sum += curr_row[j];
        }

        // softmax
        if (std::is_same<V_T, int8_t>::value) {
          for (int j = 0; j < unmasked; ++j) curr_row[j] = roundf(curr_row[j] * UINT8_MAX) / UINT8_MAX / exp_sum;
        } else {
          for (int j = 0; j < unmasked; ++j) {
            curr_row[j] /= exp_sum;
            if (IS_BF16_GEMM) curr_row[j] = static_cast<float>(static_cast<bf16>(curr_row[j]));
          }
        }

        // P x V
        for (int j = 0; j < p.head_size; ++j) {
          float dst_f32_val = 0.f;
          for (int k = 0; k < unmasked; ++k) {
            if (p.V_layout != ATTN_FWD_LAYOUT_PLAIN) {
              const auto j_remain = j % NTILE;
              const auto j_block = j - j_remain;
              const auto k_remain = k % ROWPACK;
              const auto k_block = k - k_remain;
              const auto v_value = static_cast<float>(
                  v_curr[j_block * p.step_v_head_size + k_block * NTILE + j_remain * ROWPACK + k_remain]);
              dst_f32_val += curr_row[k] * v_value;
            } else if (IS_BF16_GEMM) {
              dst_f32_val += curr_row[k] * static_cast<float>(static_cast<bf16>(v_curr[k * p.step_v_sl + j]));
            } else {
              dst_f32_val += curr_row[k] * static_cast<float>(v_curr[k * p.step_v_sl + j]);
            }
          }
          dst_curr[j] = static_cast<DST_T>(dst_f32_val * p.V_sc / p.dst_sc);
        }
      }
}
}  // namespace mha
}  // namespace custom
}  // namespace ne_bestla
#endif  // NE_CORE_GRAPH_MHA_DENSE_WRAPPER_H
