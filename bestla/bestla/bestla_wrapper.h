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
#include <thread>

#include "bestla_epilogue.h"
#include "bestla_gemm.h"
#include "bestla_prologue_a.h"
#include "bestla_prologue_b.h"
#include "bestla_utils.h"

namespace bestla {
namespace wrapper {

namespace gemm {
template <BTLA_ISA _RT_ISA_T, class _GemmCore_T, template <class _T, BTLA_ISA> class _PrologueA_T,
          template <class _T, BTLA_ISA> class _PrologueB_T, template <BTLA_ISA> class _Epilogue_T>
class LauncherBase {
 public:
  using GemmCore = _GemmCore_T;
  static constexpr BTLA_ISA ISA = _RT_ISA_T;
  using PrologueA = _PrologueA_T<GemmCore, _RT_ISA_T>;
  using PrologueB = _PrologueB_T<GemmCore, _RT_ISA_T>;
  using Epilogue = _Epilogue_T<_RT_ISA_T>;
  using AType = typename GemmCore::AType;
  using AParam = typename PrologueA::Param;
  using BType = typename GemmCore::BType;
  using BParam = typename PrologueB::Param;
  using CType = typename GemmCore::CType;
  using EpiParam = typename Epilogue::Param;
  static_assert(GemmCore::ISA <= _RT_ISA_T, "RunTime ISA should cover GEMM's ISA");
  struct Param {
    const utils::GemmProblem problem;
    const AParam paramA;
    const BParam paramB;
    const EpiParam paramC;
  };
  _GemmCore_T mGemmCore;
  PrologueA mProA;
  PrologueB mProB;
  Epilogue mEpilogue;

  class GEMVWrapper {
   public:
    static constexpr bool support() {
      if constexpr (!std::is_same_v<PrologueB, prologue_b::gemm::WeightKBlockNInteger<_GemmCore_T, _RT_ISA_T>>) {
        return false;
      }
      if constexpr (!std::is_same_v<PrologueA,
                                    prologue_a::gemm::ShuffleActivationKBlockBaseF32<_GemmCore_T, _RT_ISA_T>> &&
                    !std::is_same_v<PrologueA, prologue_a::gemm::ActivationKBlockBaseF32<_GemmCore_T, _RT_ISA_T>> &&
                    !std::is_same_v<PrologueA, prologue_a::gemm::ActivationBase<_GemmCore_T, _RT_ISA_T>>) {
        return false;
      }

      if constexpr (GemmCore::ISA == BTLA_ISA::AVX2) {
        static_assert(GemmCore::PACK_ROW == 1);
        if constexpr (GemmCore::COMP == bestla::gemm::CompType::COMP_FP32) {
          return true;
        }
      }
      return false;
    }
    static int constexpr MaxGemvM = 4;
    static bool implemented(const Param& _param) {
      bool impl = true;
      impl &= _param.paramB.packedW->mDType == BTLA_DTYPE::S4_CLIP ||
              _param.paramB.packedW->mDType == BTLA_DTYPE::S3_CLIP ||
              _param.paramB.packedW->mDType == BTLA_DTYPE::S2_CLIP;
      if constexpr (support()) {
        impl &= _param.paramB.packedW->mCorrection.mScaT == BTLA_DTYPE::F32 ||
                _param.paramB.packedW->mCorrection.mScaT == BTLA_DTYPE::BF16;
      }

      impl &= _param.problem.dims[1] <= MaxGemvM;
      return impl;
    }
    template <typename ScaleT, int MTILE>
    static void gemv_s4(const Param& _param, const parallel::gemm::ThreadProblemBase& _config) {
      int constexpr NBits = 4;
      if constexpr (support()) {
        auto constexpr TmpSize = 3 * 1024LL;
        auto constexpr CSize = 1 * 1024LL;
        auto StackTmp_ = alloca(TmpSize + CSize);
        auto StackTmp = utils::cpu_pointer_align<void>(StackTmp_);
        auto tmpc_ptr = reinterpret_cast<CType*>((char*)StackTmp + TmpSize);
        auto isasym = _param.paramB.packedW->IsAsym();
        auto bzptr = _param.paramB.packedW->template ZPtr<int8_t>();
        int ld_scaleb = _param.paramB.packedW->CStep();
        utils::GemvParamB<ScaleT> paramB{_param.paramB.packedW->template WPtr<uint8_t>(),
                                         nullptr,
                                         nullptr,
                                         _param.paramB.packedW->template SPtr<ScaleT>(),
                                         isasym ? bzptr : nullptr,
                                         NBits,
                                         ld_scaleb};
        const float* Aptr = _param.paramA.A;
        if constexpr (std::is_same_v<PrologueA,
                                     prologue_a::gemm::ShuffleActivationKBlockBaseF32<_GemmCore_T, _RT_ISA_T>>) {
          if (_param.paramA.reordered && _param.paramA.reordered->template APtr<float>()) {
            Aptr = _param.paramA.reordered->template APtr<float>();
          }
        }
        int m = _param.problem.dims[1];
        int n = _param.problem.dims[2];
        int k = _param.problem.dims[3];
        int kblocksize = _param.problem.dims[4];
        auto Cptr = _param.paramC.C + _config.loc[1];
        paramB.b4ptr += _config.loc[1] * _param.paramB.packedW->mKPad / 2;
        paramB.sptr += _config.loc[1];
        if (isasym) {
          paramB.zpptr += _config.loc[1];
        }
        int size_padded = utils::padto_le(_config.size[1], GemmCore::NTILE);
        int in = 0;
        for (; in < size_padded; in += GemmCore::NTILE) {
          if constexpr (std::is_same_v<AType, float>) {
            kernel::wrapper::GEMVWoqNBits::forward_fp32_fp32<_RT_ISA_T, ScaleT, GemmCore::NTILE, MTILE>(
                Aptr, _param.paramA.lda, paramB, Cptr, _param.paramC.ldc, k, kblocksize, StackTmp, TmpSize);
          }

          Cptr += GemmCore::NTILE;
          paramB.b4ptr += GemmCore::NTILE * _param.paramB.packedW->mKPad / 2;
          paramB.sptr += GemmCore::NTILE;
          if (isasym) {
            paramB.zpptr += GemmCore::NTILE;
          }
        }
        if (size_padded != _config.size[1]) {
          if constexpr (std::is_same_v<AType, float>) {
            kernel::wrapper::GEMVWoqNBits::forward_fp32_fp32<_RT_ISA_T, ScaleT, GemmCore::NTILE, MTILE>(
                Aptr, _param.paramA.lda, paramB, tmpc_ptr, GemmCore::NTILE, k, kblocksize, StackTmp, TmpSize);
          }
          for (int i = 0; i < MTILE; i++) {
            memcpy(Cptr + i * _param.paramC.ldc, tmpc_ptr + i * GemmCore::NTILE,
                   (_config.size[1] - in) * sizeof(CType));
          }
        }
        Epilogue::forward(_param.paramC.C + _config.loc[1], 0, 0, _config.loc[1], 1, _config.size[1], _param.paramC,
                          StackTmp, TmpSize);
      }
    }

    template <typename ScaleT, int MTILE>
    static void gemv_s3(const Param& _param, const parallel::gemm::ThreadProblemBase& _config) {
      int constexpr NBits = 3;
      if constexpr (support()) {
        auto constexpr TmpSize = 3 * 1024LL;
        auto constexpr CSize = 1 * 1024LL;
        auto StackTmp_ = alloca(TmpSize + CSize);
        auto StackTmp = utils::cpu_pointer_align<void>(StackTmp_);
        auto tmpc_ptr = reinterpret_cast<CType*>((char*)StackTmp + TmpSize);
        auto isasym = _param.paramB.packedW->IsAsym();
        int m = _param.problem.dims[1];
        int n = _param.problem.dims[2];
        int k = _param.problem.dims[3];
        int kblocksize = _param.problem.dims[4];
        auto Cptr = _param.paramC.C + _config.loc[1];
        auto const KPad = _param.paramB.packedW->mKPad;
        const float* Aptr = _param.paramA.A;
        if constexpr (std::is_same_v<PrologueA,
                                     prologue_a::gemm::ShuffleActivationKBlockBaseF32<_GemmCore_T, _RT_ISA_T>>) {
          if (_param.paramA.reordered && _param.paramA.reordered->template APtr<float>()) {
            Aptr = _param.paramA.reordered->template APtr<float>();
          }
        }
        int size_padded = utils::padto_le(_config.size[1], GemmCore::NTILE);
        int in = 0;
        int ld_scaleb = _param.paramB.packedW->CStep();
        auto bit3_ptr = _param.paramB.packedW->template WPtr<uint8_t>();
        auto bit1_offset = _param.paramB.packedW->mNPad * KPad / 4;
        for (; in < size_padded; in += GemmCore::NTILE) {
          auto elt_offset = (_config.loc[1] + in) * KPad;
          auto bit2ptr = bit3_ptr + elt_offset / 4;
          auto bit1ptr = bit3_ptr + bit1_offset + elt_offset / 8;
          utils::GemvParamB<ScaleT> paramB{
              nullptr,
              bit2ptr,
              bit1ptr,
              _param.paramB.packedW->template SPtr<ScaleT>() + _config.loc[1] + in,
              isasym ? _param.paramB.packedW->template ZPtr<int8_t>() + _config.loc[1] + in : nullptr,
              NBits,
              ld_scaleb};
          kernel::wrapper::GEMVWoqNBits::forward_fp32_fp32<_RT_ISA_T, ScaleT, GemmCore::NTILE, MTILE>(
              Aptr, _param.paramA.lda, paramB, Cptr, _param.paramC.ldc, k, kblocksize, StackTmp, TmpSize);
          Cptr += GemmCore::NTILE;
        }
        if (size_padded != _config.size[1]) {
          auto elt_offset = (_config.loc[1] + in) * KPad;
          auto bit2ptr = bit3_ptr + elt_offset / 4;
          auto bit1ptr = bit3_ptr + bit1_offset + elt_offset / 8;
          utils::GemvParamB<ScaleT> paramB{
              nullptr,
              bit2ptr,
              bit1ptr,
              _param.paramB.packedW->template SPtr<ScaleT>() + _config.loc[1] + in,
              isasym ? _param.paramB.packedW->template ZPtr<int8_t>() + _config.loc[1] + in : nullptr,
              NBits,
              ld_scaleb};
          kernel::wrapper::GEMVWoqNBits::forward_fp32_fp32<_RT_ISA_T, ScaleT, GemmCore::NTILE, MTILE>(
              Aptr, _param.paramA.lda, paramB, tmpc_ptr, GemmCore::NTILE, k, kblocksize, StackTmp, TmpSize);
          for (int i = 0; i < MTILE; i++) {
            memcpy(Cptr + i * _param.paramC.ldc, tmpc_ptr + i * GemmCore::NTILE,
                   (_config.size[1] - in) * sizeof(CType));
          }
        }
        Epilogue::forward(_param.paramC.C + _config.loc[1], 0, 0, _config.loc[1], 1, _config.size[1], _param.paramC,
                          StackTmp, TmpSize);
      }
    }

    template <typename ScaleT, int MTILE>
    static void gemv_s2(const Param& _param, const parallel::gemm::ThreadProblemBase& _config) {
      int constexpr NBits = 2;
      if constexpr (support()) {
        auto constexpr TmpSize = 3 * 1024LL;
        auto constexpr CSize = 1 * 1024LL;
        auto StackTmp_ = alloca(TmpSize + CSize);
        auto StackTmp = utils::cpu_pointer_align<void>(StackTmp_);
        auto tmpc_ptr = reinterpret_cast<CType*>((char*)StackTmp + TmpSize);
        auto isasym = _param.paramB.packedW->IsAsym();
        int m = _param.problem.dims[1];
        int n = _param.problem.dims[2];
        int k = _param.problem.dims[3];
        int kblocksize = _param.problem.dims[4];
        auto Cptr = _param.paramC.C + _config.loc[1];
        auto const KPad = _param.paramB.packedW->mKPad;
        const float* Aptr = _param.paramA.A;
        if constexpr (std::is_same_v<PrologueA,
                                     prologue_a::gemm::ShuffleActivationKBlockBaseF32<_GemmCore_T, _RT_ISA_T>>) {
          if (_param.paramA.reordered && _param.paramA.reordered->template APtr<float>()) {
            Aptr = _param.paramA.reordered->template APtr<float>();
          }
        }
        int size_padded = utils::padto_le(_config.size[1], GemmCore::NTILE);
        int in = 0;
        int ld_scaleb = _param.paramB.packedW->CStep();
        auto bit2_ptr = _param.paramB.packedW->template WPtr<uint8_t>();
        for (; in < size_padded; in += GemmCore::NTILE) {
          auto elt_offset = (_config.loc[1] + in) * KPad;
          auto bit2ptr = bit2_ptr + elt_offset / 4;
          utils::GemvParamB<ScaleT> paramB{
              nullptr,
              bit2ptr,
              nullptr,
              _param.paramB.packedW->template SPtr<ScaleT>() + _config.loc[1] + in,
              isasym ? _param.paramB.packedW->template ZPtr<int8_t>() + _config.loc[1] + in : nullptr,
              NBits,
              ld_scaleb};
          kernel::wrapper::GEMVWoqNBits::forward_fp32_fp32<_RT_ISA_T, ScaleT, GemmCore::NTILE, MTILE>(
              Aptr, _param.paramA.lda, paramB, Cptr, _param.paramC.ldc, k, kblocksize, StackTmp, TmpSize);
          Cptr += GemmCore::NTILE;
        }
        if (size_padded != _config.size[1]) {
          auto elt_offset = (_config.loc[1] + in) * KPad;
          auto bit2ptr = bit2_ptr + elt_offset / 4;
          utils::GemvParamB<ScaleT> paramB{
              nullptr,
              bit2ptr,
              nullptr,
              _param.paramB.packedW->template SPtr<ScaleT>() + _config.loc[1] + in,
              isasym ? _param.paramB.packedW->template ZPtr<int8_t>() + _config.loc[1] + in : nullptr,
              NBits,
              ld_scaleb};
          kernel::wrapper::GEMVWoqNBits::forward_fp32_fp32<_RT_ISA_T, ScaleT, GemmCore::NTILE, MTILE>(
              Aptr, _param.paramA.lda, paramB, tmpc_ptr, GemmCore::NTILE, k, kblocksize, StackTmp, TmpSize);
          for (int i = 0; i < MTILE; i++) {
            memcpy(Cptr + i * _param.paramC.ldc, tmpc_ptr + i * GemmCore::NTILE,
                   (_config.size[1] - in) * sizeof(CType));
          }
        }
        Epilogue::forward(_param.paramC.C + _config.loc[1], 0, 0, _config.loc[1], 1, _config.size[1], _param.paramC,
                          StackTmp, TmpSize);
      }
    }

    static void gemv(const Param& _param, const parallel::gemm::ThreadProblemBase& _config) {
      if constexpr (support()) {
        assert(_param.problem.dims[4] > 0);
        auto& m = _param.problem.dims[1];
        if (_param.paramB.packedW->mDType == BTLA_DTYPE::S4_CLIP) {
          if (_param.paramB.packedW->SDtype() == BTLA_DTYPE::F32) {
            if (m == 1) gemv_s4<float, 1>(_param, _config);
            if (m == 2) gemv_s4<float, 2>(_param, _config);
            if (m == 3) gemv_s4<float, 3>(_param, _config);
            if (m == 4) gemv_s4<float, 4>(_param, _config);

          } else if (_param.paramB.packedW->SDtype() == BTLA_DTYPE::BF16) {
            if (m == 1) gemv_s4<utils::bf16, 1>(_param, _config);
            if (m == 2) gemv_s4<utils::bf16, 2>(_param, _config);
            if (m == 3) gemv_s4<utils::bf16, 3>(_param, _config);
            if (m == 4) gemv_s4<utils::bf16, 4>(_param, _config);
          }
          return;
        }
        if (_param.paramB.packedW->mDType == BTLA_DTYPE::S3_CLIP) {
          if (_param.paramB.packedW->SDtype() == BTLA_DTYPE::F32) {
            if (m == 1) gemv_s3<float, 1>(_param, _config);
            if (m == 2) gemv_s3<float, 2>(_param, _config);
            if (m == 3) gemv_s3<float, 3>(_param, _config);
            if (m == 4) gemv_s3<float, 4>(_param, _config);

          } else if (_param.paramB.packedW->SDtype() == BTLA_DTYPE::BF16) {
            if (m == 1) gemv_s3<utils::bf16, 1>(_param, _config);
            if (m == 2) gemv_s3<utils::bf16, 2>(_param, _config);
            if (m == 3) gemv_s3<utils::bf16, 3>(_param, _config);
            if (m == 4) gemv_s3<utils::bf16, 4>(_param, _config);
          }
          return;
        }
        if (_param.paramB.packedW->mDType == BTLA_DTYPE::S2_CLIP) {
          if (_param.paramB.packedW->SDtype() == BTLA_DTYPE::F32) {
            if (m == 1) gemv_s2<float, 1>(_param, _config);
            if (m == 2) gemv_s2<float, 2>(_param, _config);
            if (m == 3) gemv_s2<float, 3>(_param, _config);
            if (m == 4) gemv_s2<float, 4>(_param, _config);

          } else if (_param.paramB.packedW->SDtype() == BTLA_DTYPE::BF16) {
            if (m == 1) gemv_s2<utils::bf16, 1>(_param, _config);
            if (m == 2) gemv_s2<utils::bf16, 2>(_param, _config);
            if (m == 3) gemv_s2<utils::bf16, 3>(_param, _config);
            if (m == 4) gemv_s2<utils::bf16, 4>(_param, _config);
          }
          return;
        }
      }
    }
  };

  void run(const Param& _param, const parallel::gemm::ThreadProblemBase& _config) {
    if (GEMVWrapper::support() && GEMVWrapper::implemented(_param)) {
      GEMVWrapper::gemv(_param, _config);
    } else {
      gemm(_param, _config);
    }
  }

 protected:
  void gemm(const Param& _param, const parallel::gemm::ThreadProblemBase& _config) {
    mGemmCore.configure(_config.size[0], _config.size[1], _param.problem.dims[3]);
    auto StackTmp = alloca(_config.stacksize);
    auto tmpB = reinterpret_cast<BType*>(StackTmp);
    tmpB = utils::cpu_pointer_align(tmpB);
    auto tmpA = reinterpret_cast<AType*>(tmpB + static_cast<size_t>(_config.block[1]) * _config.block[2]);
    tmpA = utils::cpu_pointer_align(tmpA);
    auto tmpC = reinterpret_cast<CType*>(tmpA + static_cast<size_t>(GemmCore::MTILE) * _config.block[2]);
    tmpC = utils::cpu_pointer_align(tmpC);
    auto tmpCache = (void*)(tmpC + static_cast<size_t>(_config.block[0]) * _config.block[1]);
    tmpCache = utils::cpu_pointer_align(tmpCache);
    for (int itern = 0; itern < _config.size[1]; itern += _config.block[1]) {
      int n_remain = utils::remainsize(itern, _config.size[1], _config.block[1]);
      for (int iterm = 0; iterm < _config.size[0]; iterm += _config.block[0]) {
        int m_remain = utils::remainsize(iterm, _config.size[0], _config.block[0]);
        run_block(_param, _config, iterm, itern, m_remain, n_remain, tmpA, tmpB, tmpC, tmpCache);
      }
    }
  }

  void run_block(const Param& _param, const parallel::gemm::ThreadProblemBase& _config, int blk_m, int blk_n,
                 int blk_msize, int blk_nsize, AType* tmpA, BType* tmpB, CType* tmpC, void* tmpcache) {
    int n_padded = utils::padto(blk_nsize, GemmCore::NTILE);
    auto& K = _param.problem.dims[3];
    for (int iterk = 0; iterk < _param.problem.dims[3]; iterk += _config.block[2]) {
      int k_remain = utils::remainsize(iterk, K, _config.block[2]);
      int k_padded = utils::padto(k_remain, GemmCore::KTILE);
      int k_paddedle = utils::padto_le(k_remain, GemmCore::KTILE);
      auto bptr_cache = tmpB;
      int bcache_step = 0;
      mProB.getWeight(&bptr_cache, &bcache_step, k_padded, n_padded, iterk, _config.loc[1] + blk_n, _param.paramB,
                      tmpcache, _config.tmpcachesize);
      int bcache_stride = bcache_step * sizeof(BType);
      for (int i = 0; i < blk_msize; i += GemmCore::MTILE) {
        int m_remain = utils::remainsize(i, blk_msize, GemmCore::MTILE);
        auto cptr_cache = tmpC + i * _config.block[1];
        int ccache_stride = _config.block[1] * sizeof(CType);
        if (k_paddedle) {
          AType* aptr_cache = tmpA;
          int acache_step = 0;
          mProA.getActivation(&aptr_cache, &acache_step, _param.paramA, m_remain, k_paddedle,
                              (blk_m + i + _config.loc[0]), iterk, tmpcache, _config.tmpcachesize);
          mGemmCore.forward(aptr_cache, bptr_cache, cptr_cache, m_remain, n_padded, k_paddedle,
                            acache_step * sizeof(AType), bcache_stride, ccache_stride, iterk, tmpcache,
                            _config.tmpcachesize);
        }
        int k_tail = k_remain - k_paddedle;
        if (k_tail) {
          AType* aptr_cache = tmpA;
          int acache_step = 0;
          mProA.getActivation(&aptr_cache, &acache_step, _param.paramA, m_remain, k_tail, (blk_m + i + _config.loc[0]),
                              iterk + k_paddedle, tmpcache, _config.tmpcachesize);
          mGemmCore.forward(aptr_cache, bptr_cache + k_paddedle * GemmCore::NTILE, cptr_cache, m_remain, n_padded,
                            GemmCore::KTILE, acache_step * sizeof(AType), bcache_stride, ccache_stride,
                            iterk + k_paddedle, tmpcache, _config.tmpcachesize);
        }
      }
    }
    mEpilogue.forward(tmpC, _config.block[1], (_config.loc[0] + blk_m), _config.loc[1] + blk_n, blk_msize, blk_nsize,
                      _param.paramC, tmpcache, _config.tmpcachesize);
  }
};

template <BTLA_ISA _RT_ISA_T, class _GemmCore_T, template <class _T, BTLA_ISA> class _PrologueA_T,
          template <class _T, BTLA_ISA> class _PrologueB_T, template <BTLA_ISA> class _Epilogue_T>
class LauncherIntKBlock {
 public:
  using GemmCore = _GemmCore_T;
  static constexpr BTLA_ISA ISA = _RT_ISA_T;
  using PrologueA = _PrologueA_T<GemmCore, _RT_ISA_T>;
  using PrologueB = _PrologueB_T<GemmCore, _RT_ISA_T>;
  using Epilogue = _Epilogue_T<_RT_ISA_T>;
  using AType = typename GemmCore::AType;
  using AParam = typename PrologueA::Param;
  using BType = typename GemmCore::BType;
  using BParam = typename PrologueB::Param;
  using CType = typename GemmCore::CType;
  using EpiParam = typename Epilogue::Param;
  using AccType = float;
  static_assert(GemmCore::ISA <= _RT_ISA_T, "RunTime ISA should cover GEMM's ISA");
  struct Param {
    const utils::GemmProblem problem;
    const AParam paramA;
    const BParam paramB;
    const EpiParam paramC;
  };
  _GemmCore_T mGemmCore;
  PrologueA mProA;
  PrologueB mProB;
  Epilogue mEpilogue;

  class GEMVWrapper {
   public:
    static constexpr bool support() {
      if constexpr (!std::is_same_v<PrologueB, prologue_b::gemm::WeightKBlockNInteger<_GemmCore_T, _RT_ISA_T>>) {
        return false;
      }
      if constexpr (!std::is_same_v<PrologueA, prologue_a::gemm::ActivationF32KBlockQuantize<_GemmCore_T, _RT_ISA_T>> &&
                    !std::is_same_v<PrologueA,
                                    prologue_a::gemm::ShuffleActivationKBlockQuantizeF32<_GemmCore_T, _RT_ISA_T>>) {
        return false;
      }
      if constexpr (GemmCore::ISA == BTLA_ISA::AVX_VNNI) {
        static_assert(GemmCore::PACK_ROW == 4);
        if constexpr (GemmCore::COMP == bestla::gemm::CompType::COMP_INT8_US_FP32) {
          return true;
        }
        if constexpr (GemmCore::COMP == bestla::gemm::CompType::COMP_INT8_SS_FP32) {
          return true;
        }
      }
      return false;
    }
    static int constexpr MaxGemvM = 4;

    static bool implemented(const Param& _param) {
      bool impl = true;
      impl &= _param.paramB.packedW->mDType == BTLA_DTYPE::S4_CLIP ||
              _param.paramB.packedW->mDType == BTLA_DTYPE::S3_CLIP ||
              _param.paramB.packedW->mDType == BTLA_DTYPE::S2_CLIP;
      impl &= _param.paramB.packedW->mCorrection.mScaT == BTLA_DTYPE::F32 ||
              _param.paramB.packedW->mCorrection.mScaT == BTLA_DTYPE::BF16;
      impl &= _param.problem.dims[1] <= MaxGemvM;
      return impl;
    }
    template <typename ScaleT, int MTILE>
    static void gemv_s4(const Param& _param, const parallel::gemm::ThreadProblemBase& _config) {
      int constexpr NBits = 4;
      if constexpr (support()) {
        auto constexpr TmpSize = 3 * 1024LL;
        auto constexpr CSize = 1 * 1024LL;
        auto StackTmp_ = alloca(TmpSize + CSize);
        auto StackTmp = utils::cpu_pointer_align<void>(StackTmp_);
        auto tmpc_ptr = reinterpret_cast<CType*>((char*)StackTmp + TmpSize);
        auto isasym = _param.paramB.packedW->IsAsym();
        auto bzptr = _param.paramB.packedW->template ZPtr<int8_t>();
        utils::GemvParamB<ScaleT> paramB{_param.paramB.packedW->template WPtr<uint8_t>(),
                                         nullptr,
                                         nullptr,
                                         _param.paramB.packedW->template SPtr<ScaleT>(),
                                         isasym ? bzptr : nullptr,
                                         NBits,
                                         _param.paramB.packedW->CStep()};
        utils::GemvParamA paramA{
            _param.paramA.quan->template APtr<uint8_t>(), _param.paramA.quan->template SPtr<float>(),
            _param.paramA.quan->template ZPtr<uint8_t>(), _param.paramA.quan->mKPad, _param.paramA.quan->CStep()};
        int m = _param.problem.dims[1];
        int n = _param.problem.dims[2];
        int k = _param.problem.dims[3];
        int kblocksize = _param.problem.dims[4];
        auto Cptr = _param.paramC.C + _config.loc[1];
        paramB.b4ptr += _config.loc[1] * _param.paramB.packedW->mKPad / 2;
        paramB.sptr += _config.loc[1];
        if (isasym) {
          paramB.zpptr += _config.loc[1];
        }
        int size_padded = utils::padto_le(_config.size[1], GemmCore::NTILE);
        int in = 0;
        int ld_scaleb = _param.paramB.packedW->CStep();
        for (; in < size_padded; in += GemmCore::NTILE) {
          if constexpr (std::is_same_v<AType, uint8_t>) {
            kernel::wrapper::GEMVWoqNBits::forward_u8s8_fp32<_RT_ISA_T, ScaleT, GemmCore::NTILE, MTILE>(
                paramA, paramB, Cptr, _param.paramC.ldc, k, kblocksize, StackTmp, TmpSize);
          } else {
            kernel::wrapper::GEMVWoqNBits::forward_s8s8_fp32<_RT_ISA_T, ScaleT, GemmCore::NTILE, MTILE>(
                paramA, paramB, Cptr, _param.paramC.ldc, k, kblocksize, StackTmp, TmpSize);
          }

          Cptr += GemmCore::NTILE;
          paramB.b4ptr += GemmCore::NTILE * _param.paramB.packedW->mKPad / 2;
          paramB.sptr += GemmCore::NTILE;
          if (isasym) {
            paramB.zpptr += GemmCore::NTILE;
          }
        }
        if (size_padded != _config.size[1]) {
          if constexpr (std::is_same_v<AType, uint8_t>) {
            kernel::wrapper::GEMVWoqNBits::forward_u8s8_fp32<_RT_ISA_T, ScaleT, GemmCore::NTILE, MTILE>(
                paramA, paramB, tmpc_ptr, GemmCore::NTILE, k, kblocksize, StackTmp, TmpSize);
          } else {
            kernel::wrapper::GEMVWoqNBits::forward_s8s8_fp32<_RT_ISA_T, ScaleT, GemmCore::NTILE, MTILE>(
                paramA, paramB, tmpc_ptr, GemmCore::NTILE, k, kblocksize, StackTmp, TmpSize);
          }
          for (int im = 0; im < MTILE; im++) {
            memcpy(Cptr + im * _param.paramC.ldc, tmpc_ptr + im * GemmCore::NTILE,
                   (_config.size[1] - in) * sizeof(CType));
          }
        }
        Epilogue::forward(_param.paramC.C + _config.loc[1], 0, 0, _config.loc[1], 1, _config.size[1], _param.paramC,
                          StackTmp, TmpSize);
      }
    }

    template <typename ScaleT, int MTILE>
    static void gemv_s3(const Param& _param, const parallel::gemm::ThreadProblemBase& _config) {
      int constexpr NBits = 3;
      if constexpr (support()) {
        auto constexpr TmpSize = 3 * 1024LL;
        auto constexpr CSize = 1 * 1024LL;
        auto StackTmp_ = alloca(TmpSize + CSize);
        auto StackTmp = utils::cpu_pointer_align<void>(StackTmp_);
        auto tmpc_ptr = reinterpret_cast<CType*>((char*)StackTmp + TmpSize);
        utils::GemvParamA paramA{
            _param.paramA.quan->template APtr<uint8_t>(), _param.paramA.quan->template SPtr<float>(),
            _param.paramA.quan->template ZPtr<uint8_t>(), _param.paramA.quan->mKPad, _param.paramA.quan->CStep()};
        int m = _param.problem.dims[1];
        int n = _param.problem.dims[2];
        int k = _param.problem.dims[3];
        int kblocksize = _param.problem.dims[4];
        auto Cptr = _param.paramC.C + _config.loc[1];
        auto const KPad = _param.paramB.packedW->mKPad;

        int size_padded = utils::padto_le(_config.size[1], GemmCore::NTILE);
        int in = 0;
        int ld_scaleb = _param.paramB.packedW->CStep();
        auto bit3_ptr = _param.paramB.packedW->template WPtr<uint8_t>();
        auto isasym = _param.paramB.packedW->IsAsym();
        auto bzptr = _param.paramB.packedW->template ZPtr<int8_t>();
        auto bit1_offset = _param.paramB.packedW->mNPad * KPad / 4;
        for (; in < size_padded; in += GemmCore::NTILE) {
          auto elt_offset = (_config.loc[1] + in) * KPad;
          auto bit2ptr = bit3_ptr + elt_offset / 4;
          auto bit1ptr = bit3_ptr + bit1_offset + elt_offset / 8;
          utils::GemvParamB<ScaleT> paramB{nullptr,
                                           bit2ptr,
                                           bit1ptr,
                                           _param.paramB.packedW->template SPtr<ScaleT>() + _config.loc[1] + in,
                                           isasym ? bzptr + _config.loc[1] + in : nullptr,
                                           NBits,
                                           ld_scaleb};
          if constexpr (std::is_same_v<AType, uint8_t>) {
            kernel::wrapper::GEMVWoqNBits::forward_u8s8_fp32<_RT_ISA_T, ScaleT, GemmCore::NTILE, MTILE>(
                paramA, paramB, Cptr, _param.paramC.ldc, k, kblocksize, StackTmp, TmpSize);
          } else {
            kernel::wrapper::GEMVWoqNBits::forward_s8s8_fp32<_RT_ISA_T, ScaleT, GemmCore::NTILE, MTILE>(
                paramA, paramB, Cptr, _param.paramC.ldc, k, kblocksize, StackTmp, TmpSize);
          }
          Cptr += GemmCore::NTILE;
        }
        if (size_padded != _config.size[1]) {
          auto elt_offset = (_config.loc[1] + in) * KPad;
          auto bit2ptr = bit3_ptr + elt_offset / 4;
          auto bit1ptr = bit3_ptr + bit1_offset + elt_offset / 8;
          utils::GemvParamB<ScaleT> paramB{nullptr,
                                           bit2ptr,
                                           bit1ptr,
                                           _param.paramB.packedW->template SPtr<ScaleT>() + _config.loc[1] + in,
                                           isasym ? bzptr + _config.loc[1] + in : nullptr,
                                           NBits,
                                           ld_scaleb};
          if constexpr (std::is_same_v<AType, uint8_t>) {
            kernel::wrapper::GEMVWoqNBits::forward_u8s8_fp32<_RT_ISA_T, ScaleT, GemmCore::NTILE, MTILE>(
                paramA, paramB, tmpc_ptr, GemmCore::NTILE, k, kblocksize, StackTmp, TmpSize);
          } else {
            kernel::wrapper::GEMVWoqNBits::forward_s8s8_fp32<_RT_ISA_T, ScaleT, GemmCore::NTILE, MTILE>(
                paramA, paramB, tmpc_ptr, GemmCore::NTILE, k, kblocksize, StackTmp, TmpSize);
          }
          for (int im = 0; im < MTILE; im++) {
            memcpy(Cptr + im * _param.paramC.ldc, tmpc_ptr + im * GemmCore::NTILE,
                   (_config.size[1] - in) * sizeof(CType));
          }
        }
        Epilogue::forward(_param.paramC.C + _config.loc[1], 0, 0, _config.loc[1], 1, _config.size[1], _param.paramC,
                          StackTmp, TmpSize);
      }
    }

    template <typename ScaleT, int MTILE>
    static void gemv_s2(const Param& _param, const parallel::gemm::ThreadProblemBase& _config) {
      int constexpr NBits = 2;
      if constexpr (support()) {
        auto constexpr TmpSize = 3 * 1024LL;
        auto constexpr CSize = 1 * 1024LL;
        auto StackTmp_ = alloca(TmpSize + CSize);
        auto StackTmp = utils::cpu_pointer_align<void>(StackTmp_);
        auto tmpc_ptr = reinterpret_cast<CType*>((char*)StackTmp + TmpSize);
        utils::GemvParamA paramA{
            _param.paramA.quan->template APtr<uint8_t>(), _param.paramA.quan->template SPtr<float>(),
            _param.paramA.quan->template ZPtr<uint8_t>(), _param.paramA.quan->mKPad, _param.paramA.quan->CStep()};
        int m = _param.problem.dims[1];
        int n = _param.problem.dims[2];
        int k = _param.problem.dims[3];
        int kblocksize = _param.problem.dims[4];
        auto Cptr = _param.paramC.C + _config.loc[1];
        auto const KPad = _param.paramB.packedW->mKPad;

        int size_padded = utils::padto_le(_config.size[1], GemmCore::NTILE);
        int in = 0;
        int ld_scaleb = _param.paramB.packedW->CStep();
        auto bit2_ptr = _param.paramB.packedW->template WPtr<uint8_t>() + _config.loc[1] * KPad / 4;
        auto is_asym = _param.paramB.packedW->IsAsym();
        for (; in < size_padded; in += GemmCore::NTILE) {
          utils::GemvParamB<ScaleT> paramB{
              nullptr,
              bit2_ptr + in * KPad / 4,
              nullptr,
              _param.paramB.packedW->template SPtr<ScaleT>() + _config.loc[1] + in,
              is_asym ? _param.paramB.packedW->template ZPtr<int8_t>() + _config.loc[1] + in : nullptr,
              NBits,
              ld_scaleb};
          if constexpr (std::is_same_v<AType, uint8_t>) {
            kernel::wrapper::GEMVWoqNBits::forward_u8s8_fp32<_RT_ISA_T, ScaleT, GemmCore::NTILE, MTILE>(
                paramA, paramB, Cptr, _param.paramC.ldc, k, kblocksize, StackTmp, TmpSize);
          } else {
            kernel::wrapper::GEMVWoqNBits::forward_s8s8_fp32<_RT_ISA_T, ScaleT, GemmCore::NTILE, MTILE>(
                paramA, paramB, Cptr, _param.paramC.ldc, k, kblocksize, StackTmp, TmpSize);
          }
          Cptr += GemmCore::NTILE;
        }
        if (size_padded != _config.size[1]) {
          utils::GemvParamB<ScaleT> paramB{
              nullptr,
              bit2_ptr + in * KPad / 4,
              nullptr,
              _param.paramB.packedW->template SPtr<ScaleT>() + _config.loc[1] + in,
              is_asym ? _param.paramB.packedW->template ZPtr<int8_t>() + _config.loc[1] + in : nullptr,
              NBits,
              ld_scaleb};
          if constexpr (std::is_same_v<AType, uint8_t>) {
            kernel::wrapper::GEMVWoqNBits::forward_u8s8_fp32<_RT_ISA_T, ScaleT, GemmCore::NTILE, MTILE>(
                paramA, paramB, tmpc_ptr, GemmCore::NTILE, k, kblocksize, StackTmp, TmpSize);
          } else {
            kernel::wrapper::GEMVWoqNBits::forward_s8s8_fp32<_RT_ISA_T, ScaleT, GemmCore::NTILE, MTILE>(
                paramA, paramB, tmpc_ptr, GemmCore::NTILE, k, kblocksize, StackTmp, TmpSize);
          }
          for (int im = 0; im < MTILE; im++) {
            memcpy(Cptr + im * _param.paramC.ldc, tmpc_ptr + im * GemmCore::NTILE,
                   (_config.size[1] - in) * sizeof(CType));
          }
        }
        Epilogue::forward(_param.paramC.C + _config.loc[1], 0, 0, _config.loc[1], 1, _config.size[1], _param.paramC,
                          StackTmp, TmpSize);
      }
    }

    static void gemv(const Param& _param, const parallel::gemm::ThreadProblemBase& _config) {
      if constexpr (support()) {
        auto& m = _param.problem.dims[1];
        if (_param.paramB.packedW->mDType == BTLA_DTYPE::S4_CLIP) {
          if (_param.paramB.packedW->SDtype() == BTLA_DTYPE::F32) {
            if (m == 1) gemv_s4<float, 1>(_param, _config);
            if (m == 2) gemv_s4<float, 2>(_param, _config);
            if (m == 3) gemv_s4<float, 3>(_param, _config);
            if (m == 4) gemv_s4<float, 4>(_param, _config);

          } else if (_param.paramB.packedW->SDtype() == BTLA_DTYPE::BF16) {
            if (m == 1) gemv_s4<utils::bf16, 1>(_param, _config);
            if (m == 2) gemv_s4<utils::bf16, 2>(_param, _config);
            if (m == 3) gemv_s4<utils::bf16, 3>(_param, _config);
            if (m == 4) gemv_s4<utils::bf16, 4>(_param, _config);
          }
          return;
        }

        if (_param.paramB.packedW->mDType == BTLA_DTYPE::S3_CLIP) {
          if (_param.paramB.packedW->SDtype() == BTLA_DTYPE::F32) {
            if (m == 1) gemv_s3<float, 1>(_param, _config);
            if (m == 2) gemv_s3<float, 2>(_param, _config);
            if (m == 3) gemv_s3<float, 3>(_param, _config);
            if (m == 4) gemv_s3<float, 4>(_param, _config);
          } else if (_param.paramB.packedW->SDtype() == BTLA_DTYPE::BF16) {
            if (m == 1) gemv_s3<utils::bf16, 1>(_param, _config);
            if (m == 2) gemv_s3<utils::bf16, 2>(_param, _config);
            if (m == 3) gemv_s3<utils::bf16, 3>(_param, _config);
            if (m == 4) gemv_s3<utils::bf16, 4>(_param, _config);
          }
          return;
        }
        if (_param.paramB.packedW->mDType == BTLA_DTYPE::S2_CLIP) {
          if (_param.paramB.packedW->SDtype() == BTLA_DTYPE::F32) {
            if (m == 1) gemv_s2<float, 1>(_param, _config);
            if (m == 2) gemv_s2<float, 2>(_param, _config);
            if (m == 3) gemv_s2<float, 3>(_param, _config);
            if (m == 4) gemv_s2<float, 4>(_param, _config);
          } else if (_param.paramB.packedW->SDtype() == BTLA_DTYPE::BF16) {
            if (m == 1) gemv_s2<utils::bf16, 1>(_param, _config);
            if (m == 2) gemv_s2<utils::bf16, 2>(_param, _config);
            if (m == 3) gemv_s2<utils::bf16, 3>(_param, _config);
            if (m == 4) gemv_s2<utils::bf16, 4>(_param, _config);
          }
          return;
        }
      }
    }
  };

  void run(const Param& _param, const parallel::gemm::ThreadProblemBase& _config) {
    if (GEMVWrapper::support() && GEMVWrapper::implemented(_param)) {
      GEMVWrapper::gemv(_param, _config);
    } else {
      gemm(_param, _config);
    }
  }

 protected:
  void gemm(const Param& _param, const parallel::gemm::ThreadProblemBase& _config) {
    mGemmCore.configure(_config.size[0], _config.size[1], _param.problem.dims[3]);
    auto StackTmp = alloca(_config.stacksize);
    auto tmpB = reinterpret_cast<BType*>(StackTmp);
    tmpB = utils::cpu_pointer_align(tmpB);
    auto tmpA = reinterpret_cast<AType*>(tmpB + static_cast<size_t>(_config.block[1]) * _config.block[2]);
    tmpA = utils::cpu_pointer_align(tmpA);
    auto tmpC = reinterpret_cast<AccType*>(tmpA + static_cast<size_t>(GemmCore::MTILE) * _config.block[2]);
    tmpC = utils::cpu_pointer_align(tmpC);
    auto tmpCache = reinterpret_cast<int8_t*>(tmpC + static_cast<size_t>(_config.block[0]) * _config.block[1]);
    tmpCache = utils::cpu_pointer_align(tmpCache);
    for (int itern = 0; itern < _config.size[1]; itern += _config.block[1]) {
      int n_remain = utils::remainsize(itern, _config.size[1], _config.block[1]);
      for (int iterm = 0; iterm < _config.size[0]; iterm += _config.block[0]) {
        int m_remain = utils::remainsize(iterm, _config.size[0], _config.block[0]);
        auto& KBlock = _param.problem.dims[4];
        if (_config.block[2] >= KBlock) {
          run_block(_param, _config, iterm, itern, m_remain, n_remain, tmpA, tmpB, tmpC, tmpCache);
        } else {
          run_largekblock(_param, _config, iterm, itern, m_remain, n_remain, tmpA, tmpB, tmpC, tmpCache);
        }
      }
    }
  }

  // _config.block[2]%kblock==0
  // _config.block[2]>=kblock
  void run_block(const Param& _param, const parallel::gemm::ThreadProblemBase& _config, int blk_m, int blk_n,
                 int blk_msize, int blk_nsize, AType* tmpA, BType* tmpB, AccType* tmpC, int8_t* tmpcache) {
    int n_padded = utils::padto(blk_nsize, GemmCore::NTILE);
    auto& K = _param.problem.dims[3];
    auto& KBlock = _param.problem.dims[4];
    assert(_config.block[2] % KBlock == 0);
    assert(_config.block[2] % GemmCore::KTILE == 0);
    // GemmCore: int8+int8=int32=>dequant to fp32
    // accumulate to tmpC
    // extra parameters: zpA, scaleA, scaleB, reduceB
    // zpA scaleA: [MTILE,kblk_perstep]
    // scaleB reduceB: [kblk_perstep, NStep]
    int kblk_perstep = utils::updiv(_config.block[2], KBlock);
    int tmp_ldsb = _config.block[1];
    int tmp_ldsa = kblk_perstep;
    auto zpA = reinterpret_cast<AType*>(tmpcache);
    zpA = utils::cpu_pointer_align(zpA);
    auto scaleA = reinterpret_cast<float*>(zpA + GemmCore::MTILE * tmp_ldsa);
    scaleA = utils::cpu_pointer_align(scaleA);
    auto scaleB = reinterpret_cast<float*>(scaleA + GemmCore::MTILE * tmp_ldsa);
    scaleB = utils::cpu_pointer_align(scaleB);
    auto reduceB = reinterpret_cast<float*>(scaleB + _config.block[1] * tmp_ldsa);
    reduceB = utils::cpu_pointer_align(reduceB);
    auto tmp_ = reinterpret_cast<int8_t*>(reduceB + _config.block[1] * tmp_ldsa);
    tmp_ = utils::cpu_pointer_align(tmp_);

    for (int iterk = 0; iterk < K; iterk += _config.block[2]) {
      int k_remain = utils::remainsize(iterk, K, _config.block[2]);
      int k_padded = utils::padto(k_remain, GemmCore::KTILE);
      auto bptr_cache = tmpB;
      int bcache_step = 0;
      int ldsb_cache = tmp_ldsb;
      auto scaleB_cache = scaleB;
      auto reduceB_cache = reduceB;
      mProB.getWeight(&bptr_cache, &bcache_step, k_padded, n_padded, iterk, _config.loc[1] + blk_n, _param.paramB, tmp_,
                      _config.tmpcachesize);
      mProB.getScale(&scaleB_cache, &ldsb_cache, k_padded, n_padded, iterk, _config.loc[1] + blk_n, _param.paramB, tmp_,
                     _config.tmpcachesize);
      mProB.getReduce(&reduceB_cache, &ldsb_cache, k_padded, n_padded, iterk, _config.loc[1] + blk_n, _param.paramB,
                      tmp_, _config.tmpcachesize);
      int bcache_stride = bcache_step * sizeof(BType);
      for (int i = 0; i < blk_msize; i += GemmCore::MTILE) {
        int m_remain = utils::remainsize(i, blk_msize, GemmCore::MTILE);
        auto cptr_cache = tmpC + i * _config.block[1];
        int ccache_stride = _config.block[1] * sizeof(CType);
        auto aptr_cache = tmpA;
        int acache_step = k_padded;
        auto zpA_cache = zpA;
        auto scaleA_cache = scaleA;
        int ldsa_cache = tmp_ldsa;
        mProA.getActivation(&aptr_cache, &acache_step, _param.paramA, m_remain, k_padded, (blk_m + i + _config.loc[0]),
                            iterk, tmp_, _config.tmpcachesize);
        mProA.getZp(&zpA_cache, &ldsa_cache, _param.paramA, m_remain, k_padded, (blk_m + i + _config.loc[0]), iterk,
                    tmp_, _config.tmpcachesize);
        mProA.getScale(&scaleA_cache, &ldsa_cache, _param.paramA, m_remain, k_padded, (blk_m + i + _config.loc[0]),
                       iterk, tmp_, _config.tmpcachesize);
        mGemmCore.forward(aptr_cache, bptr_cache, cptr_cache, zpA_cache, scaleA_cache, ldsa_cache, scaleB_cache,
                          reduceB_cache, ldsb_cache, m_remain, n_padded, k_padded, KBlock, acache_step * sizeof(AType),
                          bcache_stride, ccache_stride, iterk, 1.f, tmp_, _config.tmpcachesize);
      }
    }
    mEpilogue.forward(tmpC, _config.block[1], (_config.loc[0] + blk_m), _config.loc[1] + blk_n, blk_msize, blk_nsize,
                      _param.paramC, tmpcache, _config.tmpcachesize);
  }

  // _config.block[2]<kblock
  void run_largekblock(const Param& _param, const parallel::gemm::ThreadProblemBase& _config, int blk_m, int blk_n,
                       int blk_msize, int blk_nsize, AType* tmpA, BType* tmpB, AccType* tmpC, int8_t* tmpcache) {
    int n_padded = utils::padto(blk_nsize, GemmCore::NTILE);
    auto& K = _param.problem.dims[3];
    auto& KBlock = _param.problem.dims[4];
    // GemmCore: int8+int8=int32=>dequant to fp32
    // accumulate to tmpC
    // extra parameters: zpA, scaleA, scaleB, reduceB
    // zpA scaleA: [MTILE,kblk_perstep]
    // scaleB reduceB: [kblk_perstep, NStep]
    int kblk_perstep = 1;
    int tmp_ldsb = _config.block[1];
    int tmp_ldsa = kblk_perstep;
    auto zpA = reinterpret_cast<AType*>(tmpcache);
    zpA = utils::cpu_pointer_align(zpA);
    auto scaleA = reinterpret_cast<float*>(zpA + GemmCore::MTILE * tmp_ldsa);
    scaleA = utils::cpu_pointer_align(scaleA);
    auto scaleB = reinterpret_cast<float*>(scaleA + GemmCore::MTILE * tmp_ldsa);
    scaleB = utils::cpu_pointer_align(scaleB);
    auto reduceB = reinterpret_cast<float*>(scaleB + _config.block[1] * tmp_ldsa);
    reduceB = utils::cpu_pointer_align(reduceB);
    auto tmp_ = reinterpret_cast<int8_t*>(reduceB + _config.block[1] * tmp_ldsa);
    tmp_ = utils::cpu_pointer_align(tmp_);

    for (int iterk = 0; iterk < K; iterk += KBlock) {
      for (int iterkk = iterk; iterkk < iterk + KBlock; iterkk += _config.block[2]) {
        int k_remain = utils::remainsize(iterkk, K, _config.block[2]);
        k_remain = utils::remainsize(iterkk, iterk + KBlock, _config.block[2]);
        int k_padded = utils::padto(k_remain, GemmCore::KTILE);
        auto bptr_cache = tmpB;
        int bcache_step = 0;
        int ldsb_cache = tmp_ldsb;
        auto scaleB_cache = scaleB;
        auto reduceB_cache = reduceB;
        mProB.getWeight(&bptr_cache, &bcache_step, k_padded, n_padded, iterkk, _config.loc[1] + blk_n, _param.paramB,
                        tmp_, _config.tmpcachesize);
        mProB.getScale(&scaleB_cache, &ldsb_cache, k_padded, n_padded, iterkk, _config.loc[1] + blk_n, _param.paramB,
                       tmp_, _config.tmpcachesize);
        mProB.getReduce(&reduceB_cache, &ldsb_cache, k_padded, n_padded, iterkk, _config.loc[1] + blk_n, _param.paramB,
                        tmp_, _config.tmpcachesize);

        int bcache_stride = bcache_step * sizeof(BType);
        for (int i = 0; i < blk_msize; i += GemmCore::MTILE) {
          int m_remain = utils::remainsize(i, blk_msize, GemmCore::MTILE);
          auto cptr_cache = tmpC + i * _config.block[1];
          int ccache_stride = _config.block[1] * sizeof(CType);
          auto aptr_cache = tmpA;
          int acache_step = k_padded;
          auto zpA_cache = zpA;
          auto scaleA_cache = scaleA;
          int ldsa_cache = tmp_ldsa;
          mProA.getActivation(&aptr_cache, &acache_step, _param.paramA, m_remain, k_padded,
                              (blk_m + i + _config.loc[0]), iterkk, tmp_, _config.tmpcachesize);
          mProA.getZp(&zpA_cache, &ldsa_cache, _param.paramA, m_remain, k_padded, (blk_m + i + _config.loc[0]), iterkk,
                      tmp_, _config.tmpcachesize);
          mProA.getScale(&scaleA_cache, &ldsa_cache, _param.paramA, m_remain, k_padded, (blk_m + i + _config.loc[0]),
                         iterkk, tmp_, _config.tmpcachesize);
          auto kscale = k_remain / float(KBlock);
          mGemmCore.forward(aptr_cache, bptr_cache, cptr_cache, zpA_cache, scaleA_cache, ldsa_cache, scaleB_cache,
                            reduceB_cache, ldsb_cache, m_remain, n_padded, k_padded, k_padded,
                            acache_step * sizeof(AType), bcache_stride, ccache_stride, iterkk, kscale, tmp_,
                            _config.tmpcachesize);
        }
      }
    }
    mEpilogue.forward(tmpC, _config.block[1], (_config.loc[0] + blk_m), _config.loc[1] + blk_n, blk_msize, blk_nsize,
                      _param.paramC, tmpcache, _config.tmpcachesize);
  }
};
}  // namespace gemm
}  // namespace wrapper
}  // namespace bestla
