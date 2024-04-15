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

  void run(const Param& _param, const parallel::gemm::ThreadProblemBase& _config) {
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

 protected:
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
          template <class _T, BTLA_ISA> class _PrologueB_T, template <BTLA_ISA> class _BlockEpilogue_T,
          template <BTLA_ISA> class _Epilogue_T>
class LauncherKBlock {
 public:
  using GemmCore = _GemmCore_T;
  static constexpr BTLA_ISA ISA = _RT_ISA_T;
  using PrologueA = _PrologueA_T<GemmCore, _RT_ISA_T>;
  using PrologueB = _PrologueB_T<GemmCore, _RT_ISA_T>;
  using Epilogue = _Epilogue_T<_RT_ISA_T>;
  using BlockEpilogue = _BlockEpilogue_T<_RT_ISA_T>;
  using AType = typename GemmCore::AType;
  using AParam = typename PrologueA::Param;
  using BType = typename GemmCore::BType;
  using BParam = typename PrologueB::Param;
  using CType = typename GemmCore::CType;
  using BEpiParam = typename BlockEpilogue::Param;
  using EpiParam = typename Epilogue::Param;
  using AccType = float;
  static_assert(GemmCore::ISA <= _RT_ISA_T, "RunTime ISA should cover GEMM's ISA");
  struct Param {
    const utils::GemmProblem problem;
    const AParam paramA;
    const BParam paramB;
    const BEpiParam paramBlk;
    const EpiParam paramC;
  };
  _GemmCore_T mGemmCore;
  PrologueA mProA;
  PrologueB mProB;
  BlockEpilogue mBlockEpi;
  Epilogue mEpilogue;

  void run(const Param& _param, const parallel::gemm::ThreadProblemBase& _config) {
    mGemmCore.configure(_config.size[0], _config.size[1], _param.problem.dims[3]);
    auto StackTmp = alloca(_config.stacksize);
    auto tmpB = reinterpret_cast<BType*>(StackTmp);
    tmpB = utils::cpu_pointer_align(tmpB);
    auto tmpA = reinterpret_cast<AType*>(tmpB + static_cast<size_t>(_config.block[1]) * _config.block[2]);
    tmpA = utils::cpu_pointer_align(tmpA);
    auto tmpC = reinterpret_cast<AccType*>(tmpA + static_cast<size_t>(GemmCore::MTILE) * _config.block[2]);
    tmpC = utils::cpu_pointer_align(tmpC);
    auto tmpBlk = reinterpret_cast<CType*>(tmpC + static_cast<size_t>(_config.block[0]) * _config.block[1]);
    tmpBlk = utils::cpu_pointer_align(tmpBlk);
    auto tmpCache = reinterpret_cast<void*>(tmpBlk + static_cast<size_t>(_config.block[0]) * _config.block[1]);
    tmpCache = utils::cpu_pointer_align(tmpCache);
    for (int itern = 0; itern < _config.size[1]; itern += _config.block[1]) {
      int n_remain = utils::remainsize(itern, _config.size[1], _config.block[1]);
      for (int iterm = 0; iterm < _config.size[0]; iterm += _config.block[0]) {
        int m_remain = utils::remainsize(iterm, _config.size[0], _config.block[0]);
        std::memset(tmpC, 0, _config.block[0] * _config.block[1] * sizeof(AccType));
        auto& KBlock = _param.problem.dims[4];
        if (KBlock <= _config.block[2]) {
          run_block(_param, _config, iterm, itern, m_remain, n_remain, tmpA, tmpB, tmpBlk, tmpC, tmpCache);
        } else {
          run_block_large(_param, _config, iterm, itern, m_remain, n_remain, tmpA, tmpB, tmpBlk, tmpC, tmpCache);
        }
      }
    }
  }

 protected:
  void run_block(const Param& _param, const parallel::gemm::ThreadProblemBase& _config, int blk_m, int blk_n,
                 int blk_msize, int blk_nsize, AType* tmpA, BType* tmpB, CType* tmpBlk, AccType* tmpC, void* tmpcache) {
    int n_padded = utils::padto(blk_nsize, GemmCore::NTILE);
    auto& K = _param.problem.dims[3];
    auto& KBlock = _param.problem.dims[4];
    for (int iterk = 0; iterk < K; iterk += _config.block[2]) {
      int k_remain = utils::remainsize(iterk, K, _config.block[2]);
      int k_padded = utils::padto(k_remain, GemmCore::KTILE);
      auto bptr_cache = tmpB;
      int bcache_step = 0;
      mProB.getKBlockWeight(&bptr_cache, &bcache_step, k_padded, n_padded, iterk, _config.loc[1] + blk_n, _param.paramB,
                            tmpcache, _config.tmpcachesize);
      int bcache_stride = bcache_step * sizeof(BType);

      for (int ikk = 0; ikk < k_remain; ikk += KBlock) {
        int k_remain1 = utils::remainsize(iterk + ikk, K, KBlock);
        int k_paddedle1 = utils::padto_le(k_remain1, GemmCore::KTILE);
        for (int i = 0; i < blk_msize; i += GemmCore::MTILE) {
          int m_remain = utils::remainsize(i, blk_msize, GemmCore::MTILE);
          auto cptr_cache = tmpBlk + i * _config.block[1];
          int ccache_stride = _config.block[1] * sizeof(CType);
          if (k_paddedle1) {
            AType* aptr_cache = tmpA;
            int acache_step = 0;
            mProA.getActivation(&aptr_cache, &acache_step, _param.paramA, m_remain, k_paddedle1,
                                (blk_m + i + _config.loc[0]), iterk + ikk, tmpcache, _config.tmpcachesize);
            mGemmCore.forward(aptr_cache, bptr_cache + ikk * GemmCore::NTILE, cptr_cache, m_remain, n_padded,
                              k_paddedle1, acache_step * sizeof(AType), bcache_stride, ccache_stride, 0, tmpcache,
                              _config.tmpcachesize);
          }
          int k_tail = k_remain1 - k_paddedle1;
          if (k_tail) {
            AType* aptr_cache = tmpA;
            int acache_step = 0;
            mProA.getActivation(&aptr_cache, &acache_step, _param.paramA, m_remain, k_tail,
                                (blk_m + i + _config.loc[0]), iterk + ikk + k_paddedle1, tmpcache,
                                _config.tmpcachesize);
            mGemmCore.forward(aptr_cache, bptr_cache + (ikk + k_paddedle1) * GemmCore::NTILE, cptr_cache, m_remain,
                              n_padded, k_tail, acache_step * sizeof(AType), bcache_stride, ccache_stride,
                              0 + k_paddedle1, tmpcache, _config.tmpcachesize);
          }
        }
        mBlockEpi.forward(tmpBlk, tmpC, _config.block[1], (_config.loc[0] + blk_m), _config.loc[1] + blk_n,
                          (iterk + ikk) / KBlock, blk_msize, blk_nsize, _param.paramBlk, tmpcache,
                          _config.tmpcachesize);
      }
    }
    auto cachewithblk = _config.tmpcachesize + static_cast<size_t>(_config.block[0]) * _config.block[1] * sizeof(CType);
    mEpilogue.forward(tmpC, _config.block[1], (_config.loc[0] + blk_m), _config.loc[1] + blk_n, blk_msize, blk_nsize,
                      _param.paramC, tmpBlk, cachewithblk);
  }

  void run_block_large(const Param& _param, const parallel::gemm::ThreadProblemBase& _config, int blk_m, int blk_n,
                       int blk_msize, int blk_nsize, AType* tmpA, BType* tmpB, CType* tmpBlk, AccType* tmpC,
                       void* tmpcache) {
    int n_padded = utils::padto(blk_nsize, GemmCore::NTILE);
    auto& K = _param.problem.dims[3];
    auto KBlock = _param.problem.dims[4];
    assert(K % KBlock == 0);
    for (int iterk = 0; iterk < K; iterk += KBlock) {
      memset(tmpBlk, 0, sizeof(CType) * blk_msize * _config.block[1]);
      for (int iblkk = 0; iblkk < KBlock; iblkk += _config.block[2]) {
        int k_remain = utils::remainsize(iterk + iblkk, iterk + KBlock, _config.block[2]);
        int k_padded = utils::padto(k_remain, GemmCore::KTILE);
        int k_paddedle = utils::padto_le(k_remain, GemmCore::KTILE);
        auto bptr_cache = tmpB;
        int bcache_step = 0;
        mProB.getKBlockWeight(&bptr_cache, &bcache_step, k_padded, n_padded, iterk + iblkk, _config.loc[1] + blk_n,
                              _param.paramB, tmpcache, _config.tmpcachesize);
        int bcache_stride = bcache_step * sizeof(BType);
        for (int i = 0; i < blk_msize; i += GemmCore::MTILE) {
          int m_remain = utils::remainsize(i, blk_msize, GemmCore::MTILE);
          auto cptr_cache = tmpBlk + i * _config.block[1];
          int ccache_stride = _config.block[1] * sizeof(CType);
          if (k_paddedle) {
            AType* aptr_cache = tmpA;
            int acache_step = 0;
            mProA.getActivation(&aptr_cache, &acache_step, _param.paramA, m_remain, k_paddedle,
                                (blk_m + i + _config.loc[0]), iterk + iblkk, tmpcache, _config.tmpcachesize);
            mGemmCore.forward(aptr_cache, bptr_cache, cptr_cache, m_remain, n_padded, k_paddedle,
                              acache_step * sizeof(AType), bcache_stride, ccache_stride, iblkk, tmpcache,
                              _config.tmpcachesize);
          }
          int k_tail = k_remain - k_paddedle;
          if (k_tail) {
            AType* aptr_cache = tmpA;
            int acache_step = 0;
            mProA.getActivation(&aptr_cache, &acache_step, _param.paramA, m_remain, k_tail,
                                (blk_m + i + _config.loc[0]), iterk + k_paddedle + iblkk, tmpcache,
                                _config.tmpcachesize);
            mGemmCore.forward(aptr_cache, bptr_cache + k_paddedle * GemmCore::NTILE, cptr_cache, m_remain, n_padded,
                              k_tail, acache_step * sizeof(AType), bcache_stride, ccache_stride, iblkk + k_paddedle,
                              tmpcache, _config.tmpcachesize);
          }
        }
      }
      mBlockEpi.forward(tmpBlk, tmpC, _config.block[1], (_config.loc[0] + blk_m), _config.loc[1] + blk_n,
                        iterk / KBlock, blk_msize, blk_nsize, _param.paramBlk, tmpcache, _config.tmpcachesize);
    }
    auto cachewithblk = _config.tmpcachesize + static_cast<size_t>(_config.block[0]) * _config.block[1] * sizeof(CType);
    mEpilogue.forward(tmpC, _config.block[1], (_config.loc[0] + blk_m), _config.loc[1] + blk_n, blk_msize, blk_nsize,
                      _param.paramC, tmpBlk, cachewithblk);
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
      if constexpr (!std::is_same_v<PrologueA, prologue_a::gemm::ActivationF32KBlockQuantize<_GemmCore_T, _RT_ISA_T>>) {
        return false;
      }
      if constexpr (GemmCore::ISA == BTLA_ISA::AVX_VNNI) {
        static_assert(GemmCore::PACK_ROW == 4);
        if constexpr (GemmCore::COMP == bestla::gemm::CompType::COMP_INT8_US_FP32) {
          return true;
        }
      }
      return false;
    }

    static bool implemented(const Param& _param) {
      bool impl = true;
      impl &= _param.paramB.packedW->mDType == BTLA_DTYPE::S4_CLIP;
      impl &= _param.paramB.packedW->mCorrection.mScaT == BTLA_DTYPE::F32 ||
              _param.paramB.packedW->mCorrection.mScaT == BTLA_DTYPE::BF16;
      impl &= _param.problem.dims[1] == 1;  // m==1
      return impl;
    }
    template <typename ScaleT>
    static void gemv_s4(const Param& _param, const parallel::gemm::ThreadProblemBase& _config) {
      utils::GemvParamB<ScaleT> paramB{_param.paramB.packedW->template WPtr<uint8_t>(), nullptr, nullptr,
                                       _param.paramB.packedW->template SPtr<ScaleT>(), nullptr};
      utils::GemvParamA paramA{_param.paramA.quan->template APtr<uint8_t>(), _param.paramA.quan->template SPtr<float>(),
                               _param.paramA.quan->template ZPtr<uint8_t>()};
      auto constexpr TmpSize = 4 * 1024LL;
      auto StackTmp = alloca(TmpSize);
      int m = _param.problem.dims[1];
      int n = _param.problem.dims[2];
      int k = _param.problem.dims[3];
      int kblocksize = _param.problem.dims[4];
      auto Cptr = _param.paramC.C + _config.loc[1];
      paramB.b4ptr += _config.loc[1] * _param.paramB.packedW->mKPad / 2;
      paramB.sptr += _config.loc[1];
      int size_padded = utils::padto_le(_config.size[1], GemmCore::NTILE);
      int in = 0;
      int ld_scaleb = _param.paramB.packedW->CStep();
      for (; in < size_padded; in += GemmCore::NTILE) {
        kernel::wrapper::GEMV_4Bit::forward_u8s8_fp32<_RT_ISA_T, ScaleT, GemmCore::NTILE>(paramA, paramB, Cptr, k,
                                                                                          ld_scaleb, kblocksize);
        Cptr += GemmCore::NTILE;
        paramB.b4ptr += GemmCore::NTILE * _param.paramB.packedW->mKPad / 2;
        paramB.sptr += GemmCore::NTILE;
      }
      if (size_padded != _config.size[1]) {
        auto tmpptr = reinterpret_cast<CType*>(StackTmp);
        kernel::wrapper::GEMV_4Bit::forward_u8s8_fp32<_RT_ISA_T, ScaleT, GemmCore::NTILE>(paramA, paramB, tmpptr, k,
                                                                                          ld_scaleb, kblocksize);
        memcpy(Cptr, tmpptr, (_config.size[1] - in) * sizeof(CType));
      }
      Epilogue::forward(_param.paramC.C + _config.loc[1], 0, 0, _config.loc[1], 1, _config.size[1], _param.paramC,
                        StackTmp, TmpSize);
    }

    static void gemv(const Param& _param, const parallel::gemm::ThreadProblemBase& _config) {
      if (_param.paramB.packedW->mDType == BTLA_DTYPE::S4_CLIP) {
        if (_param.paramB.packedW->SDtype() == BTLA_DTYPE::F32) {
          gemv_s4<float>(_param, _config);
        } else if (_param.paramB.packedW->SDtype() == BTLA_DTYPE::BF16) {
          gemv_s4<utils::bf16>(_param, _config);
        }
        return;
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
      mProB.getKBlockWeight(&bptr_cache, &bcache_step, k_padded, n_padded, iterk, _config.loc[1] + blk_n, _param.paramB,
                            tmp_, _config.tmpcachesize);
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
        mProB.getKBlockWeight(&bptr_cache, &bcache_step, k_padded, n_padded, iterkk, _config.loc[1] + blk_n,
                              _param.paramB, tmp_, _config.tmpcachesize);
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
