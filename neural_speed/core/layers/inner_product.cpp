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
#include "bestla_common.hpp"
#include "bestla_gemm.h"

using namespace bestla;     // NOLINT
using namespace ne_bestla;  // NOLINT

unsigned long long bestla_f32f32_get_workspace_size(int _m, int _n, int _k, void* wptr) {  // NOLINT
  // maximum padding
  int constexpr padding = 128;
  size_t s = static_cast<size_t>(_m) * utils::padto(static_cast<size_t>(_k), padding) * 4;
  return s;
}

// f32f32: activation & output dtype
void bestla_f32f32_forward(float* activation, void* weiptr, float* output, int _m, int _n, int _k, int lda, int ldo,
                           void* workspace) {
  BTLA_GEMM_DATA_PACKED_PARAMS param{activation, weiptr, output, lda, ldo};
  if (!BTLAGemmBatchDriver(_m, _n, _k, 1, &param, reinterpret_cast<int8_t*>(workspace),
                           ne_bestla::ne_threading::get())) {
    printf("Err: invalid parameters\n");
    assert(0);
  }
}

namespace ip_add {
template <class GemmCore_T, template <class, BTLA_ISA> class Wei_T, template <class, BTLA_ISA> class Act_T>
void BTLAGemmCompF32(const int M, const int N, const int K, const float* A, const int lda,
                     storage::gemm::IWeightBase* _B, float* C, const int ldc, float* bias, bool broadcast_bias,
                     int8_t* WorkSpace, parallel::IThreading* th) {
  if (M <= 16) {
    using Parallel = parallel::gemm::SchedulerKBlock<GemmCore_T>;
    using Launcher = wrapper::gemm::LauncherKBlock<GemmCore_T::ISA, GemmCore_T, Act_T, Wei_T,
                                                   epilogue::gemm::CompFp32BlockEpilogue, custom::epilogue::AddFp32>;
    auto B = reinterpret_cast<typename Launcher::PrologueB::StorageWeight*>(_B);
    static Launcher kernel;
    utils::GemmProblem gp(1, M, N, K, B->mBlockSize);
    auto reduceA = kernel.mProA.createReduceStorage(M, K, B->mBlockSize);
    auto reordA = kernel.mProA.createReorderStorage(M, K, B->mBlockSize);
    if (B->IsAsym()) {
      reduceA.assign(WorkSpace);
      WorkSpace += reduceA.mSize;
    }
    if (B->ShfIndice()) {
      reordA.assign(WorkSpace);
    }
    typename Launcher::BEpiParam blkargs{
        B->template SPtr<int8_t>(),     B->SDtype(), B->CStep(), B->template ZPtr<int8_t>(),
        reduceA.template RPtr<float>(), reduceA.lda};
    typename Launcher::Param args{
        gp, {A, lda, &reduceA, B->ShfIndice(), &reordA}, {B}, blkargs, {C, bias, ldc, broadcast_bias ? 0 : ldc}};
    if (B->IsAsym() || B->ShfIndice()) {
      parallel::GemmRunWithA<Parallel>(kernel, args, th);
    } else {
      parallel::GemmRun<Parallel>(kernel, args, th);
    }
  } else {
    using Parallel = parallel::gemm::SchedulerBase<GemmCore_T>;
    using Launcher = wrapper::gemm::LauncherBase<GemmCore_T::ISA, GemmCore_T, Act_T, Wei_T, custom::epilogue::AddFp32>;
    auto B = reinterpret_cast<typename Launcher::PrologueB::StorageWeight*>(_B);
    utils::GemmProblem gp(1, M, N, K, B->mBlockSize);
    static Launcher kernel;
    auto reordA = kernel.mProA.createReorderStorage(M, K, B->mBlockSize);
    typename Launcher::Param args{
        gp, {A, lda, nullptr, B->ShfIndice(), &reordA}, {B}, {C, bias, ldc, broadcast_bias ? 0 : ldc}};
    if (B->ShfIndice()) {
      reordA.assign(WorkSpace);
      parallel::GemmRunWithA<Parallel>(kernel, args, th);
    } else {
      parallel::GemmRun<Parallel>(kernel, args, th);
    }
  }
}

template <class GemmCore_T, template <class, BTLA_ISA> class Wei_T>
void BTLAGemmCompInt8(const int M, const int N, const int K, const float* A, const int lda,
                      storage::gemm::IWeightBase* _B, float* C, const int ldc, float* bias, bool broadcast_bias,
                      int8_t* WorkSpace, parallel::IThreading* th) {
  using Parallel = parallel::gemm::SchedulerKBlockS<GemmCore_T>;
  using Launcher = wrapper::gemm::LauncherIntKBlock<GemmCore_T::ISA, GemmCore_T,
                                                    prologue_a::gemm::ShuffleActivationKBlockQuantizeF32, Wei_T,
                                                    custom::epilogue::AddFp32>;

  auto B = reinterpret_cast<typename Launcher::PrologueB::StorageWeight*>(_B);
  utils::GemmProblem gp(1, M, N, K, B->mBlockSize);
  static Launcher kernel;
  auto quanA = kernel.mProA.createQuantStorage(M, K, B->mBlockSize, B->IsAsym());
  quanA.assign(WorkSpace);
  WorkSpace += quanA.mSize;
  auto reordA = kernel.mProA.createReorderStorage(M, K, B->mBlockSize);
  typename Launcher::Param args{
      gp, {A, lda, &quanA, B->ShfIndice(), &reordA}, {B}, {C, bias, ldc, broadcast_bias ? 0 : ldc}};
  if (B->ShfIndice()) {
    reordA.assign(WorkSpace);
    kernel.mProA.quantize({A, lda, &quanA, B->ShfIndice(), &reordA}, M, K, th);
    parallel::GemmRun<Parallel>(kernel, args, th);
  } else {
    parallel::GemmRunWithA<Parallel>(kernel, args, th);
  }
}
}  // namespace ip_add

bool bestla_fusion_add_f32f32_support(void* weiptr, int _m, int _n, int _k) {
  GetCPUDevice();
  bool support = false;
  auto wtmp = storage::gemm::PackedWeightParser::deserialBuffer(weiptr);
  if (wtmp) {
    if (wtmp->mPrologueID == BTLA_PROLOGUEB_IDS::WeightKBlockNInteger) {
      constexpr size_t EleNum = sizeof(AllKBlockCores) / sizeof(AllKBlockCores[0]);  // supported cores
      support = contains(wtmp->mCoreId, AllKBlockCores, EleNum);
      support &= hasISA(AllKBlockCores, EleNum);
    } else if (wtmp->mPrologueID == BTLA_PROLOGUEB_IDS::WeightKBlockNFloat) {
      constexpr size_t EleNum = sizeof(FloatCores) / sizeof(FloatCores[0]);
      support = contains(wtmp->mCoreId, FloatCores, EleNum);
      support &= hasISA(FloatCores, EleNum);
    }
  }
  safe_delete(wtmp);
  return support;
}

void bestla_fusion_add_f32f32_forward(float* activation, void* weiptr, float* bias, float* output, int _m, int _n,
                                      int _k, int lda, int ldo, bool broadcast_bias, void* _workspace) {
  GetCPUDevice();
  auto pth = ne_bestla::ne_threading::get();
  auto ptr = storage::gemm::PackedWeightParser::deserialBuffer(const_cast<void*>(weiptr));
  auto workspace = reinterpret_cast<int8_t*>(_workspace);
  if (ptr) {
    auto coretype = ptr->mCoreId;
    auto NTile = gemm::CoreAttr::get_mask_val(ptr->mCoreId, gemm::CoreAttr::NTILE_MASK, gemm::CoreAttr::NTILE_SHIFT);
    auto PackRow = gemm::CoreAttr::get_packrow(ptr->mCoreId);
    auto CType = gemm::CoreAttr::get_comp(ptr->mCoreId);
    auto btype = static_cast<gemm::CompType>(gemm::CompTypeHelper::get_B(CType));
    if (ptr->mPrologueID == BTLA_PROLOGUEB_IDS::WeightKBlockNInteger) {
      auto bptr = reinterpret_cast<storage::gemm::IWeightKBlockBase*>(ptr);
      auto BlkSize = bptr->mBlockSize;
      if (btype == gemm::CompType::tFP32 && PackRow == 1) {
        if (NTile == tAVX512F::NTILE && _cd->AVX512F() && BlkSize % tAVX512F::KTILE == 0) {
          ip_add::BTLAGemmCompF32<tAVX512F, tWeiNInt, tActKBaseF32>(_m, _n, _k, activation, lda, ptr, output, ldo, bias,
                                                                    broadcast_bias, workspace, pth);
        } else if (NTile == tAVX2::NTILE && _cd->AVX2() && BlkSize % tAVX2::KTILE == 0) {
          ip_add::BTLAGemmCompF32<tAVX2, tWeiNInt, tActKBaseF32>(_m, _n, _k, activation, lda, ptr, output, ldo, bias,
                                                                 broadcast_bias, workspace, pth);
        }
      }
      if (btype == gemm::CompType::tBF16 && PackRow == 2) {
        if (NTile == tAMX_BF16::NTILE && _cd->AMX_BF16() && BlkSize % tAMX_BF16::KTILE == 0) {
          if (_m <= tAVX512_BF16::MTILE) {
            static_assert(tAVX512_BF16::NTILE == tAMX_BF16::NTILE);
            ip_add::BTLAGemmCompF32<tAVX512_BF16, tWeiNInt, tActKBaseF32>(_m, _n, _k, activation, lda, ptr, output, ldo,
                                                                          bias, broadcast_bias, workspace, pth);
          } else {
            ip_add::BTLAGemmCompF32<tAMX_BF16, tWeiNInt, tActKBaseF32>(_m, _n, _k, activation, lda, ptr, output, ldo,
                                                                       bias, broadcast_bias, workspace, pth);
          }
        }
      }
      if (btype == gemm::CompType::tS8 && PackRow == 4) {
        if (NTile == tAMX_INT8_SS_KBlock::NTILE && _cd->AMX_INT8() && BlkSize % tAMX_INT8_SS_KBlock::KTILE == 0) {
          static_assert(tAMX_INT8_SS_KBlock::NTILE == tAVX512_VNNI_KBlock::NTILE);
          if (_m <= tAVX512_VNNI_KBlock::MTILE) {
            ip_add::BTLAGemmCompInt8<tAVX512_VNNI_KBlock, tWeiNInt>(_m, _n, _k, activation, lda, ptr, output, ldo, bias,
                                                                    broadcast_bias, workspace, pth);
          } else {
            ip_add::BTLAGemmCompInt8<tAMX_INT8_SS_KBlock, tWeiNInt>(_m, _n, _k, activation, lda, ptr, output, ldo, bias,
                                                                    broadcast_bias, workspace, pth);
          }

        } else if (NTile == tAVX512_VNNI_KBlock::NTILE && _cd->AVX512_VNNI() &&
                   BlkSize % tAVX512_VNNI_KBlock::KTILE == 0) {
          ip_add::BTLAGemmCompInt8<tAVX512_VNNI_KBlock, tWeiNInt>(_m, _n, _k, activation, lda, ptr, output, ldo, bias,
                                                                  broadcast_bias, workspace, pth);
        } else if (NTile == tAVX_VNNI_KBlock::NTILE && _cd->AVX_VNNI() && BlkSize % tAVX_VNNI_KBlock::KTILE == 0) {
          ip_add::BTLAGemmCompInt8<tAVX_VNNI_KBlock, tWeiNInt>(_m, _n, _k, activation, lda, ptr, output, ldo, bias,
                                                               broadcast_bias, workspace, pth);
        }
      }
    }
    if (ptr->mPrologueID == BTLA_PROLOGUEB_IDS::WeightKBlockNFloat) {
      auto bptr = reinterpret_cast<storage::gemm::IWeightKBlockBase*>(ptr);
      auto BlkSize = bptr->mBlockSize;
      if (btype == gemm::CompType::tFP32 && PackRow == 1) {
        if (NTile == tAVX512F::NTILE && _cd->AVX512F() && BlkSize % tAVX512F::KTILE == 0) {
          ip_add::BTLAGemmCompF32<tAVX512F, tWeiNFloat, tActKBaseF32>(_m, _n, _k, activation, lda, ptr, output, ldo,
                                                                      bias, broadcast_bias, workspace, pth);
        } else if (NTile == tAVX2::NTILE && _cd->AVX2() && BlkSize % tAVX2::KTILE == 0) {
          ip_add::BTLAGemmCompF32<tAVX2, tWeiNFloat, tActKBaseF32>(_m, _n, _k, activation, lda, ptr, output, ldo, bias,
                                                                   broadcast_bias, workspace, pth);
        }
      }
      if (btype == gemm::CompType::tBF16 && PackRow == 2 && BlkSize % tAMX_BF16::KTILE == 0) {
        if (NTile == tAMX_BF16::NTILE && _cd->AMX_BF16()) {
          if (_m <= tAVX512_BF16::MTILE) {
            static_assert(tAVX512_BF16::NTILE == tAMX_BF16::NTILE);
            ip_add::BTLAGemmCompF32<tAVX512_BF16, tWeiNFloat, tActKBaseF32>(_m, _n, _k, activation, lda, ptr, output,
                                                                            ldo, bias, broadcast_bias, workspace, pth);
          } else {
            ip_add::BTLAGemmCompF32<tAMX_BF16, tWeiNFloat, tActKBaseF32>(_m, _n, _k, activation, lda, ptr, output, ldo,
                                                                         bias, broadcast_bias, workspace, pth);
          }
        }
      }
    }
    delete ptr;
  } else {
    printf("Wrong Input\n");
    assert(0);
  }
}
