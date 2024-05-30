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

/*++

Module Name:

    bestla_gemm.cpp

Abstract:

    C APIs of BesTLA GEMMs.
--*/

#include "bestla_gemm.h"
#include <cstdint>

#include "bestla_defs.h"

using namespace bestla;  // NOLINT

namespace {
template <class GemmCore_T, template <class> class Wei_T>
void BTLAGemmCompF32(const int M, const int N, const int K, const float* A, const int lda,
                     storage::gemm::IWeightBase* _B, float* C, const int ldc, int8_t* WorkSpace,
                     parallel::IThreading* th) {
  using Parallel = parallel::gemm::SchedulerBase<GemmCore_T>;
  using Launcher = wrapper::gemm::LauncherBase<GemmCore_T, prologue_a::gemm::ShuffleActivationKBlockBaseF32, Wei_T,
                                               epilogue::gemm::AccumulatorWriteBackFp32>;
  auto B = reinterpret_cast<typename Launcher::PrologueB::StorageWeight*>(_B);
  utils::GemmProblem gp(1, M, N, K, B->mBlockSize);
  auto reordA = Launcher::PrologueA::createReorderStorage(M, K, B->mBlockSize);
  typename Launcher::Param args{gp, {A, K, nullptr, B->ShfIndice(), &reordA}, {B}, {C, N}};
  if (B->ShfIndice()) {
    reordA.assign(WorkSpace);
    parallel::GemmRunWithA<Parallel, Launcher>(args, th);
  } else {
    parallel::GemmRun<Parallel, Launcher>(args, th);
  }
}

template <class GemmCore_T, template <class> class Wei_T>
void BTLAGemmCompInt8Pc(const int M, const int N, const int K, const float* A, const int lda,
                        storage::gemm::IWeightBase* _B, float* C, const int ldc, int8_t* WorkSpace,
                        parallel::IThreading* th) {
  using Parallel = parallel::gemm::SchedulerBase<GemmCore_T>;
  using Launcher = wrapper::gemm::LauncherBase<GemmCore_T, prologue_a::gemm::ShuffleActivationKBlockQuantizeF32, Wei_T,
                                               PcWriteBackF32>;
  auto B = reinterpret_cast<typename Launcher::PrologueB::StorageWeight*>(_B);
  assert(B->mBlockSize >= K);
  utils::GemmProblem gp(1, M, N, K, B->mBlockSize);
  auto quanA = Launcher::PrologueA::createQuantStorage(M, K, B->mBlockSize, B->IsAsym());
  quanA.assign(WorkSpace);
  WorkSpace += quanA.mSize;
  auto reordA = Launcher::PrologueA::createReorderStorage(M, K, B->mBlockSize);
  typename Launcher::Param args{
      gp,
      {A, K, &quanA, B->ShfIndice(), &reordA},
      {B},
      {{B->template SPtr<char>(), B->SDtype(), quanA.template SPtr<float>(), quanA.template ZPtr<uint8_t>(),
        B->template RPtr<char>(), B->RDtype(), nullptr, nullptr, K},
       {C, N}}};
  if (B->ShfIndice()) {
    reordA.assign(WorkSpace);
    Launcher::PrologueA::quantize({A, K, &quanA, B->ShfIndice(), &reordA}, M, K, th);
    parallel::GemmRun<Parallel, Launcher>(args, th);
  } else {
    parallel::GemmRunWithA<Parallel, Launcher>(args, th);
  }
}

template <class GemmCore_T, template <class> class Wei_T>
void BTLAGemmCompInt8(const int M, const int N, const int K, const float* A, const int lda,
                      storage::gemm::IWeightBase* _B, float* C, const int ldc, int8_t* WorkSpace,
                      parallel::IThreading* th) {
  using Parallel = parallel::gemm::SchedulerKBlockS<GemmCore_T>;
  using Launcher = tLauncher_Int8_F32F32<GemmCore_T, Wei_T>;
  auto B = reinterpret_cast<typename Launcher::PrologueB::StorageWeight*>(_B);
  utils::GemmProblem gp(1, M, N, K, B->mBlockSize);
  auto quanA = Launcher::PrologueA::createQuantStorage(M, K, B->mBlockSize, B->IsAsym());
  quanA.assign(WorkSpace);
  WorkSpace += quanA.mSize;
  auto reordA = Launcher::PrologueA::createReorderStorage(M, K, B->mBlockSize);
  typename Launcher::Param args{gp, {A, K, &quanA, B->ShfIndice(), &reordA}, {B}, {C, N}};
  if (B->ShfIndice()) {
    reordA.assign(WorkSpace);
    Launcher::PrologueA::quantize({A, K, &quanA, B->ShfIndice(), &reordA}, M, K, th);
    parallel::GemmRun<Parallel, Launcher>(args, th);
  } else {
    parallel::GemmRunWithA<Parallel, Launcher>(args, th);
  }
}

bool BTLAGemmBatchDriver(const size_t M, const size_t N, const size_t K, const size_t BatchN,
                         const BTLA_GEMM_DATA_PACKED_PARAMS* DataParams, int8_t* WorkSpace, void* ThreadPool) {
  GetCPUDevice();
  auto pth = reinterpret_cast<parallel::IThreading*>(ThreadPool);
  bool processed = true;
  for (size_t i = 0; i < BatchN; i++) {
    auto ptr = storage::gemm::PackedWeightParser::deserialBuffer(const_cast<void*>(DataParams[i].B));
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
            BTLAGemmCompF32<tAVX512F, tWeiNInt>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr, DataParams[i].C,
                                                DataParams[i].ldc, WorkSpace, pth);
          } else if (NTile == tAVX2::NTILE && _cd->AVX2() && BlkSize % tAVX2::KTILE == 0) {
            BTLAGemmCompF32<tAVX2, tWeiNInt>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr, DataParams[i].C,
                                             DataParams[i].ldc, WorkSpace, pth);
          }
        }
        if (btype == gemm::CompType::tBF16 && PackRow == 2) {
          if (NTile == tAMX_BF16::NTILE && _cd->AMX_BF16() && BlkSize % tAMX_BF16::KTILE == 0) {
            if (M <= tAVX512_BF16::MTILE) {
              static_assert(tAVX512_BF16::NTILE == tAMX_BF16::NTILE);
              BTLAGemmCompF32<tAVX512_BF16, tWeiNInt>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr, DataParams[i].C,
                                                      DataParams[i].ldc, WorkSpace, pth);
            } else {
              BTLAGemmCompF32<tAMX_BF16, tWeiNInt>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr, DataParams[i].C,
                                                   DataParams[i].ldc, WorkSpace, pth);
            }
          }
        }
        if (btype == gemm::CompType::tS8 && PackRow == 4) {
          if (NTile == tAMX_INT8_US_KBlock::NTILE && _cd->AMX_INT8() && BlkSize % tAMX_INT8_US_KBlock::KTILE == 0) {
            if (bptr->mBlockSize < K) {
              BTLAGemmCompInt8<tAMX_INT8_US_KBlock, tWeiNInt>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr,
                                                              DataParams[i].C, DataParams[i].ldc, WorkSpace, pth);
            } else {
              BTLAGemmCompInt8Pc<tAMX_INT8_US, tWeiNInt>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr,
                                                         DataParams[i].C, DataParams[i].ldc, WorkSpace, pth);
            }

          } else if (NTile == tAVX512_VNNI_KBlock::NTILE && _cd->AVX512_VNNI() &&
                     BlkSize % tAVX512_VNNI_KBlock::KTILE == 0) {
            if (bptr->mBlockSize < K) {
              BTLAGemmCompInt8<tAVX512_VNNI_KBlock, tWeiNInt>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr,
                                                              DataParams[i].C, DataParams[i].ldc, WorkSpace, pth);
            } else {
              BTLAGemmCompInt8Pc<tAVX512_VNNI, tWeiNInt>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr,
                                                         DataParams[i].C, DataParams[i].ldc, WorkSpace, pth);
            }
          } else if (NTile == tAVX512BW_KBlock::NTILE && _cd->AVX512BW() && BlkSize % tAVX512BW_KBlock::KTILE == 0) {
            if (bptr->mBlockSize < K) {
              BTLAGemmCompInt8<tAVX512BW_KBlock, tWeiNInt>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr,
                                                           DataParams[i].C, DataParams[i].ldc, WorkSpace, pth);
            } else {
              BTLAGemmCompInt8Pc<tAVX512BW, tWeiNInt>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr, DataParams[i].C,
                                                      DataParams[i].ldc, WorkSpace, pth);
            }

          } else if (NTile == tAVX_VNNI_KBlock::NTILE && _cd->AVX_VNNI() && BlkSize % tAVX_VNNI_KBlock::KTILE == 0) {
            if (bptr->mBlockSize < K) {
              BTLAGemmCompInt8<tAVX_VNNI_KBlock, tWeiNInt>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr,
                                                           DataParams[i].C, DataParams[i].ldc, WorkSpace, pth);
            } else {
              BTLAGemmCompInt8Pc<tAVX_VNNI, tWeiNInt>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr, DataParams[i].C,
                                                      DataParams[i].ldc, WorkSpace, pth);
            }
          } else if (NTile == tAVX2_VNNI_KBlock::NTILE && _cd->AVX2() && BlkSize % tAVX2_VNNI_KBlock::KTILE == 0) {
            if (bptr->mBlockSize < K) {
              BTLAGemmCompInt8<tAVX2_VNNI_KBlock, tWeiNInt>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr,
                                                            DataParams[i].C, DataParams[i].ldc, WorkSpace, pth);
            } else {
              BTLAGemmCompInt8Pc<tAVX2_VNNI, tWeiNInt>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr,
                                                       DataParams[i].C, DataParams[i].ldc, WorkSpace, pth);
            }
          }
        }
      }
      if (ptr->mPrologueID == BTLA_PROLOGUEB_IDS::WeightKBlockNFloat) {
        auto bptr = reinterpret_cast<storage::gemm::IWeightKBlockBase*>(ptr);
        auto BlkSize = bptr->mBlockSize;
        if (btype == gemm::CompType::tFP32 && PackRow == 1) {
          if (NTile == tAVX512F::NTILE && _cd->AVX512F() && BlkSize % tAVX512F::KTILE == 0) {
            BTLAGemmCompF32<tAVX512F, tWeiNFloat>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr, DataParams[i].C,
                                                  DataParams[i].ldc, WorkSpace, pth);
          } else if (NTile == tAVX2::NTILE && _cd->AVX2() && BlkSize % tAVX2::KTILE == 0) {
            BTLAGemmCompF32<tAVX2, tWeiNFloat>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr, DataParams[i].C,
                                               DataParams[i].ldc, WorkSpace, pth);
          }
        }
        if (btype == gemm::CompType::tBF16 && PackRow == 2) {
          if (NTile == tAMX_BF16::NTILE && _cd->AMX_BF16() && BlkSize % tAMX_BF16::KTILE == 0) {
            if (M <= tAVX512_BF16::MTILE) {
              static_assert(tAVX512_BF16::NTILE == tAMX_BF16::NTILE);
              BTLAGemmCompF32<tAVX512_BF16, tWeiNFloat>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr,
                                                        DataParams[i].C, DataParams[i].ldc, WorkSpace, pth);
            } else {
              BTLAGemmCompF32<tAMX_BF16, tWeiNFloat>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr, DataParams[i].C,
                                                     DataParams[i].ldc, WorkSpace, pth);
            }
          }
        }
      }
      delete ptr;
    } else {
      processed = false;
      break;
    }
  }
  return processed;
}

template <typename T>
size_t BTLABuSize(int block_size, size_t N, size_t K, BTLA_DTYPE QuantType, BTLA_DTYPE ScaleDtype, bool isAsym,
                  int* shuffle_indice) {
  using WType = typename T::PrologueB::StorageWeight;
  WType stor(0);
  if constexpr (std::is_same_v<typename T::PrologueB,
                               prologue_b::gemm::WeightKBlockNInteger<typename T::GemmCore>>) {
    stor = T::PrologueB::createStorage(N, K, block_size, QuantType, ScaleDtype, BTLA_DTYPE::BF16, isAsym);
    if (shuffle_indice != nullptr) {
      T::PrologueB::enableShuffle(&stor);
    }
  } else {
    stor = T::PrologueB::createStorage(N, K, block_size, QuantType, ScaleDtype);
    (void)(shuffle_indice);
  }

  // Reduce dtype set to bf16
  return stor.mSize;
}
template <template <class> class Wei_T>
size_t BTLAGemmPackBSizeLocal(size_t N, size_t K, size_t BlkSize, BTLA_DTYPE QuantType, BTLA_DTYPE ScaleDtype,
                              bool isAsym, ne_comp_type CompType, int* shuffle_indice) {
  GetCPUDevice();
  auto dtype_type = utils::bestla_dtype_type(QuantType);
  auto constexpr dtype_int = utils::bestla_dtype_type(BTLA_DTYPE::TypeInt);
  // from low precision to high precision
  switch (CompType) {
    case NE_COMP_INT8:
      if (dtype_type == dtype_int && !(QuantType == BTLA_DTYPE::S8 && isAsym)) {
        if (_cd->AMX_INT8() && BlkSize % tAMX_INT8_US_KBlock::KTILE == 0) {
          return BTLABuSize<tLauncher_Int8_F32F32<tAMX_INT8_US_KBlock, Wei_T>>(
              static_cast<int>(BlkSize), N, K, QuantType, ScaleDtype, isAsym, shuffle_indice);
        }
        if (_cd->AVX512_VNNI() && BlkSize % tAVX512_VNNI_KBlock::KTILE == 0) {
          return BTLABuSize<tLauncher_Int8_F32F32<tAVX512_VNNI_KBlock, Wei_T>>(
              static_cast<int>(BlkSize), N, K, QuantType, ScaleDtype, isAsym, shuffle_indice);
        }
        if (_cd->AVX512BW() && BlkSize % tAVX512BW_KBlock::KTILE == 0) {
          return BTLABuSize<tLauncher_Int8_F32F32<tAVX512BW_KBlock, Wei_T>>(static_cast<int>(BlkSize), N, K, QuantType,
                                                                            ScaleDtype, isAsym, shuffle_indice);
        }
        if (_cd->AVX_VNNI() && BlkSize % tAVX_VNNI_KBlock::KTILE == 0) {
          return BTLABuSize<tLauncher_Int8_F32F32<tAVX_VNNI_KBlock, Wei_T>>(static_cast<int>(BlkSize), N, K, QuantType,
                                                                            ScaleDtype, isAsym, shuffle_indice);
        }
        if (_cd->AVX2() && BlkSize % tAVX2_VNNI_KBlock::KTILE == 0) {
          return BTLABuSize<tLauncher_Int8_F32F32<tAVX2_VNNI_KBlock, Wei_T>>(static_cast<int>(BlkSize), N, K, QuantType,
                                                                             ScaleDtype, isAsym, shuffle_indice);
        }
      }
      [[fallthrough]];
    case NE_COMP_F16:
    case NE_COMP_BF16:
      if (_cd->AMX_BF16() && BlkSize % tAMX_BF16::KTILE == 0) {
        return BTLABuSize<tLauncher_Int8_F32F32<tAMX_BF16, Wei_T>>(static_cast<int>(BlkSize), N, K, QuantType,
                                                                   ScaleDtype, isAsym, shuffle_indice);
      }
      [[fallthrough]];
    case NE_COMP_F32:
    case NE_COMP_UNDEF:  // currently only f32 activation
      if (_cd->AVX512F() && BlkSize % tAVX512F::KTILE == 0) {
        return BTLABuSize<tLauncher_Fp_F32F32<tAVX512F, Wei_T>>(static_cast<int>(BlkSize), N, K, QuantType, ScaleDtype,
                                                                isAsym, shuffle_indice);
      }
      if (_cd->AVX2() && BlkSize % tAVX2::KTILE == 0) {
        return BTLABuSize<tLauncher_Fp_F32F32<tAVX2, Wei_T>>(static_cast<int>(BlkSize), N, K, QuantType, ScaleDtype,
                                                             isAsym, shuffle_indice);
      }
      [[fallthrough]];
    default:
      return 0;
  }
  return 0;
}

template <typename T>
void BTLAGemmQuantPackB(void* PackedBuf, int BlkSize, const float* FpData, int N, int K, BTLA_DTYPE QuantType,
                        BTLA_DTYPE ScaleDtype, bool IsAsym, int ldb, bool IsTrans, void* ThreadPool) {
  using WType = typename T::PrologueB::StorageWeight;
  WType stor(0);
  if constexpr (std::is_same_v<typename T::PrologueB,
                               prologue_b::gemm::WeightKBlockNInteger<typename T::GemmCore>>) {
    stor = T::PrologueB::createStorage(N, K, BlkSize, QuantType, ScaleDtype, BTLA_DTYPE::BF16, IsAsym);
  } else {
    stor = T::PrologueB::createStorage(N, K, BlkSize, QuantType, ScaleDtype);
  }
  stor.assign(reinterpret_cast<int8_t*>(PackedBuf));
  auto pth = reinterpret_cast<parallel::IThreading*>(ThreadPool);
  if (IsTrans) {
    T::PrologueB::packTransposeWeight(N, K, FpData, ldb, &stor, pth);
  } else {
    T::PrologueB::packWeight(N, K, FpData, ldb, &stor, pth);
  }
}

template <template <class> class Wei_T>
bool BTLAGemmQuantPackBLocal(void* PackedBuf, const float* FpData, size_t N, size_t K, size_t ldb, size_t BlkSize,
                             BTLA_DTYPE QuantType, BTLA_DTYPE ScaleDtype, bool isAsym, ne_comp_type CompType,
                             bool isTrans, void* ThreadPool) {
  GetCPUDevice();
  auto dtype_type = utils::bestla_dtype_type(QuantType);
  auto constexpr dtype_int = utils::bestla_dtype_type(BTLA_DTYPE::TypeInt);
  switch (CompType) {
    case NE_COMP_INT8:
      if (dtype_type == dtype_int && !(QuantType == BTLA_DTYPE::S8 && isAsym)) {
        if (_cd->AMX_INT8() && BlkSize % tAMX_INT8_US_KBlock::KTILE == 0) {
          BTLAGemmQuantPackB<tLauncher_Int8_F32F32<tAMX_INT8_US_KBlock, Wei_T>>(
              PackedBuf, static_cast<int>(BlkSize), FpData, static_cast<int>(N), static_cast<int>(K), QuantType,
              ScaleDtype, isAsym, static_cast<int>(ldb), isTrans, ThreadPool);
          return true;
        }
        if (_cd->AVX512_VNNI() && BlkSize % tAVX512_VNNI_KBlock::KTILE == 0) {
          BTLAGemmQuantPackB<tLauncher_Int8_F32F32<tAVX512_VNNI_KBlock, Wei_T>>(
              PackedBuf, static_cast<int>(BlkSize), FpData, static_cast<int>(N), static_cast<int>(K), QuantType,
              ScaleDtype, isAsym, static_cast<int>(ldb), isTrans, ThreadPool);
          return true;
        }
        if (_cd->AVX512BW() && BlkSize % tAVX512BW_KBlock::KTILE == 0) {
          BTLAGemmQuantPackB<tLauncher_Int8_F32F32<tAVX512BW_KBlock, Wei_T>>(
              PackedBuf, static_cast<int>(BlkSize), FpData, static_cast<int>(N), static_cast<int>(K), QuantType,
              ScaleDtype, isAsym, static_cast<int>(ldb), isTrans, ThreadPool);
          return true;
        }
        if (_cd->AVX_VNNI() && BlkSize % tAVX_VNNI_KBlock::KTILE == 0) {
          BTLAGemmQuantPackB<tLauncher_Int8_F32F32<tAVX_VNNI_KBlock, Wei_T>>(
              PackedBuf, static_cast<int>(BlkSize), FpData, static_cast<int>(N), static_cast<int>(K), QuantType,
              ScaleDtype, isAsym, static_cast<int>(ldb), isTrans, ThreadPool);
          return true;
        }
        if (_cd->AVX2() && BlkSize % tAVX2_VNNI_KBlock::KTILE == 0) {
          BTLAGemmQuantPackB<tLauncher_Int8_F32F32<tAVX2_VNNI_KBlock, Wei_T>>(
              PackedBuf, static_cast<int>(BlkSize), FpData, static_cast<int>(N), static_cast<int>(K), QuantType,
              ScaleDtype, isAsym, static_cast<int>(ldb), isTrans, ThreadPool);
          return true;
        }
      }
      [[fallthrough]];
    case NE_COMP_F16:
    case NE_COMP_BF16:
      if (_cd->AMX_BF16() && BlkSize % tAMX_BF16::KTILE == 0) {
        BTLAGemmQuantPackB<tLauncher_Fp_F32F32<tAMX_BF16, Wei_T>>(
            PackedBuf, static_cast<int>(BlkSize), FpData, static_cast<int>(N), static_cast<int>(K), QuantType,
            ScaleDtype, isAsym, static_cast<int>(ldb), isTrans, ThreadPool);
        return true;
      }
      [[fallthrough]];
    case NE_COMP_F32:
    case NE_COMP_UNDEF:  // currently only f32 activation
      if (_cd->AVX512F() && BlkSize % tAVX512F::KTILE == 0) {
        BTLAGemmQuantPackB<tLauncher_Fp_F32F32<tAVX512F, Wei_T>>(
            PackedBuf, static_cast<int>(BlkSize), FpData, static_cast<int>(N), static_cast<int>(K), QuantType,
            ScaleDtype, isAsym, static_cast<int>(ldb), isTrans, ThreadPool);
        return true;
      }
      if (_cd->AVX2() && BlkSize % tAVX2::KTILE == 0) {
        BTLAGemmQuantPackB<tLauncher_Fp_F32F32<tAVX2, Wei_T>>(
            PackedBuf, static_cast<int>(BlkSize), FpData, static_cast<int>(N), static_cast<int>(K), QuantType,
            ScaleDtype, isAsym, static_cast<int>(ldb), isTrans, ThreadPool);
        return true;
      }
      [[fallthrough]];
    default:
      return false;
  }
  return false;
}

template <typename T>
void BTLAGemmPackBImpl(void* PackedBuf, int BlkSize, const int8_t* QData, const float* Scales, const int8_t* Zp, int N,
                       int K, BTLA_DTYPE QuantType, BTLA_DTYPE ScaleDtype, bool IsAsym, int ldb, int* shuffle_indice,
                       void* ThreadPool) {
  using WType = typename T::PrologueB::StorageWeight;
  auto pth = reinterpret_cast<parallel::IThreading*>(ThreadPool);
  WType stor(0);
  if constexpr (std::is_same_v<typename T::PrologueB,
                               prologue_b::gemm::WeightKBlockNInteger<typename T::GemmCore>>) {
    stor = T::PrologueB::createStorage(N, K, BlkSize, QuantType, ScaleDtype, BTLA_DTYPE::BF16, IsAsym);
    if (shuffle_indice != nullptr) {
      T::PrologueB::enableShuffle(&stor);
      stor.assign(reinterpret_cast<int8_t*>(PackedBuf));
      T::PrologueB::setShuffleIndices(shuffle_indice, &stor, pth);
    } else {
      stor.assign(reinterpret_cast<int8_t*>(PackedBuf));
    }
  } else {
    (void)(shuffle_indice);
    stor = T::PrologueB::createStorage(N, K, BlkSize, QuantType, ScaleDtype);
    stor.assign(reinterpret_cast<int8_t*>(PackedBuf));
  }
  T::PrologueB::packQWeight(N, K, QData, ldb, Scales, IsAsym ? Zp : nullptr, &stor, pth);
}

template <template <class> class Wei_T>
bool BTLAGemmPackBLocal(void* PackedBuf, const int8_t* QData, const float* Scales, const int8_t* Zp, size_t N, size_t K,
                        size_t ldb, size_t BlkSize, BTLA_DTYPE QuantType, BTLA_DTYPE ScaleDtype, bool isAsym,
                        ne_comp_type CompType, int* shuffle_indice, void* ThreadPool) {
  GetCPUDevice();
  auto dtype_type = utils::bestla_dtype_type(QuantType);
  auto constexpr dtype_int = utils::bestla_dtype_type(BTLA_DTYPE::TypeInt);
  if (dtype_type != dtype_int) {
    return false;
  }
  switch (CompType) {
    case NE_COMP_INT8:
      if (dtype_type == dtype_int) {
        if (_cd->AMX_INT8() && BlkSize % tAMX_INT8_US_KBlock::KTILE == 0) {
          BTLAGemmPackBImpl<tLauncher_Int8_F32F32<tAMX_INT8_US_KBlock, Wei_T>>(
              PackedBuf, static_cast<int>(BlkSize), QData, Scales, Zp, static_cast<int>(N), static_cast<int>(K),
              QuantType, ScaleDtype, isAsym, static_cast<int>(ldb), shuffle_indice, ThreadPool);
          return true;
        }
        if (_cd->AVX512_VNNI() && BlkSize % tAVX512_VNNI_KBlock::KTILE == 0) {
          BTLAGemmPackBImpl<tLauncher_Int8_F32F32<tAVX512_VNNI_KBlock, Wei_T>>(
              PackedBuf, static_cast<int>(BlkSize), QData, Scales, Zp, static_cast<int>(N), static_cast<int>(K),
              QuantType, ScaleDtype, isAsym, static_cast<int>(ldb), shuffle_indice, ThreadPool);
          return true;
        }
        if (_cd->AVX512BW() && BlkSize % tAVX512BW_KBlock::KTILE == 0) {
          BTLAGemmPackBImpl<tLauncher_Int8_F32F32<tAVX512BW_KBlock, Wei_T>>(
              PackedBuf, static_cast<int>(BlkSize), QData, Scales, Zp, static_cast<int>(N), static_cast<int>(K),
              QuantType, ScaleDtype, isAsym, static_cast<int>(ldb), shuffle_indice, ThreadPool);
          return true;
        }
        if (_cd->AVX_VNNI() && BlkSize % tAVX_VNNI_KBlock::KTILE == 0) {
          BTLAGemmPackBImpl<tLauncher_Int8_F32F32<tAVX_VNNI_KBlock, Wei_T>>(
              PackedBuf, static_cast<int>(BlkSize), QData, Scales, Zp, static_cast<int>(N), static_cast<int>(K),
              QuantType, ScaleDtype, isAsym, static_cast<int>(ldb), shuffle_indice, ThreadPool);
          return true;
        }
        if (_cd->AVX2() && BlkSize % tAVX2_VNNI_KBlock::KTILE == 0) {
          BTLAGemmPackBImpl<tLauncher_Int8_F32F32<tAVX2_VNNI_KBlock, Wei_T>>(
              PackedBuf, static_cast<int>(BlkSize), QData, Scales, Zp, static_cast<int>(N), static_cast<int>(K),
              QuantType, ScaleDtype, isAsym, static_cast<int>(ldb), shuffle_indice, ThreadPool);
          return true;
        }
      }
      [[fallthrough]];
    case NE_COMP_F16:
    case NE_COMP_BF16:
      if (_cd->AMX_BF16() && BlkSize % tAMX_BF16::KTILE == 0) {
        BTLAGemmPackBImpl<tLauncher_Fp_F32F32<tAMX_BF16, Wei_T>>(
            PackedBuf, static_cast<int>(BlkSize), QData, Scales, Zp, static_cast<int>(N), static_cast<int>(K),
            QuantType, ScaleDtype, isAsym, static_cast<int>(ldb), shuffle_indice, ThreadPool);
        return true;
      }
      [[fallthrough]];
    case NE_COMP_F32:
    case NE_COMP_UNDEF:  // currently only f32 activation
      if (_cd->AVX512F() && BlkSize % tAVX512F::KTILE == 0) {
        BTLAGemmPackBImpl<tLauncher_Fp_F32F32<tAVX512F, Wei_T>>(
            PackedBuf, static_cast<int>(BlkSize), QData, Scales, Zp, static_cast<int>(N), static_cast<int>(K),
            QuantType, ScaleDtype, isAsym, static_cast<int>(ldb), shuffle_indice, ThreadPool);
        return true;
      }
      if (_cd->AVX2() && BlkSize % tAVX2::KTILE == 0) {
        BTLAGemmPackBImpl<tLauncher_Fp_F32F32<tAVX2, Wei_T>>(
            PackedBuf, static_cast<int>(BlkSize), QData, Scales, Zp, static_cast<int>(N), static_cast<int>(K),
            QuantType, ScaleDtype, isAsym, static_cast<int>(ldb), shuffle_indice, ThreadPool);
        return true;
      }
      [[fallthrough]];
    default:
      return false;
  }
  return false;
}

}  // namespace

bool BTLAGemmBatchDriver(const size_t M, const size_t N, const size_t K, const size_t BatchN,
                         const BTLA_GEMM_DATA_PACKED_PARAMS* DataParams, int8_t* WorkSpace, void* ThreadPool) {
  GetCPUDevice();
  auto pth = reinterpret_cast<parallel::IThreading*>(ThreadPool);
  bool processed = true;
  for (size_t i = 0; i < BatchN; i++) {
    auto ptr = storage::gemm::PackedWeightParser::deserialBuffer(const_cast<void*>(DataParams[i].B));
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
            BTLAGemmCompF32<tAVX512F, tWeiNInt>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr, DataParams[i].C,
                                                DataParams[i].ldc, WorkSpace, pth);
          } else if (NTile == tAVX2::NTILE && _cd->AVX2() && BlkSize % tAVX2::KTILE == 0) {
            BTLAGemmCompF32<tAVX2, tWeiNInt>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr, DataParams[i].C,
                                             DataParams[i].ldc, WorkSpace, pth);
          }
        }
        if (btype == gemm::CompType::tBF16 && PackRow == 2) {
          if (NTile == tAMX_BF16::NTILE && _cd->AMX_BF16() && BlkSize % tAMX_BF16::KTILE == 0) {
            if (M <= tAVX512_BF16::MTILE) {
              static_assert(tAVX512_BF16::NTILE == tAMX_BF16::NTILE);
              BTLAGemmCompF32<tAVX512_BF16, tWeiNInt>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr, DataParams[i].C,
                                                      DataParams[i].ldc, WorkSpace, pth);
            } else {
              BTLAGemmCompF32<tAMX_BF16, tWeiNInt>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr, DataParams[i].C,
                                                   DataParams[i].ldc, WorkSpace, pth);
            }
          }
        }
        if (btype == gemm::CompType::tS8 && PackRow == 4) {
          if (NTile == tAMX_INT8_US_KBlock::NTILE && _cd->AMX_INT8() && BlkSize % tAMX_INT8_US_KBlock::KTILE == 0) {
            if (bptr->mBlockSize < K) {
              BTLAGemmCompInt8<tAMX_INT8_US_KBlock, tWeiNInt>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr,
                                                              DataParams[i].C, DataParams[i].ldc, WorkSpace, pth);
            } else {
              BTLAGemmCompInt8Pc<tAMX_INT8_US, tWeiNInt>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr,
                                                         DataParams[i].C, DataParams[i].ldc, WorkSpace, pth);
            }

          } else if (NTile == tAVX512_VNNI_KBlock::NTILE && _cd->AVX512_VNNI() &&
                     BlkSize % tAVX512_VNNI_KBlock::KTILE == 0) {
            if (bptr->mBlockSize < K) {
              BTLAGemmCompInt8<tAVX512_VNNI_KBlock, tWeiNInt>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr,
                                                              DataParams[i].C, DataParams[i].ldc, WorkSpace, pth);
            } else {
              BTLAGemmCompInt8Pc<tAVX512_VNNI, tWeiNInt>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr,
                                                         DataParams[i].C, DataParams[i].ldc, WorkSpace, pth);
            }
          } else if (NTile == tAVX512BW_KBlock::NTILE && _cd->AVX512BW() && BlkSize % tAVX512BW_KBlock::KTILE == 0) {
            if (bptr->mBlockSize < K) {
              BTLAGemmCompInt8<tAVX512BW_KBlock, tWeiNInt>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr,
                                                           DataParams[i].C, DataParams[i].ldc, WorkSpace, pth);
            } else {
              BTLAGemmCompInt8Pc<tAVX512BW, tWeiNInt>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr, DataParams[i].C,
                                                      DataParams[i].ldc, WorkSpace, pth);
            }

          } else if (NTile == tAVX_VNNI_KBlock::NTILE && _cd->AVX_VNNI() && BlkSize % tAVX_VNNI_KBlock::KTILE == 0) {
            if (bptr->mBlockSize < K) {
              BTLAGemmCompInt8<tAVX_VNNI_KBlock, tWeiNInt>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr,
                                                           DataParams[i].C, DataParams[i].ldc, WorkSpace, pth);
            } else {
              BTLAGemmCompInt8Pc<tAVX_VNNI, tWeiNInt>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr, DataParams[i].C,
                                                      DataParams[i].ldc, WorkSpace, pth);
            }
          } else if (NTile == tAVX2_VNNI_KBlock::NTILE && _cd->AVX2() && BlkSize % tAVX2_VNNI_KBlock::KTILE == 0) {
            if (bptr->mBlockSize < K) {
              BTLAGemmCompInt8<tAVX2_VNNI_KBlock, tWeiNInt>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr,
                                                            DataParams[i].C, DataParams[i].ldc, WorkSpace, pth);
            } else {
              BTLAGemmCompInt8Pc<tAVX2_VNNI, tWeiNInt>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr,
                                                       DataParams[i].C, DataParams[i].ldc, WorkSpace, pth);
            }
          }
        }
      }
      if (ptr->mPrologueID == BTLA_PROLOGUEB_IDS::WeightKBlockNFloat) {
        auto bptr = reinterpret_cast<storage::gemm::IWeightKBlockBase*>(ptr);
        auto BlkSize = bptr->mBlockSize;
        if (btype == gemm::CompType::tFP32 && PackRow == 1) {
          if (NTile == tAVX512F::NTILE && _cd->AVX512F() && BlkSize % tAVX512F::KTILE == 0) {
            BTLAGemmCompF32<tAVX512F, tWeiNFloat>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr, DataParams[i].C,
                                                  DataParams[i].ldc, WorkSpace, pth);
          } else if (NTile == tAVX2::NTILE && _cd->AVX2() && BlkSize % tAVX2::KTILE == 0) {
            BTLAGemmCompF32<tAVX2, tWeiNFloat>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr, DataParams[i].C,
                                               DataParams[i].ldc, WorkSpace, pth);
          }
        }
        if (btype == gemm::CompType::tBF16 && PackRow == 2) {
          if (NTile == tAMX_BF16::NTILE && _cd->AMX_BF16() && BlkSize % tAMX_BF16::KTILE == 0) {
            if (M <= tAVX512_BF16::MTILE) {
              static_assert(tAVX512_BF16::NTILE == tAMX_BF16::NTILE);
              BTLAGemmCompF32<tAVX512_BF16, tWeiNFloat>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr,
                                                        DataParams[i].C, DataParams[i].ldc, WorkSpace, pth);
            } else {
              BTLAGemmCompF32<tAMX_BF16, tWeiNFloat>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr, DataParams[i].C,
                                                     DataParams[i].ldc, WorkSpace, pth);
            }
          }
        }
      }
      delete ptr;
    } else {
      processed = false;
      break;
    }
  }
  return processed;
}

size_t BTLAGemmPackBSize(size_t N, size_t K, size_t BlkSize, BTLA_DTYPE QuantType, BTLA_DTYPE ScaleDtype, bool isAsym,
                         ne_comp_type CompType, int* shuffle_indice) {
  auto qtype = utils::bestla_dtype_type(QuantType);
  if (qtype == utils::bestla_dtype_type(BTLA_DTYPE::TypeInt)) {
    return BTLAGemmPackBSizeLocal<prologue_b::gemm::WeightKBlockNInteger>(N, K, BlkSize, QuantType, ScaleDtype, isAsym,
                                                                          CompType, shuffle_indice);
  } else if (qtype == utils::bestla_dtype_type(BTLA_DTYPE::TypeFloat)) {
    return BTLAGemmPackBSizeLocal<prologue_b::gemm::WeightKBlockNFloat>(N, K, BlkSize, QuantType, ScaleDtype, isAsym,
                                                                        CompType, shuffle_indice);
  } else {
    assert(0);
  }
  return 0;
}

bool BTLAGemmQuantPackB(void* PackedBuf, const float* FpData, size_t N, size_t K, size_t ldb, size_t BlkSize,
                        BTLA_DTYPE QuantType, BTLA_DTYPE ScaleDtype, bool isAsym, ne_comp_type CompType, bool isTrans,
                        void* ThreadPool) {
  auto qtype = utils::bestla_dtype_type(QuantType);
  if (qtype == utils::bestla_dtype_type(BTLA_DTYPE::TypeInt)) {
    return BTLAGemmQuantPackBLocal<prologue_b::gemm::WeightKBlockNInteger>(
        PackedBuf, FpData, N, K, ldb, BlkSize, QuantType, ScaleDtype, isAsym, CompType, isTrans, ThreadPool);
  } else if (qtype == utils::bestla_dtype_type(BTLA_DTYPE::TypeFloat)) {
    return BTLAGemmQuantPackBLocal<prologue_b::gemm::WeightKBlockNFloat>(
        PackedBuf, FpData, N, K, ldb, BlkSize, QuantType, ScaleDtype, isAsym, CompType, isTrans, ThreadPool);
  } else {
    assert(0);
    return false;
  }
}

bool BTLAGemmPackB(void* PackedBuf, const int8_t* QData, const float* Scales, const int8_t* Zp, size_t N, size_t K,
                   size_t ldb, size_t BlkSize, BTLA_DTYPE QuantType, BTLA_DTYPE ScaleDtype, bool isAsym,
                   ne_comp_type CompType, int* shuffle_indice, void* ThreadPool) {
  auto qtype = utils::bestla_dtype_type(QuantType);
  if (qtype == utils::bestla_dtype_type(BTLA_DTYPE::TypeInt)) {
    return BTLAGemmPackBLocal<prologue_b::gemm::WeightKBlockNInteger>(PackedBuf, QData, Scales, Zp, N, K, ldb, BlkSize,
                                                                      QuantType, ScaleDtype, isAsym, CompType,
                                                                      shuffle_indice, ThreadPool);
  } else if (qtype == utils::bestla_dtype_type(BTLA_DTYPE::TypeFloat)) {
    assert(0);
  } else {
    assert(0);
  }
  return false;
}

bool BTLAGemmUnPackB(float* FpData, const void* PackedBuf, size_t N, size_t K, size_t ldb, void* ThreadPool) {
  auto ptr = storage::gemm::PackedWeightParser::deserialBuffer(const_cast<void*>(PackedBuf));
  auto pth = reinterpret_cast<parallel::IThreading*>(ThreadPool);
  GetCPUDevice();
  if (ptr) {
    auto NTile = gemm::CoreAttr::get_mask_val(ptr->mCoreId, gemm::CoreAttr::NTILE_MASK, gemm::CoreAttr::NTILE_SHIFT);
    auto PackRow = gemm::CoreAttr::get_packrow(ptr->mCoreId);
    auto CType = gemm::CoreAttr::get_comp(ptr->mCoreId);
    auto btype = static_cast<gemm::CompType>(gemm::CompTypeHelper::get_B(CType));
    if (ptr->mPrologueID == BTLA_PROLOGUEB_IDS::WeightKBlockNInteger) {
      auto sptr = reinterpret_cast<storage::gemm::StorageWeightKBlockNInteger*>(ptr);
      if (btype == gemm::CompType::tFP32 && PackRow == 1) {
        if (NTile == tAVX512F::NTILE && _cd->AVX512F()) {
          static prologue_b::gemm::WeightKBlockNInteger<tAVX512F> proB;
          proB.unpackWeight(static_cast<int>(N), static_cast<int>(K), sptr, FpData, static_cast<int>(ldb), pth);
        } else if (NTile == tAVX2::NTILE && _cd->AVX2()) {
          static prologue_b::gemm::WeightKBlockNInteger<tAVX2> proB;
          proB.unpackWeight(static_cast<int>(N), static_cast<int>(K), sptr, FpData, static_cast<int>(ldb), pth);
        }
      }
      if (btype == gemm::CompType::tS8 && PackRow == 4) {
        if (NTile == tAMX_INT8_US_KBlock::NTILE && _cd->AMX_INT8()) {
          static prologue_b::gemm::WeightKBlockNInteger<tAMX_INT8_US_KBlock> proB;
          proB.unpackWeight(static_cast<int>(N), static_cast<int>(K), sptr, FpData, static_cast<int>(ldb), pth);
        } else if (NTile == tAVX512_VNNI_KBlock::NTILE && _cd->AVX512_VNNI()) {
          static prologue_b::gemm::WeightKBlockNInteger<tAVX512_VNNI_KBlock> proB;
          proB.unpackWeight(static_cast<int>(N), static_cast<int>(K), sptr, FpData, static_cast<int>(ldb), pth);
        } else if (NTile == tAVX512BW_KBlock::NTILE && _cd->AVX512BW()) {
          static prologue_b::gemm::WeightKBlockNInteger<tAVX512BW_KBlock> proB;
          proB.unpackWeight(static_cast<int>(N), static_cast<int>(K), sptr, FpData, static_cast<int>(ldb), pth);
        } else if (NTile == tAVX_VNNI_KBlock::NTILE && _cd->AVX_VNNI()) {
          static prologue_b::gemm::WeightKBlockNInteger<tAVX_VNNI_KBlock> proB;
          proB.unpackWeight(static_cast<int>(N), static_cast<int>(K), sptr, FpData, static_cast<int>(ldb), pth);
        }
      }
      if (btype == gemm::CompType::tBF16 && PackRow == 2) {
        if (NTile == tAMX_BF16::NTILE && _cd->AMX_BF16()) {
          static prologue_b::gemm::WeightKBlockNInteger<tAMX_BF16> proB;
          proB.unpackWeight(static_cast<int>(N), static_cast<int>(K), sptr, FpData, static_cast<int>(ldb), pth);
        }
      }
    }
    if (ptr->mPrologueID == BTLA_PROLOGUEB_IDS::WeightKBlockNFloat) {
      auto sptr = reinterpret_cast<storage::gemm::StorageWeightKBlockNFloat*>(ptr);
      if (btype == gemm::CompType::tFP32 && PackRow == 1) {
        if (NTile == tAVX512F::NTILE && _cd->AVX512F()) {
          static prologue_b::gemm::WeightKBlockNFloat<tAVX512F> proB;
          proB.unpackWeight(static_cast<int>(N), static_cast<int>(K), sptr, FpData, static_cast<int>(ldb), pth);
        } else if (NTile == tAVX2::NTILE && _cd->AVX2()) {
          static prologue_b::gemm::WeightKBlockNFloat<tAVX2> proB;
          proB.unpackWeight(static_cast<int>(N), static_cast<int>(K), sptr, FpData, static_cast<int>(ldb), pth);
        }
      }
      if (btype == gemm::CompType::tBF16 && PackRow == 2) {
        if (NTile == tAMX_BF16::NTILE && _cd->AMX_BF16()) {
          static prologue_b::gemm::WeightKBlockNFloat<tAMX_BF16> proB;
          proB.unpackWeight(static_cast<int>(N), static_cast<int>(K), sptr, FpData, static_cast<int>(ldb), pth);
        }
      }
    }
    delete ptr;
    return true;
  }
  return false;
}

bool BTLALayerNorm(size_t norm_count, size_t norm_size, bool isrms, float epsilon, const float* FpIn, float* FpOut,
                   void* ThreadPool) {
  auto inorm_count = static_cast<int>(norm_count);
  auto inorm_size = static_cast<int>(norm_size);
  auto pth = reinterpret_cast<parallel::IThreading*>(ThreadPool);
  int threads = inorm_count <= 4 ? 1 : pth->num_threads();
  parallel::Scheduler2D sch({threads, inorm_count, inorm_size, 1, inorm_size});
  auto threadfunc = [&](int tidx) {
    parallel::ThreadProblem2D tp{tidx};
    sch.getIndex(tp);
    if (tp.valid) {
      for (size_t i = 0; i < tp.size[0]; i++) {
        auto srcptr = FpIn + (tp.loc[0] + i) * inorm_size;
        auto dstptr = FpOut + (tp.loc[0] + i) * inorm_size;
        auto ret = kernel::wrapper::LayerNormalization::forward_auto<float>(
            srcptr, nullptr, nullptr, epsilon, inorm_size, dstptr, nullptr, nullptr, isrms);
      }
    }
  };
  if (threads == 1) {
    parallel::SingleThread st;
    st.parallel_for(threadfunc);
  } else {
    pth->parallel_for(threadfunc);
  }
  return true;
}
