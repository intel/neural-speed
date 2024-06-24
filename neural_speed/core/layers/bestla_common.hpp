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
#include <utility>
#include "ne_bestla.h"
#include "bestla/bestla_prologue_b.h"
#include "bestla/bestla_device.h"
#include "bestla/bestla_utils.h"
#include "bestla/bestla_parallel.h"
#include "layers/ele_wise.h"
#include "bestla_defs.h"

namespace ne_bestla {

class ne_threading {
 public:
  static bestla::parallel::IThreading* get() {
    GetCPUDevice();
    static bestla::parallel::StdThreading OptmizedThreading;
#if (BTLA_OPENMP && NS_USE_OMP)
    static bestla::parallel::OMPThreading DefaultThreading;
#ifdef NS_SYCL
    return &DefaultThreading;
#endif
    if (!_cd->isHybrid()) {
      return &DefaultThreading;
    }
#endif  // _OPNEMP
    return &OptmizedThreading;
  }

  static void set_threads(int n_thread) { get()->set_threads(n_thread); }
};

template <typename T>
static inline void safe_delete(T* ptr) {
  if (ptr) {
    delete ptr;
  }
}

template <typename T>
static bool contains(const T& val, const T* set, size_t len) {
  for (size_t i = 0; i < len; i++) {
    if (val == set[i]) {
      return true;
    }
  }
  return false;
}

static bool hasISA(const uint64_t* coreset, size_t len) {
  GetCPUDevice();
  bool support = false;
  for (size_t i = 0; i < len; i++) {
    auto isa = bestla::gemm::CoreAttr::get_ISA(coreset[i]);
    switch (isa) {
      case BTLA_ISA::AVX:
        support |= _cd->AVX();
        break;
      case BTLA_ISA::AVX2:
        support |= _cd->AVX2();
        break;
      case BTLA_ISA::AMX_BF16:
        support |= _cd->AMX_BF16();
        break;
      case BTLA_ISA::AMX_INT8:
        support |= _cd->AMX_INT8();
        break;
      case BTLA_ISA::AVX512F:
        support |= _cd->AVX512F();
        break;
      case BTLA_ISA::AVX512BW:
        support |= _cd->AVX512BW();
        break;
      case BTLA_ISA::AVX512_VNNI:
        support |= _cd->AVX512_VNNI();
        break;
      case BTLA_ISA::AVX512_FP16:
        support |= _cd->AVX512_FP16();
        break;
      case BTLA_ISA::AVX_VNNI:
        support |= _cd->AVX_VNNI();
        break;
      default:
        break;
    }
    if (support) {
      break;
    }
  }
  return support;
}

static inline bool samePackedWeight(bestla::storage::gemm::IWeightBase* ptr0,
                                    bestla::storage::gemm::IWeightBase* ptr1) {
  return ptr0->mCoreId == ptr1->mCoreId && ptr0->mPrologueID == ptr1->mPrologueID;
}

static inline bool samePackedWeight(bestla::storage::gemm::IWeightBase** ptrs, size_t len) {
  bool sameKernel = samePackedWeight(ptrs[0], ptrs[1]);
  if (sameKernel) {
    for (size_t i = 2; i < len; i++) {
      sameKernel &= samePackedWeight(ptrs[0], ptrs[i]);
    }
  }
  return sameKernel;
}

namespace custom {
namespace epilogue {
template <typename _T>
struct ParamAdd {
  _T *C, *D;
  int ldc, ldd;
};
template <typename _T>
class Add {
 public:
  using Param = ParamAdd<_T>;

  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE forward(const float* cacheptr, const int cachestep, const int M_offset, const int N_offset,
                                  const int M, const int N, const Param& _param, void* tmpcache, size_t cachesize) {
    auto COffset = M_offset * _param.ldc + N_offset;
    auto DOffset = M_offset * _param.ldd + N_offset;
    auto cptr = _param.C + COffset;
    auto dptr = _param.D + DOffset;
    if constexpr (std::is_same_v<_T, float>) {
      for (int i = 0; i < M; i++) {
        bestla::kernel::wrapper::Add<_T>::template forward<ISA_T>(dptr + i * _param.ldd, cacheptr + i * cachestep,
                                                                  cptr + i * _param.ldc, N);
      }
      return BTLA_CODE::Success;
    } else {
      return BTLA_CODE::NotSupport;
    }
  }
};
using AddFp32 = Add<float>;

template <typename _T>
struct ParamMul {
  _T *C, *D;
  int ldc, ldd;
};
template <typename _T>
class Mul {
 public:
  using Param = ParamMul<_T>;
  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE forward(const float* cacheptr, const int cachestep, const int M_offset, const int N_offset,
                                  const int M, const int N, const Param& _param, void* tmpcache, size_t cachesize) {
    auto COffset = M_offset * _param.ldc + N_offset;
    auto DOffset = M_offset * _param.ldd + N_offset;
    auto cptr = _param.C + COffset;
    auto dptr = _param.D + DOffset;
    if constexpr (std::is_same_v<_T, float>) {
      for (int i = 0; i < M; i++) {
        bestla::kernel::wrapper::Mul<_T>::template forward<ISA_T>(dptr + i * _param.ldd, cacheptr + i * cachestep,
                                                                  cptr + i * _param.ldc, N);
      }
      return BTLA_CODE::Success;
    } else {
      return BTLA_CODE::NotSupport;
    }
  }
};
using MulFp32 = Mul<float>;

template <typename _T>
struct ParamAdd_Gelu {
  _T *C, *D;
  int ldc, ldd;
};
template <typename _T>
class Add_Gelu {
 public:
  using Param = ParamAdd_Gelu<_T>;
  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE forward(  // NOLINT [build/include_what_you_use]
      const float* cacheptr, const int cachestep, const int M_offset, const int N_offset, const int M, const int N,
      const Param& _param, void* tmpcache, size_t cachesize) {
    auto COffset = M_offset * _param.ldc + N_offset;
    auto DOffset = M_offset * _param.ldd + N_offset;
    auto cptr = _param.C + COffset;
    auto dptr = _param.D + DOffset;
    if constexpr (std::is_same_v<_T, float>) {
      for (int i = 0; i < M; i++) {
        bestla::kernel::wrapper::Add<_T>::template forward<ISA_T>(dptr + i * _param.ldd, cacheptr + i * cachestep,
                                                                  cptr + i * _param.ldc, N);
      }
      using GeluKernel = bestla::epilogue::gemm::AccumulatorWriteBackWithGeluFp32;
      typename GeluKernel::Param param{_param.C, _param.ldc, nullptr};
      auto ret = GeluKernel::forward<ISA_T>(cptr, _param.ldc, M_offset, N_offset, M, N, param, tmpcache, cachesize);
      return ret;
    } else {
      return BTLA_CODE::NotSupport;
    }
  }
};
using Add_GeluFp32 = Add_Gelu<float>;

}  // namespace epilogue
}  // namespace custom
}  // namespace ne_bestla
