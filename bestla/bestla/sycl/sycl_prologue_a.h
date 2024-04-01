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

#ifdef BTLA_SYCL
#include <array>

#include "bestla_utils.h"
#include <sycl/sycl.hpp>

namespace bestla {
namespace sycl_prologue_a {

template <typename SrcT>
struct ParamActivationBase {
  const SrcT* A;
  int lda;
};
template <class _GemmCore_T, typename SrcT>
class ActivationBase {
 public:
  using AType = typename _GemmCore_T::AType;
  using SRCType = SrcT;
  using Param = ParamActivationBase<SRCType>;
  BTLA_CODE getActivation(AType** dstptr, int* dststep, const Param& _param, int m_size, int k_size, int m_offset,
                          int k_offset, void* tmpcache, size_t cachesize) {
    auto aptr = const_cast<AType*>(_param.A) + m_offset * _param.lda + k_offset;
    auto alignedptr = utils::cpu_pointer_align(aptr);
    bool use_rawptr = k_size % _GemmCore_T::KTILE == 0 && m_size >= _GemmCore_T::MTILE;
    use_rawptr = use_rawptr && (alignedptr == aptr);
    if (use_rawptr) {
      *dstptr = aptr;
      *dststep = _param.lda;
      return BTLA_CODE::Success;
    } else {
      auto k_pad = utils::padto(k_size, _GemmCore_T::KTILE);
      *dststep = k_pad;
      return kernel::wrapper::Memcpy2D::forward<BTLA_ISA::NoSIMD, AType, AType>(aptr, *dstptr, m_size, k_size,
                                                                                _param.lda, k_pad);
    }
  }
};

}  // namespace xve

}  // namespace sycl_gemm
}  // namespace bestla
#endif
