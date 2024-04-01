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
namespace sycl_prologue_b {

template <typename SrcT>
struct ParamWeightBase {
  const SrcT* B;
  int ldb;
};
template <class _GemmCore_T, typename SrcT>
class WeightBase {
 public:
  using BType = typename _GemmCore_T::TB;
  using SRCType = SrcT;
  using Param = ParamWeightBase<SRCType>;
  template <int UnrollK>
  BTLA_CODE getWeight(const Param& _param, BType* dstptr, size_t _srcoffset, int sgId) {
    auto ptr = &_param.B[_srcoffset];
    auto ld = _param.ldb;
    int constexpr Iter_PerWorker = (_GemmCore_T::TileK + _GemmCore_T::WgM - 1) / _GemmCore_T::WgM;
#pragma unroll
    for (int icp = 0; icp < Iter_PerWorker; icp++) {
      {
        for (size_t in = 0; in < _GemmCore_T::TileN; in++) {
          dstptr[(sg_idxm + icp * _GemmCore_T::WgM) * _GemmCore_T::WgNEle +
              (sg_idxn * _GemmCore_T::SgSize + sgId) * _GemmCore_T::TileN + in] =
              ptr[tn + in + (i + sg_idxm + icp * _GemmCore_T::WgM) * n];
        }
      }
    }
  }
};

}  // namespace sycl_prologue_b
}  // namespace bestla
#endif
