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
template <class GemmCoreT, typename SrcT>
class WeightBase {
 public:
  using BType = typename GemmCoreT::TB;
  using SRCType = SrcT;
  using Param = ParamWeightBase<SRCType>;

  static inline void getWeight(const Param& _param, const sycl::local_accessor<BType, 1>& dstptr, int koffset,
                               sycl_utils::nd_item_helper<GemmCoreT>& helper) {
    int constexpr Iter_PerWorker = (GemmCoreT::TileK + GemmCoreT::WgM - 1) / GemmCoreT::WgM;
#pragma unroll
    for (int icp = 0; icp < Iter_PerWorker; icp++) {
      {
        for (size_t in = 0; in < GemmCoreT::TileN; in++) {
          dstptr[(helper.sg_idx_m() + icp * GemmCoreT::WgM) * GemmCoreT::WgNEle +
                 (helper.sg_idx_n() * GemmCoreT::SgSize + helper.sg_id()) * GemmCoreT::TileN + in] =
              _param.B[helper.item_g_n() + in + (koffset + helper.sg_idx_m() + icp * GemmCoreT::WgM) * _param.ldb];
        }
      }
    }
  }
};

template <typename ScaleT>
struct ParamWeightS4 {
  const uint8_t* B;
  const ScaleT* scale;
  int ldb;
};
template <class GemmCoreT, typename ScaleT>
class WeightS4 {
 public:
  using BType = typename GemmCoreT::TB;
  using Param = ParamWeightS4<ScaleT>;

  static inline void getWeight(const Param& _param, const sycl::local_accessor<BType, 1>& dstptr, int koffset,
                               int blocksize, sycl_utils::nd_item_helper<GemmCoreT>& helper) {
    int constexpr Iter_PerWorker = (GemmCoreT::TileK + GemmCoreT::WgM - 1) / GemmCoreT::WgM;
    float scale[GemmCoreT::TileN];
    for (size_t in = 0; in < GemmCoreT::TileN; in += 1)
      scale[in] = _param.scale[helper.item_g_n() + in + koffset / blocksize * _param.ldb];
#pragma unroll
    for (int icp = 0; icp < Iter_PerWorker; icp++) {
      {
        for (size_t in = 0; in < GemmCoreT::TileN; in += 2) {
          auto tmps8 =
              _param
                  .B[(helper.item_g_n() + in + (koffset + helper.sg_idx_m() + icp * GemmCoreT::WgM) * _param.ldb) / 2];
          dstptr[(helper.sg_idx_m() + icp * GemmCoreT::WgM) * GemmCoreT::WgNEle +
                 (helper.sg_idx_n() * GemmCoreT::SgSize + helper.sg_id()) * GemmCoreT::TileN + in] =
              static_cast<int8_t>((tmps8 & 0x0f) << 4) * scale[in];
          dstptr[(helper.sg_idx_m() + icp * GemmCoreT::WgM) * GemmCoreT::WgNEle +
                 (helper.sg_idx_n() * GemmCoreT::SgSize + helper.sg_id()) * GemmCoreT::TileN + in + 1] =
              static_cast<int8_t>((tmps8 & 0xf0)) * scale[in + 1];
        }
      }
    }
  }
};

}  // namespace sycl_prologue_b
}  // namespace bestla
#endif
