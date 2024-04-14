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

class KernelConfigBase {
 public:
  static int constexpr SgSize = 16;
  static int constexpr TileK = 16;
  static int constexpr TileN = 2;
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
    ScaleT scale[GemmCoreT::TileN];
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

  template <class KernelConfigBase>
  static inline sycl::event dequant_s4(int n, int k, int blocksize, const Param& in, BType* outptr, sycl::queue* q) {
    int constexpr SgSize = KernelConfigBase::SgSize;
    int constexpr TileK = KernelConfigBase::TileK;
    int constexpr TileN = KernelConfigBase::TileN;
    int constexpr GroupN = SgSize * TileN;
    int constexpr GroupK = TileK;
    static_assert(TileN % 2 == 0);
    assert(blocksize % TileK == 0);

    int nsg_k = k / GroupK;
    int nsg_n = n / GroupN;
    sycl::range<1> group{SgSize};
    sycl::range<1> problem{nsg_n * nsg_k * SgSize};
    auto B_d = in.B;
    auto S_d = in.scale;
    int ldb = in.ldb;
    auto deq_kernel = [&](sycl::handler& cgh) {
      cgh.parallel_for(sycl::nd_range<1>(problem, group),
                       [=](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(SgSize)]] {
                         int g_idx = it.get_group(0);
                         auto sg = it.get_sub_group();
                         int sg_id = sg.get_local_id()[0];
                         int g_idx_n = g_idx % nsg_n;
                         int g_idx_k = g_idx / nsg_n;
                         int g_n = g_idx_n * GroupN;
                         int g_k = g_idx_k * GroupK;
                         auto sptr = S_d + g_k / blocksize * ldb + g_n;
                         auto bptr = B_d + (g_k * ldb + g_n) / 2;
                         auto dbptr = outptr + g_k * n + g_n;
                         float tmp[TileK * TileN];
                         float scale[TileN];
                         for (int in = 0; in < TileN; in += 1) {
                           scale[in] = sptr[sg_id * TileN + in];
                         }
                         for (int ik = 0; ik < TileK; ik += 1) {
                           for (int in = 0; in < TileN; in += 2) {
                             uint8_t srcu8 = *(bptr + (ik * ldb + sg_id * TileN + in) / 2);
                             tmp[ik * TileN + in] = static_cast<int8_t>((srcu8 & 0x0f) << 4) * scale[in];
                             tmp[ik * TileN + in + 1] = static_cast<int8_t>((srcu8 & 0xf0)) * scale[in + 1];
                           }
                         }
                         for (int ik = 0; ik < TileK; ik += 1) {
                           for (int in = 0; in < TileN; in += 1) {
                             dbptr[ik * n + sg_id * TileN + in] = tmp[ik * TileN + in];
                           }
                         }
                       });
    };
    return q->submit(deq_kernel);
  }
};

class KernelConfigTrans {
 public:
  static int constexpr SgSize = 16;
  static int constexpr TileK = 16;
  static int constexpr TileN = 1;
};

template <class GemmCoreT, typename ScaleT>
class WeightS4Trans {
 public:
  using BType = typename GemmCoreT::TB;
  using Param = ParamWeightS4<ScaleT>;

  static inline void getWeight(const Param& _param, const sycl::local_accessor<BType, 1>& dstptr, int koffset,
                               int blocksize, sycl_utils::nd_item_helper<GemmCoreT>& helper) {
    int constexpr LoadTileK = 2;
    static_assert(GemmCoreT::TileK == (LoadTileK * GemmCoreT::SgSize));
    int constexpr Iter_PerWorker = GemmCoreT::WgNEle / GemmCoreT::SgCount;
    auto wldb = _param.ldb * blocksize;
    int sgn = helper.wg_g_n() + helper.sg_group_id();
    int sg_off = helper.sg_id() * LoadTileK * GemmCoreT::WgNEle;
#pragma unroll
    for (int icp = 0; icp < Iter_PerWorker; icp++) {
      {
        auto scale = _param.scale[(sgn + icp * GemmCoreT::SgCount) * _param.ldb + koffset / blocksize];
        auto tmps8 = _param.B[((sgn + icp * GemmCoreT::SgCount) * wldb + (koffset + helper.sg_id() * LoadTileK)) / 2];
        dstptr[sg_off + helper.sg_group_id() + icp * GemmCoreT::SgCount] =
            static_cast<int8_t>((tmps8 & 0x0f) << 4) * scale;
        dstptr[sg_off + GemmCoreT::WgNEle + helper.sg_group_id() + icp * GemmCoreT::SgCount] =
            static_cast<int8_t>((tmps8 & 0xf0)) * scale;
      }
    }
  }

  template <class KernelConfigBase>
  static inline sycl::event dequant_s4(int n, int k, int blocksize, const Param& in, BType* outptr, sycl::queue* q) {
    int constexpr SgSize = KernelConfigBase::SgSize;
    int constexpr TileK = KernelConfigBase::TileK;
    int constexpr TileN = KernelConfigBase::TileN;
    int constexpr GroupN = TileN;
    int constexpr GroupK = SgSize * TileK;
    static_assert(TileN == 1);
    assert(blocksize % TileK == 0);

    int nsg_k = k / GroupK;
    int nsg_n = n / GroupN;
    sycl::range<1> group{SgSize};
    sycl::range<1> problem{nsg_n * nsg_k * SgSize};
    auto B_d = in.B;
    auto S_d = in.scale;
    int ldb = in.ldb;
    int ldbn = in.ldb * blocksize;
    auto deq_kernel = [&](sycl::handler& cgh) {
      cgh.parallel_for(sycl::nd_range<1>(problem, group),
                       [=](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(SgSize)]] {
                         int g_idx = it.get_group(0);
                         auto sg = it.get_sub_group();
                         int sg_id = sg.get_local_id()[0];
                         int g_idx_n = g_idx / nsg_k;
                         int g_idx_k = g_idx % nsg_k;
                         int g_n = g_idx_n * GroupN;
                         int g_k = g_idx_k * GroupK;
                         auto sptr = S_d + g_k / blocksize + g_n * ldb;
                         auto bptr = B_d + (g_k + g_n * ldbn) / 2;
                         auto dbptr = outptr + g_k + g_n * k;
                         float tmp[TileK];
                         float scale = sptr[sg_id * TileK / blocksize];

                         for (int ik = 0; ik < TileK; ik += 2) {
                           uint8_t srcu8 = *(bptr + (ik + sg_id * TileK) / 2);
                           tmp[ik] = static_cast<int8_t>((srcu8 & 0x0f) << 4) * scale;
                           tmp[ik + 1] = static_cast<int8_t>((srcu8 & 0xf0)) * scale;
                         }

                         for (int ik = 0; ik < TileK; ik += 1) {
                           dbptr[ik + sg_id * TileK] = tmp[ik];
                         }
                       });
    };
    return q->submit(deq_kernel);
  }

  template <class NOTVALID>
  static inline sycl::event dequant_s4_trans(int n, int k, int blocksize, const Param& in, BType* outptr,
                                             sycl::queue* q) {
    int constexpr SgSize = 16;
    int constexpr TileK = 1;
    int constexpr TileN = 16;
    int constexpr GroupN = TileN;
    int constexpr GroupK = SgSize * TileK;
    assert(blocksize % TileK == 0);
    static_assert(TileN == SgSize);
    static_assert(TileK == 1);
    int nsg_k = k / GroupK;
    int nsg_n = n / GroupN;
    sycl::range<1> group{SgSize};
    sycl::range<1> problem{nsg_n * nsg_k * SgSize};
    auto B_d = in.B;
    auto S_d = in.scale;
    int ldb = in.ldb;
    int ldbn = in.ldb * blocksize;
    auto deq_kernel = [&](sycl::handler& cgh) {
      cgh.parallel_for(sycl::nd_range<1>(problem, group),
                       [=](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(SgSize)]] {
                         int g_idx = it.get_group(0);
                         auto sg = it.get_sub_group();
                         int sg_id = sg.get_local_id()[0];
                         int g_idx_n = g_idx / nsg_k;
                         int g_idx_k = g_idx % nsg_k;
                         int g_n = g_idx_n * GroupN;
                         int g_k = g_idx_k * GroupK;
                         auto sptr = S_d + g_k / blocksize + g_n * ldb;
                         auto bptr = B_d + (g_k + g_n * ldbn) / 2;
                         auto dbptr = outptr + g_k * n + g_n;
                         float tmp[TileN];
                         bool high4 = sg_id % 2 != 0;
                         for (int in = 0; in < TileN; in++) {
                           float scale = sptr[sg_id * TileK / blocksize + in * ldb];
                           uint8_t srcu8 = *(bptr + (sg_id * TileK + in * ldbn) / 2);
                           tmp[in] = high4 ? static_cast<int8_t>((srcu8 & 0xf0)) * scale
                                           : static_cast<int8_t>((srcu8 & 0x0f) << 4) * scale;
                         }

                         float tmpT[TileN];
                         for (int in = 0; in < TileN; in++) {
                           for (int is = 0; is < SgSize; is++) {
                             auto shlv = sg.shuffle(tmp[in], is);
                             if (sg_id == in) {
                               tmpT[is] = shlv;
                             }
                           }
                         }
                         for (int in = 0; in < TileN; in++) {
                           dbptr[sg_id + in * n] = tmpT[in];
                         }
                       });
    };
    return q->submit(deq_kernel);
  }
};
}  // namespace sycl_prologue_b
}  // namespace bestla
#endif
