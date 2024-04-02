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
#include <sycl/sycl.hpp>

#include "bestla_utils.h"
#include "sycl_utils.h"
#include "sycl_device.h"
#include "sycl_gemm.h"
#include "sycl_epilogue.h"
#include "sycl_prologue_a.h"
#include "sycl_prologue_b.h"

namespace bestla {
namespace sycl_wrapper {
template <template <class GCT> class ProAT, template <class GCT> class ProBT, template <class GCT> class EpiT,
          class GemmCoreT>
class Launcher {
 public:
  using GemmCore = GemmCoreT;
  using PrologueA = ProAT<GemmCore>;
  using PrologueB = ProBT<GemmCore>;
  using Epilogue = EpiT<GemmCore>;
  using AType = typename GemmCore::TA;
  using AParam = typename PrologueA::Param;
  using BType = typename GemmCore::TB;
  using BParam = typename PrologueB::Param;
  using CType = typename GemmCore::TC;
  using EpiParam = typename Epilogue::Param;
  struct Param {
    const int m, n, k;
    const AParam paramA;
    const BParam paramB;
    const EpiParam paramC;
  };
  static inline sycl::event compute(const Param& _param, sycl::queue* q) {
    sycl::range<2> group{GemmCore::WgM, GemmCore::WgN};
    int k = _param.k;
    auto A = _param.paramA.A;
    auto B = _param.paramB.B;
    auto C = _param.paramC.C;
    int lda = _param.paramA.lda;
    int ldb = _param.paramB.ldb;
    int ldc = _param.paramC.ldc;
    sycl::range<2> problem{_param.m / GemmCore::TileM, _param.n / GemmCore::TileN};
    auto ev = q->submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> slm_b(sycl::range(GemmCore::SLM_B_Size), cgh);
      sycl::local_accessor<float, 1> slm_a(sycl::range(GemmCore::SLM_A_Size), cgh);
      cgh.parallel_for(
          sycl::nd_range<2>(problem, group),
          [=](sycl::nd_item<2> it) [[cl::reqd_work_group_size(
              1, GemmCore::WgM,
              GemmCore::WgN)]] [[intel::kernel_args_restrict]] [[intel::reqd_sub_group_size(GemmCore::SgSize)]] {
            nd_item_helper<GemmCore> helper(it);
            float tmp[GemmCore::TileM * GemmCore::TileN];
            for (size_t im = 0; im < GemmCore::TileM; im++)
              for (size_t in = 0; in < GemmCore::TileN; in++) tmp[im * GemmCore::TileN + in] = 0.f;

#pragma forceinline recursive
            for (int i = 0; i < k; i += GemmCore::TileK) {
              PrologueB::getWeight({B, ldb}, slm_b, i, helper);
              it.barrier(sycl::access::fence_space::local_space);
              GemmCore::compute(&A[helper.item_g_m() * k + i], k, slm_b, tmp, helper);
              it.barrier(sycl::access::fence_space::local_space);
            }
#pragma forceinline recursive
            Epilogue::store(_param.paramC, tmp, helper);
          });
    });
    return ev;
  }
};
}  // namespace sycl_wrapper
}  // namespace bestla
#endif
