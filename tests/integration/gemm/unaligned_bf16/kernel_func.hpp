/*******************************************************************************
 * Copyright (c) 2022-2023 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#pragma once

#include "xetla.hpp"

using namespace gpu::xetla;
using namespace gpu::xetla::group;
using namespace gpu::xetla::kernel;
using namespace gpu::xetla::subgroup;

template <
    typename dtype_a,
    typename dtype_b,
    typename dtype_c,
    typename dtype_acc,
    uint32_t wg_m,
    uint32_t wg_n,
    uint32_t sg_m,
    uint32_t sg_n,
    uint32_t sg_k,
    mem_layout layout_a,
    mem_layout layout_b,
    uint32_t global_kslicing,
    uint32_t local_kslicing,
    mma_engine engine>
struct unaligned_gemm_test_func {
  using tile_shape = tile_shape_t<wg_n, wg_m, sg_n, sg_m>;
  static constexpr uint32_t periodic_sync_interval = 8;
  static constexpr uint32_t prefetch_distance = 3;
  using gemm_t = typename gemm_selector_t<
      dtype_a,
      dtype_b,
      layout_a,
      layout_b,
      mem_space::global,
      mem_space::global,
      1,
      1,
      dtype_acc,
      tile_shape,
      sg_k,
      engine,
      gpu_arch::XeHpc,
      prefetch_distance,
      periodic_sync_interval>::gemm;

  using epilogue_t = epilogue_t<
      epilogue_policy_unaligned<gpu_arch::XeHpc>,
      tile_shape,
      mem_desc_t<dtype_c, mem_layout::row_major, mem_space::global, 1>>;

  using group_swizzle =
      gpu::xetla::kernel::group_swizzle_default<gpu_arch::XeHpc>;
  using dispatch_policy =
      dispatch_policy_kslicing<group_swizzle, global_kslicing, local_kslicing>;
  using gemm_op_t = gemm_universal_t<dispatch_policy, gemm_t, epilogue_t>;

  static const char* func_name() {
    return "unaligned_gemm_test_func";
  }

  static inline void run(
      sycl::nd_item<3>& item,
      dtype_a* A,
      dtype_b* B,
      dtype_c* C,
      uint32_t mat_m,
      uint32_t mat_n,
      uint32_t mat_k,
      dtype_acc* Acc,
      uint32_t* Cnt) {
    typename gemm_op_t::arguments_t arg(
        mat_m,
        mat_k,
        mat_n,
        A,
        layout_a == mem_layout::col_major ? mat_m : mat_k,
        B,
        layout_b == mem_layout::col_major ? mat_k : mat_n,
        C,
        mat_n,
        Acc,
        Cnt);
    gemm_op_t gemm_op;
    gemm_op(item, arg);
  }
};
