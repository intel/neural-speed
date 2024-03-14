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

/// @file
/// C++ API

#pragma once

#include <group/epilogue/common.hpp>

namespace gpu::xetla::group {

/// @addtogroup xetla_epilogue
/// @{

/// @brief Default epilogue policy for store C.
/// @tparam arch_tag_ Is the HW architecture.
template <gpu_arch arch_tag_>
struct epilogue_policy_default {
  static constexpr gpu_arch arch_tag = arch_tag_;
};

/// @brief Epilogue policy for tile_op + store C fusion.
/// @tparam tile_op_t_ Is the tile_op functor.
/// @tparam arch_tag_ Is the HW architecture.
template <typename tile_op_t_, gpu_arch arch_tag_>
struct epilogue_policy_tile_op {
  using tile_op_t = tile_op_t_;
  static constexpr gpu_arch arch_tag = arch_tag_;
};

/// @brief Epilogue functor, specialized for quantization operator.
/// @tparam dequant_op_t_ is the dequantization op type
/// @tparam tile_op_t_ is the tile op type.
/// @tparam quant_op_t_ is the quantization op type.
/// @tparam arch_tag_ Is the HW architecture.
/// @tparam dequant_dtype_t_ Is the dequantize data type.
template <
    typename dequant_op_t_,
    typename tile_op_t_,
    typename quant_op_t_,
    gpu_arch arch_tag_,
    typename dtype_dequant_ = float>
struct epilogue_policy_quant_op {
  using dequant_op_t = dequant_op_t_;
  using tile_op_t = tile_op_t_;
  using quant_op_t = quant_op_t_;
  static constexpr gpu_arch arch_tag = arch_tag_;
  using dtype_dequant = dtype_dequant_;
};

/// @brief Epilogue policy for store unaligned C.
/// @tparam update_method_ Is the store method of matC.
/// @tparam arch_ Is the HW architecture.
template <gpu_arch arch_tag_>
struct epilogue_policy_unaligned {
  static constexpr gpu_arch arch_tag = arch_tag_;
};
/// @} xetla_epilogue

} // namespace gpu::xetla::group
