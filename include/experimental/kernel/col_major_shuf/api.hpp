/*******************************************************************************
 * Copyright (c) 2023-2024 Intel Corporation
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

#include <experimental/kernel/col_major_shuf/common.hpp>
#include <experimental/kernel/col_major_shuf/config.hpp>

namespace gpu::xetla::kernel {

/// @brief
///
/// @tparam dtype_in_
/// @tparam dtype_out_
/// @tparam dtype_gidx_
/// @tparam mem_layout_in_
/// @tparam col_major_shuf_attr_
/// @tparam arch_
template <
    typename dtype_in_,
    typename dtype_out_,
    typename dtype_gidx_,
    mem_layout mem_layout_in_,
    typename col_major_shuf_attr_,
    gpu_arch arch_,
    typename enable = void>
struct col_major_shuf_t {};

} // namespace gpu::xetla::kernel
