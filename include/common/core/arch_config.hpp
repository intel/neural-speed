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

#include <common/core/common.hpp>

namespace gpu::xetla {

/// @addtogroup xetla_core_arch_config
/// @{

template <msg_type message_type, gpu_arch arch_tag>
struct load_store_attr_t {};
template <>
struct load_store_attr_t<msg_type::block_2d, gpu_arch::XeHpc> {
  /// HW limitation checks https://gfxspecs.intel.com/Predator/Home/Index/55490
  static constexpr uint32_t max_load_height_in_elem = 32;
  static constexpr uint32_t max_load_width_in_bytes = 64;
  static constexpr uint32_t max_trans_load_width_in_bytes = 32;
  static constexpr uint32_t max_vnni_load_width_in_elems = 16;
  static constexpr uint32_t min_vnni_load_height_in_bytes = 4;

  static constexpr uint32_t max_store_height_in_elem = 8;
  static constexpr uint32_t max_store_width_in_bytes = 64;

  static constexpr uint32_t max_load_size_in_bytes = 2048;
  static constexpr uint32_t max_store_size_in_bytes = 512;

  static constexpr uint32_t special_prefetch_width_in_bytes = 64;

  static constexpr uint32_t cache_line_size_in_bytes = 64;
  static constexpr uint32_t alignment_in_bytes = 8;
};

template <msg_type message_type, gpu_arch arg_tag>
struct client_load_store_attr_base_t {
  /// HW limitation checks https://gfxspecs.intel.com/Predator/Home/Index/55490
  static constexpr uint32_t max_load_height_in_elem = 32;
  static constexpr uint32_t max_load_width_in_bytes = 64;
  static constexpr uint32_t max_trans_load_width_in_bytes = 32;
  static constexpr uint32_t max_vnni_load_width_in_elems = 16;
  static constexpr uint32_t min_vnni_load_height_in_bytes = 4;

  static constexpr uint32_t max_store_height_in_elem = 8;
  static constexpr uint32_t max_store_width_in_bytes = 64;

  static constexpr uint32_t max_load_size_in_bytes = 2048;
  static constexpr uint32_t max_store_size_in_bytes = 512;

  static constexpr uint32_t special_prefetch_width_in_bytes = 64;

  static constexpr uint32_t cache_line_size_in_bytes = 64;
  static constexpr uint32_t alignment_in_bytes = 8;
};

template <>
struct load_store_attr_t<msg_type::block_2d, gpu_arch::XeHpg>
    : public client_load_store_attr_base_t<
          msg_type::block_2d,
          gpu_arch::XeHpg> {};
template <>
struct load_store_attr_t<msg_type::block_2d, gpu_arch::XeLpg>
    : public client_load_store_attr_base_t<
          msg_type::block_2d,
          gpu_arch::XeLpg> {};

template <gpu_arch arch_tag>
struct mma_attr_t {};

template <gpu_arch arch_tag>
struct client_mma_atr_base_t {
  static constexpr uint32_t mma_m_in_elem = 8;
  static constexpr uint32_t mma_n_in_elem = 8;
  static constexpr uint32_t mma_k_in_bytes = 32;
};

template <>
struct mma_attr_t<gpu_arch::XeHpc> {
  static constexpr uint32_t mma_m_in_elem = 8;
  static constexpr uint32_t mma_n_in_elem = 16;
  static constexpr uint32_t mma_k_in_bytes = 32;
};

template <>
struct mma_attr_t<gpu_arch::XeHpg>
    : public client_mma_atr_base_t<gpu_arch::XeHpg> {};

template <grf_mode grf_num_mode, gpu_arch arch_tag>
struct register_attr_t {};

template <grf_mode grf_num_mode, gpu_arch arch_tag>
struct client_register_attr_base_t {
  static constexpr uint32_t acc_reg_in_bytes =
      (grf_num_mode == grf_mode::normal) ? 4 * 32 : 8 * 32;
  static constexpr uint32_t grf_in_bytes =
      (grf_num_mode == grf_mode::normal) ? 128 * 32 : 256 * 32;
  static constexpr uint32_t reg_in_bytes = 32;
};

template <grf_mode grf_num_mode>
struct register_attr_t<grf_num_mode, gpu_arch::XeHpc> {
  static constexpr uint32_t acc_reg_in_bytes =
      (grf_num_mode == grf_mode::normal) ? 4 * 64 : 8 * 64;
  static constexpr uint32_t grf_in_bytes =
      (grf_num_mode == grf_mode::normal) ? 128 * 64 : 256 * 64;
  static constexpr uint32_t reg_in_bytes = 64;
};

template <grf_mode grf_num_mode>
struct register_attr_t<grf_num_mode, gpu_arch::XeHpg>
    : public client_register_attr_base_t<grf_num_mode, gpu_arch::XeHpg> {};

template <grf_mode grf_num_mode>
struct register_attr_t<grf_num_mode, gpu_arch::XeLpg>
    : public client_register_attr_base_t<grf_num_mode, gpu_arch::XeLpg> {};

template <gpu_arch arch_tag>
struct arch_attr_t {};

template <gpu_arch arch_tag>
struct client_arch_attr_base_t {
  template <msg_type message_type = msg_type::block_2d>
  using load_store_attr = load_store_attr_t<message_type, gpu_arch::XeHpg>;

  template <grf_mode grf_num_mode = grf_mode::double_grf>
  using register_attr = register_attr_t<grf_num_mode, gpu_arch::XeHpg>;

  using mma_attr = mma_attr_t<gpu_arch::XeHpg>;

  static constexpr uint32_t max_wg_num = 64;
  static constexpr uint32_t local_mem_size = 64 * 1024;
};

template <>
struct arch_attr_t<gpu_arch::XeHpc> {
  template <msg_type message_type = msg_type::block_2d>
  using load_store_attr = load_store_attr_t<message_type, gpu_arch::XeHpc>;

  template <grf_mode grf_num_mode = grf_mode::double_grf>
  using register_attr = register_attr_t<grf_num_mode, gpu_arch::XeHpc>;

  using mma_attr = mma_attr_t<gpu_arch::XeHpc>;

  static constexpr uint32_t max_wg_num = 64;
  static constexpr uint32_t local_mem_size = 128 * 1024;
};

template <>
struct arch_attr_t<gpu_arch::XeHpg>
    : public client_arch_attr_base_t<gpu_arch::XeHpg> {};

template <>
struct arch_attr_t<gpu_arch::XeLpg>
    : public client_arch_attr_base_t<gpu_arch::XeLpg> {};

/// @} xetla_core_arch_config

} // namespace gpu::xetla
