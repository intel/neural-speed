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

#include <subgroup/tile/api.hpp>
#include <subgroup/tile/impl/op_function.hpp>
#include <subgroup/tile/impl/payload_xe.hpp>

namespace gpu::xetla::subgroup {

namespace detail {
template <typename tile_t, typename payload_t, bool is_lsc_gather_ = true>
struct check_load_type {
  static constexpr bool is_lsc_gather = is_lsc_gather_;
  static constexpr bool is_global_block_2d =
      (payload_t::memory_space == mem_space::global &&
       (payload_t::message_type == msg_type::block_2d));

  static constexpr bool is_global_block_1d =
      (payload_t::memory_space == mem_space::global &&
       payload_t::message_type == msg_type::block_1d);

  static constexpr bool is_global_unaligned_2d_xe =
      ((payload_t::memory_space == mem_space::global) &&
       (payload_t::message_type == msg_type::unaligned_2d));

  static constexpr bool is_local_scatter_xe =
      ((payload_t::memory_space == mem_space::local) &&
       (payload_t::message_type == msg_type::scatter));

  static constexpr bool is_local_block_1d_xe =
      ((payload_t::memory_space == mem_space::local) &&
       (payload_t::message_type == msg_type::block_1d));
};

} // namespace detail

/// @brief This function loads data from 2D memory surface.
/// Loads an array of rectangular regions (X,Y)..(X+W,Y+H) from memory into
/// registers. Each block will be loaded serially by its corresponding payload.
/// @tparam tile_t Is the tile_t struct contains registers.
/// These registers will be the destination of load operation.
/// @tparam payload_t Is the mem_payload_t struct describing the memory
/// information Payload indicates the source of load operation.
/// @tparam L1 Is the cache hint for L1 cache.
/// @tparam L2 Is the cache hint for L2 cache.
/// @param tile Is the tile object with type tile_t, holds the return data of
/// the loads.
/// @param payload Is the payload object with type payload_t. Contains all the
/// information for loads.
/// @return No return, update in place.
template <
    cache_hint L1 = cache_hint::cached,
    cache_hint L2 = cache_hint::cached,
    typename tile_t,
    typename payload_t>
__XETLA_API typename std::enable_if_t<
    detail::check_load_type<tile_t, payload_t>::is_global_block_2d &&
    arch_has_2d_load_store<payload_t::arch_tag>>
tile_load(tile_t& tile, payload_t& payload) {
  using dtype = typename tile_t::dtype;
  using load_dtype = typename payload_t::mem_dtype;
  using tile_desc = typename tile_t::tile_desc;

  static constexpr uint32_t tile_size_x = tile_desc::tile_size_x;
  static constexpr uint32_t tile_size_y = tile_desc::tile_size_y;
  static constexpr uint32_t block_size_x = tile_desc::block_size_x;
  static constexpr uint32_t block_size_y = tile_desc::block_size_y;
  static constexpr uint32_t remained_size_y = tile_desc::remained_size_y;

  static constexpr uint32_t block_elems = tile_desc::block_elems;

  static constexpr uint32_t num_block_x = tile_desc::num_block_x;
  static constexpr uint32_t num_block_y = tile_desc::num_block_y;

  static constexpr gpu_arch arch_tag = payload_t::arch_tag;

  static constexpr reg_layout reg_layout_ = tile_desc::register_layout;
  static constexpr bool is_vnni_reverse =
      payload_t::mem_transpose_dtype_less4bytes &&
      ((reg_layout_ == reg_layout::tiled) ||
       (reg_layout_ == reg_layout::transpose_tiled));
  static constexpr bool reg_transpose = tile_desc::reg_transpose;

  static constexpr bool mem_transpose = payload_t::mem_transpose;
  static constexpr bool trans = payload_t::trans;
  static constexpr uint32_t scale_factor = payload_t::scale_factor;

  static constexpr bool mem_transform = payload_t::mem_transform;

  using load_store_attr = load_store_attr_t<msg_type::block_2d, arch_tag>;

  //   static constexpr uint32_t max_load_width_in_elem = trans
  //       ? load_store_attr::max_trans_load_width_in_bytes / sizeof(dtype)
  //       : load_store_attr::max_load_width_in_bytes / sizeof(dtype);
  //   static constexpr uint32_t max_load_height_in_elem = trans
  //       ? load_store_attr::max_trans_load_height_in_elem
  //       : load_store_attr::max_load_height_in_elem;
  static constexpr uint32_t max_trans_load_width_in_elem =
      load_store_attr::max_trans_load_width_in_bytes / sizeof(dtype);
  static constexpr uint32_t max_load_width_in_elem =
      load_store_attr::max_load_width_in_bytes / sizeof(dtype);

//   static constexpr uint32_t max_trans_load_height_in_elem =
//       load_store_attr::max_trans_load_height_in_elem;
  static constexpr uint32_t max_load_height_in_elem =
      load_store_attr::max_load_height_in_elem;

  static constexpr uint32_t elems_per_CL =
      load_store_attr::cache_line_size_in_bytes / sizeof(dtype);

  static constexpr uint32_t elems_per_reg =
      register_bytes_t<arch_tag>::reg_in_bytes / sizeof(dtype);

  static constexpr uint32_t ld_blk_size_y_limit =
      mem_transpose ? max_trans_load_width_in_elem : max_load_height_in_elem;
  static constexpr uint32_t ld_blk_size_y = reg_transpose
      ? block_size_y
      : std::min(ld_blk_size_y_limit, block_size_y);

  // array len is used to make sure memory load is cache line aligned
  // disabled while register or memory transpose
  static constexpr uint8_t arr_len_candidate =
      (reg_transpose ||
       mem_transpose
       // block elements should be integer
       // times of register bytes
       || ((block_size_y * block_size_x) % elems_per_reg != 0)
       // tail blocks also need to meet above condition
       ||
       (((tile_size_y % block_size_y) * block_size_x) % elems_per_reg != 0)) ||
          (block_size_y > ld_blk_size_y_limit)
      ? 1
      : (((tile_size_x % elems_per_CL) == 0)
             ? (((elems_per_CL % block_size_x) == 0)
                    ? elems_per_CL / block_size_x
                    : 1)
             : ((tile_size_x < elems_per_CL) ? (tile_size_x / block_size_x)
                                             : 1));
  static constexpr bool is_valid_arr_len_candidate = (arr_len_candidate == 1) ||
      (arr_len_candidate == 2) || (arr_len_candidate == 4);

  static constexpr uint8_t arr_len =
      is_valid_arr_len_candidate ? arr_len_candidate : 1;

  static_assert(
      reg_transpose || mem_transpose ||
          (!mem_transpose &&
           (block_size_x * arr_len) <= max_load_width_in_elem),
      "When reg_transpose was disabled, check 2d block width "
      "restriction");
  static_assert(
      !reg_transpose ||
          (!mem_transpose &&
           (block_size_x * arr_len) <= max_trans_load_width_in_elem) ||
          (mem_transpose && (block_size_y * arr_len) <= max_load_width_in_elem),
      "When reg_transpose was enabled, check 2d block width "
      "restriction");
  static_assert(
      !reg_transpose ||
          (!mem_transpose && (block_size_y <= max_load_height_in_elem)) ||
          (mem_transpose && (block_size_x) <= max_load_height_in_elem),
      "When reg_transpose was enabled, check 2d block height "
      "restriction");
  static_assert(
      tile_size_x % (block_size_x * arr_len) == 0,
      "tile_size_x should be a multiple of (block_size_x * arr_len)");
  static_assert(
      (reg_transpose &&
       ((block_size_x * sizeof(dtype)) % sizeof(load_dtype) == 0)) ||
          ((block_size_y * sizeof(dtype)) % sizeof(load_dtype) == 0),
      "check vnni limitation for DW transpose");

//   auto payload_2d = payload.payloads.xetla_format<uint32_t, num_block, 16>();
#pragma unroll
  for (uint32_t i = 0; i < num_block_y; ++i) {
    constexpr uint32_t load_block_elems = block_elems * arr_len;
    int offset_y = i * block_size_y;
#pragma unroll
    for (uint32_t j = 0; j < num_block_x; j += arr_len) {
      int32_t offset_x = j * block_size_x;
      auto reg_blk = tile.reg.xetla_select<load_block_elems, 1>(
          (i * num_block_x + j) * block_elems);
      constexpr uint32_t ld_blk_height = (reg_transpose && trans)
          ? detail::getNextPowerOf2<ld_blk_size_y>()
          : ld_blk_size_y;
      constexpr uint32_t tmp_size = ld_blk_height * block_size_x * arr_len;
      xetla_vector<dtype, tmp_size> reg_tmp;
#pragma unroll
      for (uint32_t ii = 0; ii < block_size_y / ld_blk_size_y; ++ii) {
        constexpr uint32_t load_elems = ld_blk_size_y * block_size_x * arr_len;
        reg_tmp.xetla_format<native_type_t<load_dtype>>() = xetla_load_global<
            native_type_t<load_dtype>,
            (trans ? ld_blk_size_y : block_size_x) / scale_factor,
            (trans ? block_size_x : ld_blk_size_y),
            // block_size_x / scale_factor,
            // ld_blk_size_y,
            arr_len,
            trans,
            mem_transform,
            L1,
            L2>(
            reinterpret_cast<const native_type_t<load_dtype*>>(
                payload.base_ptr),
            payload.surface_width,
            payload.surface_height,
            payload.surface_pitch,
            payload.offset_x +
                (mem_transpose ? (offset_y / (int)scale_factor +
                                  ii * ld_blk_size_y / (int)scale_factor)
                               : (offset_x / scale_factor)),

            payload.offset_y +
                (mem_transpose ? offset_x : (offset_y + ii * ld_blk_size_y)));

        if constexpr (reg_transpose && trans) {
          reg_blk.xetla_select<load_elems, 1>(ii * load_elems)
              .xetla_format<native_type_t<load_dtype>>() =
              reg_tmp
                  .xetla_format<
                      native_type_t<load_dtype>,
                      block_size_x / scale_factor,
                      ld_blk_height>()
                  .xetla_select<
                      block_size_x / scale_factor,
                      1,
                      ld_blk_size_y,
                      1>(0, 0);
        } else {
          reg_blk.xetla_select<tmp_size, 1>(ii * tmp_size) = reg_tmp;
        }
      }
      // exceed HW limitation
      if constexpr (block_size_y % ld_blk_size_y != 0) {
        constexpr uint32_t remained_start_y =
            block_size_y / ld_blk_size_y * ld_blk_size_y;
        constexpr uint32_t remained_start =
            remained_start_y * block_size_x * arr_len;
        constexpr uint32_t remained_blk_size_y = block_size_y % ld_blk_size_y;
        constexpr uint32_t load_elems =
            remained_blk_size_y * block_size_x * arr_len / scale_factor;

        constexpr uint8_t block_width =
            (mem_transpose ? remained_blk_size_y : block_size_x) / scale_factor;
        constexpr uint8_t block_height =
            mem_transpose ? block_size_x : remained_blk_size_y;
        // constexpr uint32_t block_widthx_widthy_arrlen =
        //     (block_width - 1) | ((block_height - 1) << 8);
        // gpu::xetla::detail::xetla_set_block_widthx_widthy_arrlen(
        //     tdesc.xetla_format<uint32_t>(), block_widthx_widthy_arrlen);

        reg_blk.xetla_select<load_elems, 1>(remained_start)
            .xetla_format<native_type_t<load_dtype>>() = xetla_load_global<
            native_type_t<load_dtype>,
            block_width,
            block_height,
            arr_len,
            trans,
            mem_transform,
            L1,
            L2>(
            reinterpret_cast<const native_type_t<load_dtype*>>(
                payload.base_ptr),
            payload.surface_width,
            payload.surface_height,
            payload.surface_pitch,
            payload.offset_x + offset_x / scale_factor,
            payload.offset_y + offset_y + remained_start_y);

        // xetla_tload_global<
        // load_dtype,
        // (load_elems / scale_factor),
        // L1,
        // L2,
        // trans,
        // mem_transform,
        // arch_tag>(tdesc);
      }
    }
  }
  // process tail
  if constexpr (remained_size_y > 0) {
    constexpr uint32_t remained_block_elems = block_size_x * remained_size_y;
    constexpr uint32_t processed_elems =
        num_block_y * num_block_x * block_elems;
    constexpr uint32_t remained_ld_blk_size_y =
        (!reg_transpose && (remained_size_y > ld_blk_size_y_limit))
        ? ld_blk_size_y_limit
        : remained_size_y;
    // auto payload_row = payload_2d.xetla_select<num_block_x, 1, 16, 1>(
    //     num_block_y * num_block_x, 0);
    // detail::reset_tile_desc_core<
    //     num_block_x,
    //     block_size_x,
    //     remained_ld_blk_size_y,
    //     scale_factor,
    //     arr_len,
    //     mem_transpose>(payload_row);
#pragma unroll
    for (uint32_t j = 0; j < num_block_x; j += arr_len) {
      int32_t offset_x = j * block_size_x;
      //   xetla_tdescriptor tdesc = payload_row.row(j);
      auto reg_blk = tile.reg.xetla_select<remained_block_elems * arr_len, 1>(
          processed_elems + j * remained_block_elems);
      constexpr uint32_t ld_blk_height = (reg_transpose && trans)
          ? detail::getNextPowerOf2<remained_ld_blk_size_y>()
          : remained_ld_blk_size_y;
      constexpr uint32_t tmp_size = ld_blk_height * block_size_x * arr_len;
      xetla_vector<dtype, tmp_size> reg_tmp;
#pragma unroll
      for (uint32_t ii = 0; ii < remained_size_y / remained_ld_blk_size_y;
           ++ii) {
        constexpr uint32_t load_elems =
            remained_ld_blk_size_y * block_size_x * arr_len;

        reg_tmp.xetla_format<native_type_t<load_dtype>>() = xetla_load_global<
            native_type_t<load_dtype>,
            block_size_x / scale_factor,
            ld_blk_height,
            arr_len,
            trans,
            mem_transform,
            L1,
            L2>(
            reinterpret_cast<const native_type_t<load_dtype*>>(
                payload.base_ptr),
            payload.surface_width,
            payload.surface_height,
            payload.surface_pitch,
            payload.offset_x + offset_x / scale_factor,
            payload.offset_y + num_block_y * block_size_y +
                ii * remained_ld_blk_size_y);
        //  xetla_tload_global<
        // load_dtype,
        // (ld_blk_height * block_size_x * arr_len / scale_factor),
        // L1,
        // L2,
        // trans,
        // mem_transform,
        // arch_tag>(tdesc);

        if constexpr (reg_transpose && trans) {
          reg_blk.xetla_select<load_elems, 1>(ii * load_elems)
              .xetla_format<native_type_t<load_dtype>>() =
              reg_tmp
                  .xetla_format<
                      load_dtype,
                      block_size_x / scale_factor,
                      ld_blk_height>()
                  .xetla_select<
                      block_size_x / scale_factor,
                      1,
                      remained_ld_blk_size_y,
                      1>(0, 0);
        } else {
          reg_blk.xetla_select<tmp_size, 1>(ii * tmp_size) = reg_tmp;
        }
        // if constexpr (mem_transpose) {
        //   xetla_update_tdesc_offsetx(
        //       tdesc.xetla_format<uint32_t>(),
        //       remained_ld_blk_size_y / scale_factor);
        // } else {
        //   xetla_update_tdesc_offsety(
        //       tdesc.xetla_format<uint32_t>(), remained_ld_blk_size_y);
        // }
      }
      constexpr uint32_t final_ld_blk_size_y =
          remained_size_y % remained_ld_blk_size_y;
      if constexpr (final_ld_blk_size_y != 0) {
        constexpr uint32_t final_start = remained_size_y /
            remained_ld_blk_size_y * remained_ld_blk_size_y * block_size_x *
            arr_len;
        constexpr uint32_t final_load_elems =
            final_ld_blk_size_y * block_size_x * arr_len;
        constexpr uint8_t block_width =
            (mem_transpose ? final_ld_blk_size_y : block_size_x) / scale_factor;
        constexpr uint8_t block_height =
            mem_transpose ? block_size_x : final_ld_blk_size_y;
        // constexpr uint32_t block_widthx_widthy_arrlen =
        //     (block_width - 1) | ((block_height - 1) << 8);
        // gpu::xetla::detail::xetla_set_block_widthx_widthy_arrlen(
        //     tdesc.xetla_format<uint32_t>(), block_widthx_widthy_arrlen);
        reg_blk.xetla_select<final_load_elems, 1>(final_start)
            .xetla_format<native_type_t<load_dtype>>() = xetla_load_global<
            native_type_t<load_dtype>,
            block_width,
            block_height,
            arr_len,
            trans,
            mem_transform,
            L1,
            L2>(
            reinterpret_cast<const native_type_t<load_dtype*>>(
                payload.base_ptr),
            payload.surface_width,
            payload.surface_height,
            payload.surface_pitch,
            payload.offset_x + offset_x / scale_factor,
            payload.offset_y + num_block_y * block_size_y +
                remained_size_y / remained_ld_blk_size_y *
                    remained_ld_blk_size_y);
        // xetla_tload_global<
        // load_dtype,
        // final_load_elems / scale_factor,
        // L1,
        // L2,
        // trans,
        // mem_transform,
        // arch_tag>(tdesc);
      }
    }
  }

  if constexpr (is_vnni_reverse) {
    vnni_reverse(tile);
  }
}

/// @brief This function loads data from memory.
/// For each enabled SIMT lane, a vector is read from memory into registers.
/// @tparam tile_t Is the tile_t struct contains registers.
/// These registers will be the destination of load operation.
/// @tparam payload_t Is the mem_payload_t struct describing the memory
/// information. Payload indicates the source of load operation.
/// @tparam L1 Is the cache hint for L1 cache.
/// @tparam L2 Is the cache hint for L2 cache.
/// @param tile Is the tile object with type tile_t, holds the return data of
/// the loads.
/// @param payload Is the payload object with type payload_t. Contains all the
/// information for loads.
/// @return No return, update in place.
template <
    cache_hint L1 = cache_hint::cached,
    cache_hint L2 = cache_hint::cached,
    typename tile_t,
    typename payload_t>
__XETLA_API typename std::enable_if_t<
    detail::check_load_type<tile_t, payload_t>::is_global_block_1d>
tile_load(tile_t& tile, payload_t& payload) {
  using dtype = typename payload_t::dtype;
  static constexpr uint32_t load_len = tile_t::tile_elems;
  static constexpr gpu_arch arch_tag = payload_t::arch_tag;

  using load_store_attr = load_store_attr_t<msg_type::block_1d, arch_tag>;
  static constexpr uint32_t max_load_vec_len =
      load_store_attr::max_aligned_load_vec_len;

  static constexpr uint32_t max_load_vec_elems =
      max_load_vec_len / sizeof(dtype);

  static constexpr uint32_t load_iter_steps = load_len / max_load_vec_elems;
  if constexpr (load_len >= max_load_vec_elems) {
#pragma unroll
    for (uint32_t i = 0; i < load_iter_steps; i++) {
      uint32_t offset_x = i * max_load_vec_elems;
      auto reg_sub = tile.reg.xetla_select<max_load_vec_elems, 1>(offset_x);
      uint32_t address_offset = offset_x * sizeof(dtype);
      reg_sub.xetla_format<dtype>() =
          xetla_load_global<dtype, max_load_vec_elems, L1, L2>(
              payload.base_ptr, payload.base_offset + address_offset);
    }
  }

  constexpr uint32_t tail_len = load_len % max_load_vec_elems * sizeof(dtype);
  uint32_t tail_offset = load_iter_steps * max_load_vec_len;
  detail::process_1d_tail<
      tail_len,
      (max_load_vec_len >> 1),
      detail::process_flag::load,
      L1,
      L2>(tile, payload, tail_offset);
}

/// @brief This function loads data from unaligned-2D memory surface.
/// Loads an array of rectangular regions (X,Y)..(X+W,Y+H) from memory into
/// registers. Each block will be loaded serially by its corresponding payload.
/// @tparam tile_t Is the tile_t struct contains registers.
/// These registers will be the destination of load operation.
/// @tparam payload_t Is the mem_payload_t struct describing the memory
/// information. Payload indicates the source of load operation.
/// @tparam L1 Is the cache hint for L1 cache.
/// @tparam L2 Is the cache hint for L2 cache.
/// @param tile Is the tile object with type tile_t, holds the return data of
/// the loads.
/// @param payload Is the payload object with type payload_t. Contains all the
/// information for loads.
/// @return No return, update in place.
template <
    cache_hint L1 = cache_hint::cached,
    cache_hint L2 = cache_hint::cached,
    typename tile_t,
    typename payload_t>
__XETLA_API typename std::enable_if_t<
    detail::check_load_type<tile_t, payload_t>::is_global_block_2d &&
    detail::check_load_type<tile_t, payload_t>::is_lsc_gather &&
    !arch_has_2d_load_store<payload_t::arch_tag>>
tile_load(tile_t& tile, payload_t& payload) {
  using dtype = typename payload_t::dtype;
  using tile_desc = typename payload_t::tile_desc;
  using load_dtype = typename payload_t::mem_dtype;
  constexpr uint32_t num_channel = payload_t::num_channel;
  constexpr uint32_t load_elems = num_channel * payload_t::vector_size;
  constexpr uint32_t pack_factor = payload_t::pack_factor;
  const xetla_vector<load_dtype, load_elems> reg_zeros(0);
  constexpr uint32_t block_height = payload_t::mem_transpose
      ? tile_desc::block_size_x
      : tile_desc::block_size_y;

  auto channel_offset = payload.channel_offset + payload.base_offset;
#pragma unroll
  for (uint32_t i = 0; i < tile_desc::num_block_y; i++) {
    uint32_t offset_y = i * tile_desc::block_size_y;
#pragma unroll
    for (uint32_t j = 0; j < tile_desc::num_block_x; j++) {
      uint32_t offset_x = j * tile_desc::block_size_x;
      auto reg_sub = tile.reg.xetla_select<tile_desc::block_elems, 1>(
          (i * tile_desc::num_block_x + j) * tile_desc::block_elems);
#pragma unroll
      for (uint32_t sub_block_offset = 0; sub_block_offset < block_height;
           sub_block_offset += num_channel) {
        xetla_vector<load_dtype, load_elems> reg_tmp = 0;
        uint32_t address_offset = payload_t::mem_transpose
            ? (offset_x + sub_block_offset) * payload.pitch_in_bytes +
                offset_y * sizeof(dtype)
            : offset_x * sizeof(dtype) +
                (offset_y + sub_block_offset) * payload.pitch_in_bytes;
        xetla_mask<num_channel> mask = 1;
        if constexpr (payload_t::use_mask) {
          // For SDP load, need mask
          const uint32_t sub_block_offset_x = payload.base_x + offset_x +
              (payload_t::mem_transpose ? sub_block_offset : 0);
          const uint32_t sub_block_offset_y = payload.base_y + offset_y +
              (payload_t::mem_transpose ? 0 : sub_block_offset);
          const auto offset_ch_dim = payload_t::mem_transpose
              ? sub_block_offset_x
              : sub_block_offset_y;

          mask = offset_ch_dim + num_channel > payload.height_in_elems
              ? (xetla_vector_gen<uint32_t, num_channel>(offset_ch_dim, 1) <
                 payload.height_in_elems)
              : 1;
          reg_tmp = xetla_load_global<
              load_dtype,
              load_elems,
              payload_t::vector_size,
              L1,
              L2>(
              payload.base_ptr,
              channel_offset + address_offset,
              mask,
              reg_zeros);
        } else {
          reg_tmp = xetla_load_global<
              load_dtype,
              load_elems,
              payload_t::vector_size,
              L1,
              L2>(payload.base_ptr, channel_offset + address_offset, mask);
        }

        if constexpr (
            payload_t::vector_size > 1 && payload_t::num_channel > 1) {
          xetla_vector<load_dtype, load_elems> reg_tmp_trans;
#pragma unroll
          for (uint32_t iii = 0; iii < payload_t::num_channel; iii++) {
            reg_tmp_trans.xetla_select<payload_t::vector_size, 1>(
                iii * payload_t::vector_size) =
                reg_tmp.xetla_select<
                    payload_t::vector_size,
                    payload_t::num_channel>(iii);
          }
          reg_sub
              .xetla_select<load_elems * pack_factor, 1>(
                  sub_block_offset *
                  (payload_t::mem_transpose ? tile_desc::block_size_y
                                            : tile_desc::block_size_x))
              .xetla_format<load_dtype>() = reg_tmp_trans;
        } else {
          reg_sub
              .xetla_select<load_elems * pack_factor, 1>(
                  sub_block_offset *
                  (payload_t::mem_transpose ? tile_desc::block_size_y
                                            : tile_desc::block_size_x))
              .xetla_format<load_dtype>() = reg_tmp;
        }
      }
    }
  }

  if constexpr (payload_t::trans) {
    tile_transpose(tile);
  }
  if constexpr (payload_t::mem_transform) {
    vnni_convert(tile);
  }
}

/// @brief This function loads data from unaligned-2D memory surface.
/// Loads an array of rectangular regions (X,Y)..(X+W,Y+H) from memory into
/// registers. Each block will be loaded serially by its corresponding payload.
/// @tparam tile_t Is the tile_t struct contains registers.
/// These registers will be the destination of load operation.
/// @tparam payload_t Is the mem_payload_t struct describing the memory
/// information. Payload indicates the source of load operation.
/// @tparam L1 Is the cache hint for L1 cache.
/// @tparam L2 Is the cache hint for L2 cache.
/// @param tile Is the tile object with type tile_t, holds the return data of
/// the loads.
/// @param payload Is the payload object with type payload_t. Contains all the
/// information for loads.
/// @return No return, update in place.
template <
    cache_hint L1 = cache_hint::cached,
    cache_hint L2 = cache_hint::cached,
    typename tile_t,
    typename payload_t>
__XETLA_API typename std::enable_if_t<
    detail::check_load_type<tile_t, payload_t>::is_global_block_2d &&
    !detail::check_load_type<tile_t, payload_t>::is_lsc_gather &&
    !arch_has_2d_load_store<payload_t::arch_tag>>
tile_load(tile_t& tile, payload_t& payload) {
  using dtype = typename payload_t::dtype;
  using tile_desc = typename payload_t::tile_desc;
  constexpr uint32_t load_elems = payload_t::mem_transpose
      ? tile_desc::block_size_y
      : tile_desc::block_size_x;
  constexpr uint32_t block_height = payload_t::mem_transpose
      ? tile_desc::block_size_x
      : tile_desc::block_size_y;

#pragma unroll
  for (uint32_t i = 0; i < tile_desc::num_block_y; i++) {
    uint32_t offset_y = i * tile_desc::block_size_y;
#pragma unroll
    for (uint32_t j = 0; j < tile_desc::num_block_x; j++) {
      uint32_t offset_x = j * tile_desc::block_size_x;
      auto reg_sub = tile.reg.xetla_select<tile_desc::block_elems, 1>(
          (i * tile_desc::num_block_x + j) * tile_desc::block_elems);
#pragma unroll
      for (uint32_t sub_block_y = 0; sub_block_y < block_height;
           sub_block_y++) {
        uint32_t address_offset = payload_t::mem_transpose
            ? (offset_x + sub_block_y) * payload.pitch_in_bytes +
                offset_y * sizeof(dtype)
            : offset_x * sizeof(dtype) +
                (offset_y + sub_block_y) * payload.pitch_in_bytes;

        reg_sub.xetla_select<load_elems, 1>(sub_block_y * load_elems) =
            xetla_load_global<dtype, load_elems, L1, L2>(
                (dtype*)payload.base_ptr, payload.base_offset + address_offset);
      }
    }
  }

  if constexpr (payload_t::trans) {
    tile_transpose(tile);
  }
  if constexpr (payload_t::mem_transform) {
    vnni_convert(tile);
  }
}

/// @brief This function loads data from unaligned-2D memory surface.
/// Loads an array of rectangular regions (X,Y)..(X+W,Y+H) from memory into
/// registers. Each block will be loaded serially by its corresponding payload.
/// @tparam tile_t Is the tile_t struct contains registers.
/// These registers will be the destination of load operation.
/// @tparam payload_t Is the mem_payload_t struct describing the memory
/// information. Payload indicates the source of load operation.
/// @tparam L1 Is the cache hint for L1 cache.
/// @tparam L2 Is the cache hint for L2 cache.
/// @param tile Is the tile object with type tile_t, holds the return data of
/// the loads.
/// @param payload Is the payload object with type payload_t. Contains all the
/// information for loads.
/// @return No return, update in place.
template <
    cache_hint L1 = cache_hint::cached,
    cache_hint L2 = cache_hint::cached,
    typename tile_t,
    typename payload_t,
    typename oob_check_tag = global_atomic_oob_check_on_tag>
__XETLA_API typename std::enable_if_t<
    detail::check_load_type<tile_t, payload_t>::is_global_unaligned_2d_xe>
tile_load(
    tile_t& tile,
    payload_t& payload,
    [[maybe_unused]] oob_check_tag tag = {}) {
  constexpr bool oob_check =
      std::is_same<oob_check_tag, global_atomic_oob_check_on_tag>::value;
  using dtype = typename payload_t::dtype;
  using load_dtype = typename payload_t::mem_dtype;
  constexpr uint32_t num_channel_y = payload_t::num_channel_y;
  constexpr uint32_t load_elems = num_channel_y * payload_t::num_channel_x;
  constexpr uint32_t scale_factor = payload_t::scale_factor;

  using tile_desc = typename tile_t::tile_desc;
  static constexpr uint32_t block_elems = tile_desc::block_elems;
  static constexpr uint32_t block_size_x = tile_desc::block_size_x;
  static constexpr uint32_t num_block_x = tile_desc::num_block_x;
  static constexpr uint32_t block_size_y = tile_desc::block_size_y;
  static constexpr uint32_t num_block_y = tile_desc::num_block_y;

  auto channel_offset = payload.channel_offset + payload.base_offset;
#pragma unroll
  for (uint32_t i = 0; i < num_block_y; i++) {
    uint32_t offset_y = i * block_size_y;
#pragma unroll
    for (uint32_t j = 0; j < num_block_x; j++) {
      uint32_t offset_x = j * block_size_x;
      auto reg_sub = tile.reg.xetla_select<block_elems, 1>(
          (i * num_block_x + j) * block_elems);
      xetla_mask<load_elems> pred_x = oob_check
          ? payload.step_x + payload.base_x + offset_x < payload.width_in_elems
          : 1;
#pragma unroll
      for (uint32_t sub_block_y = 0; sub_block_y < block_size_y;
           sub_block_y += num_channel_y) {
        xetla_vector<load_dtype, load_elems> reg_tmp;
        xetla_mask<load_elems> pred_y = oob_check
            ? payload.step_y + payload.base_y + offset_y + sub_block_y <
                payload.height_in_elems
            : 1;

        uint32_t address_offset = payload_t::mem_transpose
            ? offset_x * payload.pitch_in_bytes +
                (offset_y + sub_block_y) * sizeof(dtype)
            : offset_x * sizeof(dtype) +
                (offset_y + sub_block_y) * payload.pitch_in_bytes;

        reg_tmp = xetla_load_global<load_dtype, load_elems, 1, L1, L2>(
            payload.base_ptr,
            channel_offset + address_offset,
            pred_x && pred_y);
        reg_tmp.xetla_merge(reg_tmp, 0, pred_x && pred_y);

        reg_sub
            .xetla_select<load_elems * scale_factor, 1>(
                sub_block_y * block_size_x)
            .xetla_format<load_dtype>() = reg_tmp;
      }
    }
  }
  // process the tail
  if constexpr (tile_desc::remained_size_y != 0) {
    constexpr uint32_t remained_size_y = tile_desc::remained_size_y;
    constexpr uint32_t offset_y = tile_desc::tile_size_y - remained_size_y;
    constexpr uint32_t processed_elems = offset_y * tile_desc::tile_size_x;
    constexpr uint32_t remain_block_elems =
        remained_size_y * tile_desc::block_size_x;
#pragma unroll
    for (uint32_t j = 0; j < num_block_x; j++) {
      uint32_t offset_x = j * block_size_x;
      auto reg_sub = tile.reg.xetla_select<remain_block_elems, 1>(
          processed_elems + j * remain_block_elems);
      xetla_mask<load_elems> pred_x = oob_check
          ? payload.step_x + payload.base_x + offset_x < payload.width_in_elems
          : 1;
#pragma unroll
      for (uint32_t sub_block_y = 0; sub_block_y < remained_size_y;
           sub_block_y += num_channel_y) {
        xetla_vector<load_dtype, load_elems> reg_tmp;
        xetla_mask<load_elems> pred_y = oob_check
            ? payload.step_y + payload.base_y + offset_y + sub_block_y <
                payload.height_in_elems
            : 1;

        uint32_t address_offset = payload_t::mem_transpose
            ? offset_x * payload.pitch_in_bytes +
                (offset_y + sub_block_y) * sizeof(dtype)
            : offset_x * sizeof(dtype) +
                (offset_y + sub_block_y) * payload.pitch_in_bytes;

        reg_tmp = xetla_load_global<
            load_dtype,
            1,
            data_size::default_size,
            L1,
            L2,
            load_elems>(
            payload.base_ptr,
            channel_offset + address_offset,
            pred_x && pred_y);

        reg_tmp.xetla_merge(reg_tmp, 0, pred_x && pred_y);

        reg_sub
            .xetla_select<load_elems * scale_factor, 1>(
                sub_block_y * tile_desc::block_size_x)
            .xetla_format<load_dtype>() = reg_tmp;
      }
    }
  }

  if constexpr (payload_t::reg_transpose) {
    tile_transpose(tile);
  }

  if constexpr (payload_t::mem_transform) {
    vnni_convert(tile);
  }
}

/// @brief Is the data load func from local shared memory to register file,
/// which supports the memory surface is 1d or 2d scenario. And we always assume
/// data in SLM is row major.
/// @tparam tile_t Is the tile_t struct contains registers
/// These registers will be the destination of load operation.
/// @tparam payload_t Is the mem_payload_t struct describing the memory
/// information. Payload indicates the source of load operation.
/// @tparam L1 Is the cache hint for L1 cache.
/// @tparam L2 Is the cache hint for L2 cache.
/// @param tile Is the tile object with type tile_t, holds the return data of
/// the loads.
/// @param payload Is the payload object with type payload_t. Contains all the
/// information for loads.
/// @return No return, update in place.
template <
    cache_hint L1 = cache_hint::cached,
    cache_hint L2 = cache_hint::cached,
    typename tile_t,
    typename payload_t>
__XETLA_API typename std::enable_if_t<
    detail::check_load_type<tile_t, payload_t>::is_local_scatter_xe>
tile_load(tile_t& tile, payload_t& payload) {
  using dtype = typename payload_t::dtype;
  using tile_desc = typename payload_t::tile_desc;
  using load_dtype = typename payload_t::mem_dtype;

  constexpr uint32_t num_channel_y = payload_t::num_channel_y;
  constexpr uint32_t load_elems = num_channel_y * tile_desc::block_size_x;
  static constexpr bool mem_transform = payload_t::mem_transform;

#pragma unroll
  for (uint32_t i = 0; i < tile_desc::tile_size_y / tile_desc::block_size_y;
       i++) {
    uint32_t offset_y = i * tile_desc::block_size_y;
#pragma unroll
    for (uint32_t j = 0; j < tile_desc::num_block_x; j++) {
      uint32_t offset_x = j * tile_desc::block_size_x;
      auto reg_sub = tile.reg.xetla_select<tile_desc::block_elems, 1>(
          (i * tile_desc::num_block_x + j) * tile_desc::block_elems);
#pragma unroll
      for (uint32_t sub_block_y = 0; sub_block_y < tile_desc::block_size_y;
           sub_block_y += num_channel_y) {
        uint32_t address_offset = offset_x * sizeof(dtype) +
            (sub_block_y + offset_y) * payload.pitch_in_bytes;
        reg_sub
            .xetla_select<load_elems, 1>(sub_block_y * tile_desc::block_size_x)
            .xetla_format<load_dtype>() =
            xetla_load_local<load_dtype>(payload.address + address_offset);
      }
    }
  }
  // process the tail
  if constexpr ((tile_desc::tile_size_y % tile_desc::block_size_y) != 0) {
    constexpr uint32_t remained_size_y = tile_desc::remained_size_y;
    constexpr uint32_t offset_y = tile_desc::tile_size_y - remained_size_y;
    constexpr uint32_t processed_elems = offset_y * tile_desc::tile_size_x;
    constexpr uint32_t remain_block_elems =
        remained_size_y * tile_desc::block_size_x;
#pragma unroll
    for (uint32_t j = 0; j < tile_desc::num_block_x; j++) {
      uint32_t offset_x = j * tile_desc::block_size_x;
      auto reg_sub = tile.reg.xetla_select<remain_block_elems, 1>(
          processed_elems + j * remain_block_elems);
#pragma unroll
      for (uint32_t sub_block_y = 0; sub_block_y < remained_size_y;
           sub_block_y += num_channel_y) {
        uint32_t address_offset = offset_x * sizeof(dtype) +
            (sub_block_y + offset_y) * payload.pitch_in_bytes;
        reg_sub
            .xetla_select<load_elems, 1>(sub_block_y * tile_desc::block_size_x)
            .xetla_format<load_dtype>() =
            xetla_load_local<load_dtype>(payload.address + address_offset);
      }
    }
  }
  if constexpr (payload_t::reg_transpose) {
    tile_transpose(tile);
  }
  if constexpr (mem_transform) {
    vnni_convert(tile);
  }
}

/// @brief Is the data load func from shared local memory to register file,
/// which supports the memory surface is 1d scenario. And the src memory layout
/// is always row major.
/// @tparam tile_t Is the tile_t struct contains registers.
/// These registers will be the destination of load operation.
/// @tparam payload_t Is the mem_payload_t struct describing the memory
/// information. Payload indicates the source of load operation.
/// @tparam L1 Is the cache hint for L1 cache.
/// @tparam L2 Is the cache hint for L2 cache.
/// @param tile Is the tile object with type tile_t, holds the return data of
/// the loads.
/// @param payload Is the payload object with type payload_t. Contains all the
/// information for loads.
/// @return No return, update in place.
template <
    cache_hint L1 = cache_hint::cached,
    cache_hint L2 = cache_hint::cached,
    typename tile_t,
    typename payload_t>
__XETLA_API typename std::enable_if_t<
    detail::check_load_type<tile_t, payload_t>::is_local_block_1d_xe>
tile_load(tile_t& tile, payload_t& payload) {
  using dtype = typename tile_t::dtype;
  static constexpr uint32_t load_len = tile_t::tile_elems;
  static constexpr gpu_arch arch_tag = payload_t::arch_tag;

  using load_store_attr = load_store_attr_t<msg_type::block_1d, arch_tag>;
  static constexpr uint32_t max_load_vec_len =
      load_store_attr::max_aligned_load_vec_len;
  static constexpr uint32_t max_load_vec_elems =
      max_load_vec_len / sizeof(dtype);

  static constexpr uint32_t load_iter_steps = load_len / max_load_vec_elems;
  if constexpr (load_len >= max_load_vec_elems) {
#pragma unroll
    for (uint32_t j = 0; j < load_iter_steps; j++) {
      uint32_t offset_x = j * max_load_vec_elems;
      auto reg_sub = tile.reg.xetla_select<max_load_vec_elems, 1>(offset_x);
      uint32_t address_offset = offset_x * sizeof(dtype);
      reg_sub.xetla_format<dtype>() =
          xetla_load_local<dtype, max_load_vec_elems, data_size::default_size>(
              payload.base_address + payload.address + address_offset);
    }
  }
  constexpr uint32_t tail_len = load_len % max_load_vec_elems * sizeof(dtype);
  uint32_t tail_offset = load_iter_steps * max_load_vec_len;
  detail::process_1d_tail<
      tail_len,
      (max_load_vec_len >> 1),
      detail::process_flag::load,
      L1,
      L2>(tile, payload, tail_offset);
}

} // namespace gpu::xetla::subgroup
