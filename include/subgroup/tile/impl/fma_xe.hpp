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

#include <common/core/explicit_conv.hpp>
#include <subgroup/tile/api.hpp>

namespace gpu::xetla::subgroup {

/// @brief Is the tile mma operation functor, specialized for Xe and fpu engine.
template <
    typename matAcc_dst_t_,
    typename matAcc_src_t_,
    typename matB_t_,
    typename matA_t_,
    gpu_arch arch_tag_>
struct tile_mma_t<
    matAcc_dst_t_,
    matAcc_src_t_,
    matB_t_,
    matA_t_,
    mma_engine::fpu,
    arch_tag_,
    std::enable_if_t<
        arch_has_fpu<arch_tag_> &&
        matA_t_::register_layout == reg_layout::transpose_tiled &&
        matB_t_::register_layout == reg_layout::tiled>> {
  using matA_t = matA_t_;
  using matB_t = matB_t_;
  using matSrc_t = matAcc_src_t_;
  using matDst_t = matAcc_dst_t_;
  using dtype_a = typename matA_t::dtype;
  using dtype_b = typename matB_t::dtype;
  using dtype_src = typename matSrc_t::dtype;
  using dtype_dst = typename matDst_t::dtype;

  static constexpr uint32_t a_tile_size_y = matA_t::tile_size_y;
  static constexpr uint32_t a_tile_size_x = matA_t::tile_size_x;
  static constexpr uint32_t a_tile_elems = matA_t::tile_elems;
  static constexpr uint32_t a_block_size_w = matA_t::block_size_y;
  static constexpr uint32_t a_block_size_h = matA_t::block_size_x;
  static constexpr uint32_t a_block_elems = matA_t::block_elems;

  static constexpr uint32_t b_tile_size_x = matB_t::tile_size_x;
  static constexpr uint32_t b_tile_size_y = matB_t::tile_size_y;
  static constexpr uint32_t b_tile_elems = matB_t::tile_elems;
  static constexpr uint32_t b_block_size_x = matB_t::block_size_x;
  static constexpr uint32_t b_block_size_y = matB_t::block_size_y;
  static constexpr uint32_t b_block_elems = matB_t::block_elems;

  static constexpr uint32_t tile_size_m = matDst_t::tile_size_y;
  static constexpr uint32_t tile_size_k = a_tile_size_x;
  static constexpr uint32_t tile_size_n = matDst_t::tile_size_x;
  static constexpr uint32_t tile_elems = tile_size_m * tile_size_n;
  static constexpr uint32_t block_size_n = matDst_t::block_size_x;
  static constexpr uint32_t block_size_k = a_block_size_h;
  static constexpr uint32_t block_size_m = matDst_t::block_size_y;
  static constexpr uint32_t block_elems = block_size_m * block_size_n;

  static_assert(
      tile_size_m == matA_t::tile_size_y,
      "matAcc tile m should match with matA tile m");
  static_assert(
      a_tile_size_x == b_tile_size_y,
      "matA tile k should match with matB tile k");
  static_assert(
      tile_size_n == matB_t::tile_size_x,
      "matAcc tile n should match with matB tile n");
  static_assert(
      block_size_m == a_block_size_w,
      "matAcc block m should match with matA block m");
  static_assert(
      block_size_n == b_block_size_x,
      "matAcc block n should match with matB block n");
  static_assert(
      (tile_size_k % block_size_k) == 0,
      "matAcc tile_size_k should be a multiple of block_size_k");

  static constexpr int32_t num_block_n = matDst_t::num_block_x;
  static constexpr int32_t num_block_m = matDst_t::num_block_y;
  static constexpr int32_t num_block_k = tile_size_k / block_size_k;

  using mma_attr = mma_attr_t<arch_tag_, mma_engine::fpu, tile_size_m>;
  static constexpr int32_t mma_m = mma_attr::mma_m_in_elem;

  template <int blk_m, int blk_n, int blk_k>
  __XETLA_API static void mma_core(
      xetla_vector_ref<dtype_dst, blk_m * blk_n> __REF__ dst,
      xetla_vector_ref<dtype_src, blk_m * blk_n> __REF__ src,
      xetla_vector_ref<dtype_src, blk_k * blk_n> __REF__ b_block,
      xetla_vector_ref<dtype_a, blk_k * blk_m> __REF__ a_block) {
    constexpr uint32_t blk_m_iters = blk_m / mma_m;
    constexpr uint32_t tail_m = blk_m % mma_m;
    auto dst_blk_2d = dst.xetla_format<dtype_dst, blk_m, blk_n>();
    auto src_blk_2d = src.xetla_format<dtype_src, blk_m, blk_n>();
    auto b_blk_2d = b_block.xetla_format<dtype_src, blk_k, blk_n>();
    xetla_vector<dtype_src, a_block_elems> new_a_block =
        xetla_cvt<dtype_src, dtype_a, a_block_elems>(
            a_block.xetla_select<a_block_elems, 1>(0));

    if constexpr (blk_m_iters > 0) {
#pragma unroll
      for (uint32_t i = 0; i < blk_m_iters; i++) {
        xetla_vector<dtype_dst, mma_m * blk_n> dst_tmp;
        auto dst_tmp_2d = dst_tmp.xetla_format<dtype_dst, mma_m, blk_n>();
#pragma unroll
        for (uint32_t i_acc = 0; i_acc < mma_m; i_acc++) {
          dst_tmp_2d.row(i_acc) =
              new_a_block[i_acc + i * mma_m] * b_blk_2d.row(0) +
              src_blk_2d.row(i_acc + i * mma_m);
        }
#pragma unroll
        for (uint32_t k = 1; k < blk_k - 1; k++) {
          for (uint32_t i_acc = 0; i_acc < mma_m; i_acc++) {
            int a_offset = k * blk_m + i_acc + i * mma_m;
            dst_tmp_2d.row(i_acc) += new_a_block[a_offset] * b_blk_2d.row(k);
          }
        }
        for (uint32_t i_acc = 0; i_acc < mma_m; i_acc++) {
          int a_offset = (blk_k - 1) * blk_m + i_acc + i * mma_m;
          dst_blk_2d.row(i_acc + i * mma_m) =
              new_a_block[a_offset] * b_blk_2d.row(blk_k - 1) +
              dst_tmp_2d.row(i_acc);
        }
        SW_BARRIER();
      }
    }

    if constexpr (tail_m != 0) {
      constexpr uint32_t tail_start_m = blk_m_iters * mma_m;
      xetla_vector<dtype_dst, tail_m * blk_n> dst_tmp;
      auto dst_tmp_2d = dst_tmp.xetla_format<dtype_dst, tail_m, blk_n>();
#pragma unroll
      for (uint32_t i_acc = 0; i_acc < tail_m; i_acc++) {
        dst_tmp_2d.row(i_acc) =
            new_a_block[i_acc + tail_start_m] * b_blk_2d.row(0) +
            src_blk_2d.row(i_acc + tail_start_m);
      }
#pragma unroll
      for (uint32_t k = 1; k < blk_k - 1; k++) {
        for (uint32_t i_acc = 0; i_acc < tail_m; i_acc++) {
          int a_offset = k * blk_m + i_acc + tail_start_m;
          dst_tmp_2d.row(i_acc) += new_a_block[a_offset] * b_blk_2d.row(k);
        }
      }
      for (uint32_t i_acc = 0; i_acc < tail_m; i_acc++) {
        int a_offset = (blk_k - 1) * blk_m + i_acc + tail_start_m;
        dst_blk_2d.row(i_acc + tail_start_m) =
            new_a_block[a_offset] * b_blk_2d.row(blk_k - 1) +
            dst_tmp_2d.row(i_acc);
      }
    }
  }

  __XETLA_API static void mma(
      matDst_t& dst,
      matSrc_t& src,
      matB_t& b,
      matA_t& a) {
    constexpr auto b_reg_sizes = b_block_size_y * b_tile_size_x;

    { // k_blk=0
      xetla_vector<dtype_src, b_reg_sizes> b_reg =
          xetla_cvt<dtype_src, dtype_b, b_reg_sizes>(
              b.reg.xetla_select<b_reg_sizes, 1>(0));

      if constexpr (tile_size_m >= block_size_m) {
#pragma unroll
        for (uint32_t i = 0; i < tile_size_m / block_size_m; i++) {
          auto a_block = a.reg.xetla_select<a_block_elems, 1>(
              i * num_block_k * a_block_elems);
#pragma unroll
          for (uint32_t j = 0; j < num_block_n; j++) {
            auto b_block =
                b_reg.xetla_select<b_block_elems, 1>(j * b_block_elems);
            auto src_block = src.reg.xetla_select<block_elems, 1>(
                (i * num_block_n + j) * block_elems);
            auto dst_block = dst.reg.xetla_select<block_elems, 1>(
                (i * num_block_n + j) * block_elems);
            mma_core<block_size_m, block_size_n, block_size_k>(
                dst_block, src_block, b_block, a_block);
          }
        }
      }

      // process the tail
      if constexpr ((tile_size_m % block_size_m) != 0) {
        constexpr uint32_t tail_start_m =
            tile_size_m / block_size_m * block_size_m;
        constexpr uint32_t a_tail_blk_w = a_tile_size_y - tail_start_m;
        constexpr uint32_t a_tail_blk_elems = a_block_size_h * a_tail_blk_w;
        constexpr uint32_t tail_size_m = tile_size_m - tail_start_m;
        constexpr uint32_t acc_tail_blk_elems = tail_size_m * block_size_n;
        auto a_block = a.reg.xetla_select<a_tail_blk_elems, 1>(
            a_tile_size_x * tail_start_m);
#pragma unroll
        for (uint32_t j = 0; j < num_block_n; j++) {
          auto b_block =
              b_reg.xetla_select<b_block_elems, 1>(j * b_block_elems);
          auto src_block = src.reg.xetla_select<acc_tail_blk_elems, 1>(
              (tail_start_m * tile_size_n) + j * acc_tail_blk_elems);
          auto dst_block = dst.reg.xetla_select<acc_tail_blk_elems, 1>(
              (tail_start_m * tile_size_n) + j * acc_tail_blk_elems);
          mma_core<tail_size_m, block_size_n, block_size_k>(
              dst_block, src_block, b_block, a_block);
        }
      }
    }
    // different K block
#pragma unroll
    for (uint32_t k_i = 1; k_i < num_block_k; k_i++) {
      xetla_vector<dtype_src, b_reg_sizes> b_reg =
          xetla_cvt<dtype_src, dtype_b, b_reg_sizes>(
              b.reg.xetla_select<b_reg_sizes, 1>(
                  k_i * b_block_size_y * b_tile_size_x));
#pragma unroll
      for (uint32_t i = 0; i < tile_size_m / block_size_m; i++) {
        auto a_block = a.reg.xetla_select<a_block_elems, 1>(
            (i * num_block_k + k_i) * a_block_elems);
#pragma unroll
        for (uint32_t j = 0; j < num_block_n; j++) {
          auto b_block =
              b_reg.xetla_select<b_block_elems, 1>(j * b_block_elems);
          auto dst_block = dst.reg.xetla_select<block_elems, 1>(
              (i * num_block_n + j) * block_elems);
          mma_core<block_size_m, block_size_n, block_size_k>(
              dst_block, dst_block, b_block, a_block);
        }
      }
      // process the tail
      if constexpr ((tile_size_m % block_size_m) != 0) {
        constexpr uint32_t tail_start_m =
            tile_size_m / block_size_m * block_size_m;
        constexpr uint32_t a_tail_blk_w = a_tile_size_y - tail_start_m;
        constexpr uint32_t a_tail_blk_elems = a_block_size_h * a_tail_blk_w;
        constexpr uint32_t tail_size_m = tile_size_m - tail_start_m;
        constexpr uint32_t acc_tail_blk_elems = tail_size_m * block_size_n;
        auto a_block = a.reg.xetla_select<a_tail_blk_elems, 1>(
            a_tile_size_x * tail_start_m + k_i * a_tail_blk_elems);
#pragma unroll
        for (uint32_t j = 0; j < num_block_n; j++) {
          auto b_block =
              b_reg.xetla_select<b_block_elems, 1>(j * b_block_elems);
          auto dst_block = dst.reg.xetla_select<acc_tail_blk_elems, 1>(
              (tail_start_m * tile_size_n) + j * acc_tail_blk_elems);
          mma_core<tail_size_m, block_size_n, block_size_k>(
              dst_block, dst_block, b_block, a_block);
        }
      }
    }
  }
};

template <
    typename matAcc_dst_t_,
    typename matAcc_src_t_,
    typename matB_t_,
    typename matA_t_,
    gpu_arch arch_tag_>
struct tile_mma_t<
    matAcc_dst_t_,
    matAcc_src_t_,
    matB_t_,
    matA_t_,
    mma_engine::fpu,
    arch_tag_,
    std::enable_if_t<
        arch_has_fpu<arch_tag_> &&
        matA_t_::register_layout == reg_layout::tiled &&
        matB_t_::register_layout == reg_layout::tiled>> {
  using matA_t = matA_t_;
  using matB_t = matB_t_;
  using matSrc_t = matAcc_src_t_;
  using matDst_t = matAcc_dst_t_;
  using dtype_a = typename matA_t::dtype;
  using dtype_b = typename matB_t::dtype;
  using dtype_src = typename matSrc_t::dtype;
  using dtype_dst = typename matDst_t::dtype;

  static constexpr uint32_t a_tile_size_y = matA_t::tile_size_y;
  static constexpr uint32_t a_tile_size_x = matA_t::tile_size_x;
  static constexpr uint32_t a_tile_elems = matA_t::tile_elems;
  static constexpr uint32_t a_block_size_w = matA_t::block_size_y;
  static constexpr uint32_t a_block_size_h = matA_t::block_size_x;
  static constexpr uint32_t a_block_elems = matA_t::block_elems;

  static constexpr uint32_t b_tile_size_x = matB_t::tile_size_x;
  static constexpr uint32_t b_tile_size_y = matB_t::tile_size_y;
  static constexpr uint32_t b_tile_elems = matB_t::tile_elems;
  static constexpr uint32_t b_block_size_x = matB_t::block_size_x;
  static constexpr uint32_t b_block_size_y = matB_t::block_size_y;
  static constexpr uint32_t b_block_elems = matB_t::block_elems;

  static constexpr uint32_t tile_size_m = matDst_t::tile_size_y;
  static constexpr uint32_t tile_size_k = a_tile_size_x;
  static constexpr uint32_t tile_size_n = matDst_t::tile_size_x;
  static constexpr uint32_t tile_elems = tile_size_m * tile_size_n;
  static constexpr uint32_t block_size_n = matDst_t::block_size_x;
  static constexpr uint32_t block_size_k = a_block_size_h;
  static constexpr uint32_t block_size_m = matDst_t::block_size_y;
  static constexpr uint32_t block_elems = block_size_m * block_size_n;

  static_assert(
      tile_size_m == matA_t::tile_size_y,
      "matAcc tile m should match with matA tile m");
  static_assert(
      a_tile_size_x == b_tile_size_y,
      "matA tile k should match with matB tile k");
  static_assert(
      tile_size_n == matB_t::tile_size_x,
      "matAcc tile n should match with matB tile n");
  static_assert(
      block_size_m == a_block_size_w,
      "matAcc block m should match with matA block m");
  static_assert(
      block_size_n == b_block_size_x,
      "matAcc block n should match with matB block n");
  static_assert(
      (tile_size_k % block_size_k) == 0,
      "matAcc tile_size_k should be a multiple of block_size_k");

  static constexpr int32_t num_block_n = matDst_t::num_block_x;
  static constexpr int32_t num_block_m = matDst_t::num_block_y;
  static constexpr int32_t num_block_k = tile_size_k / block_size_k;

  using mma_attr = mma_attr_t<arch_tag_, mma_engine::fpu, tile_size_m>;
  static constexpr int32_t mma_m = mma_attr::mma_m_in_elem;

  template <int blk_m, int blk_n, int blk_k>
  __XETLA_API static void mma_core(
      xetla_vector_ref<dtype_dst, blk_m * blk_n> __REF__ dst,
      xetla_vector_ref<dtype_src, blk_m * blk_n> __REF__ src,
      xetla_vector_ref<dtype_src, blk_k * blk_n> __REF__ b_block,
      xetla_vector_ref<dtype_a, blk_m * blk_k> __REF__ a_block) {
    constexpr uint32_t blk_m_iters = blk_m / mma_m;
    constexpr uint32_t tail_m = blk_m % mma_m;
    auto dst_blk_2d = dst.xetla_format<dtype_dst, blk_m, blk_n>();
    auto src_blk_2d = src.xetla_format<dtype_src, blk_m, blk_n>();
    auto b_blk_2d = b_block.xetla_format<dtype_src, blk_k, blk_n>();
    xetla_vector<dtype_src, a_block_elems> new_a_block =
        xetla_cvt<dtype_src, dtype_a, a_block_elems>(
            a_block.xetla_select<a_block_elems, 1>(0));

    if constexpr (blk_m_iters > 0) {
#pragma unroll
      for (uint32_t i = 0; i < blk_m_iters; i++) {
#pragma unroll
        for (uint32_t i_acc = 0; i_acc < mma_m; i_acc++) {
          dst_blk_2d.row(i_acc + i * mma_m) = src_blk_2d.row(i_acc + i * mma_m);
        }
        int32_t a_start_off = i * mma_m * blk_k;
#pragma unroll
        for (uint32_t k = 0; k < blk_k; k++) {
#pragma unroll
          for (uint32_t i_acc = 0; i_acc < mma_m; i_acc++) {
            int a_offset = a_start_off + i_acc * blk_k + k;
            dst_blk_2d.row(i_acc + i * mma_m) +=
                new_a_block[a_offset] * b_blk_2d.row(k);
          }
        }
        SW_BARRIER();
      }
    }

    if constexpr (tail_m != 0) {
      constexpr uint32_t tail_start_m = blk_m_iters * mma_m;
#pragma unroll
      for (uint32_t i_acc = 0; i_acc < tail_m; i_acc++) {
        dst_blk_2d.row(i_acc + tail_start_m) =
            src_blk_2d.row(i_acc + tail_start_m);
      }
      int32_t a_start_off = tail_start_m * blk_k;
#pragma unroll
      for (uint32_t k = 0; k < blk_k; k++) {
#pragma unroll
        for (uint32_t i_acc = 0; i_acc < tail_m; i_acc++) {
          int a_offset = a_start_off + i_acc * blk_k + k;
          dst_blk_2d.row(i_acc + tail_start_m) +=
              new_a_block[a_offset] * b_blk_2d.row(k);
        }
      }
    }
  }

  __XETLA_API static void mma(
      matDst_t& dst,
      matSrc_t& src,
      matB_t& b,
      matA_t& a) {
    constexpr auto b_reg_sizes = b_block_size_y * b_tile_size_x;

    { // k_blk=0
      xetla_vector<dtype_src, b_reg_sizes> b_reg =
          xetla_cvt<dtype_src, dtype_b, b_reg_sizes>(
              b.reg.xetla_select<b_reg_sizes, 1>(0));

      if constexpr (tile_size_m >= block_size_m) {
#pragma unroll
        for (uint32_t i = 0; i < tile_size_m / block_size_m; i++) {
          auto a_block = a.reg.xetla_select<a_block_elems, 1>(
              i * num_block_k * a_block_elems);
#pragma unroll
          for (uint32_t j = 0; j < num_block_n; j++) {
            auto b_block =
                b_reg.xetla_select<b_block_elems, 1>(j * b_block_elems);
            auto src_block = src.reg.xetla_select<block_elems, 1>(
                (i * num_block_n + j) * block_elems);
            auto dst_block = dst.reg.xetla_select<block_elems, 1>(
                (i * num_block_n + j) * block_elems);
            mma_core<block_size_m, block_size_n, block_size_k>(
                dst_block, src_block, b_block, a_block);
          }
        }
      }

      // process the tail
      if constexpr ((tile_size_m % block_size_m) != 0) {
        constexpr uint32_t tail_start_m =
            tile_size_m / block_size_m * block_size_m;
        constexpr uint32_t a_tail_blk_w = a_tile_size_y - tail_start_m;
        constexpr uint32_t a_tail_blk_elems = a_block_size_h * a_tail_blk_w;
        constexpr uint32_t tail_size_m = tile_size_m - tail_start_m;
        constexpr uint32_t acc_tail_blk_elems = tail_size_m * block_size_n;
        auto a_block = a.reg.xetla_select<a_tail_blk_elems, 1>(
            a_tile_size_x * tail_start_m);
#pragma unroll
        for (uint32_t j = 0; j < num_block_n; j++) {
          auto b_block =
              b_reg.xetla_select<b_block_elems, 1>(j * b_block_elems);
          auto src_block = src.reg.xetla_select<acc_tail_blk_elems, 1>(
              (tail_start_m * tile_size_n) + j * acc_tail_blk_elems);
          auto dst_block = dst.reg.xetla_select<acc_tail_blk_elems, 1>(
              (tail_start_m * tile_size_n) + j * acc_tail_blk_elems);
          mma_core<tail_size_m, block_size_n, block_size_k>(
              dst_block, src_block, b_block, a_block);
        }
      }
    }
    // different K block
#pragma unroll
    for (uint32_t k_i = 1; k_i < num_block_k; k_i++) {
      xetla_vector<dtype_src, b_reg_sizes> b_reg =
          xetla_cvt<dtype_src, dtype_b, b_reg_sizes>(
              b.reg.xetla_select<b_reg_sizes, 1>(
                  k_i * b_block_size_y * b_tile_size_x));
#pragma unroll
      for (uint32_t i = 0; i < tile_size_m / block_size_m; i++) {
        auto a_block = a.reg.xetla_select<a_block_elems, 1>(
            (i * num_block_k + k_i) * a_block_elems);
#pragma unroll
        for (uint32_t j = 0; j < num_block_n; j++) {
          auto b_block =
              b_reg.xetla_select<b_block_elems, 1>(j * b_block_elems);
          auto dst_block = dst.reg.xetla_select<block_elems, 1>(
              (i * num_block_n + j) * block_elems);
          mma_core<block_size_m, block_size_n, block_size_k>(
              dst_block, dst_block, b_block, a_block);
        }
      }
      // process the tail
      if constexpr ((tile_size_m % block_size_m) != 0) {
        constexpr uint32_t tail_start_m =
            tile_size_m / block_size_m * block_size_m;
        constexpr uint32_t a_tail_blk_w = a_tile_size_y - tail_start_m;
        constexpr uint32_t a_tail_blk_elems = a_block_size_h * a_tail_blk_w;
        constexpr uint32_t tail_size_m = tile_size_m - tail_start_m;
        constexpr uint32_t acc_tail_blk_elems = tail_size_m * block_size_n;
        auto a_block = a.reg.xetla_select<a_tail_blk_elems, 1>(
            a_tile_size_x * tail_start_m + k_i * a_tail_blk_elems);
#pragma unroll
        for (uint32_t j = 0; j < num_block_n; j++) {
          auto b_block =
              b_reg.xetla_select<b_block_elems, 1>(j * b_block_elems);
          auto dst_block = dst.reg.xetla_select<acc_tail_blk_elems, 1>(
              (tail_start_m * tile_size_n) + j * acc_tail_blk_elems);
          mma_core<tail_size_m, block_size_n, block_size_k>(
              dst_block, dst_block, b_block, a_block);
        }
      }
    }
  }
};

} // namespace gpu::xetla::subgroup
