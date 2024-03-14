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

#include <kernel/default_config/common.hpp>

namespace gpu::xetla {
namespace decision_tree_rule {

template <typename T>
struct data_type_handler {
  using type = typename T::template update_dict_t<dict_t<elem_t_t<
      tune_key::data_type_acc,
      typename T::template find_elem_t<tune_key::data_type_acc>::type>>>;
};

template <typename dict_t_>
struct tile_shape_handler {
  struct impl {
    template <
        uint32_t wg_tile_shape_n_,
        uint32_t wg_tile_shape_m_,
        uint32_t wg_tile_k_,
        uint32_t sg_tile_shape_n_,
        uint32_t sg_tile_shape_m_>
    struct tile_shape_config {
      static constexpr uint32_t wg_tile_shape_n = wg_tile_shape_n_;
      static constexpr uint32_t wg_tile_shape_m = wg_tile_shape_m_;
      static constexpr uint32_t wg_tile_k = wg_tile_k_;
      static constexpr uint32_t sg_tile_shape_n = sg_tile_shape_n_;
      static constexpr uint32_t sg_tile_shape_m = sg_tile_shape_m_;

      using to_dict = dict_t<
          elem_t_t<
              tune_key::wg_tile_shape,
              shape<wg_tile_shape_n, wg_tile_shape_m>>,
          elem_v_t<tune_key::wg_tile_k, wg_tile_k>,
          elem_t_t<
              tune_key::sg_tile_shape,
              shape<sg_tile_shape_n, sg_tile_shape_m>>>;
    };

    static constexpr int const_abs(const int& z) {
      if (z >= 0) {
        return z;
      }
      return -z;
    }

    template <typename T, typename U>
    static constexpr int distance_fcn() {
      int sum = 0;
      sum += const_abs(T::wg_tile_shape_m - U::wg_tile_shape_m);
      sum += const_abs(T::wg_tile_shape_n - U::wg_tile_shape_n);
      sum += const_abs(T::wg_tile_k - U::wg_tile_k);
      sum += const_abs(T::sg_tile_shape_m - U::sg_tile_shape_m);
      sum += const_abs(T::sg_tile_shape_n - U::sg_tile_shape_n);
      return sum;
    }

    using wg_256x256_k32_sg_32x64 = tile_shape_config<256, 256, 32, 32, 64>;
    using wg_256x256_k32_sg_64x32 = tile_shape_config<256, 256, 32, 64, 32>;
    using wg_128x512_k16_sg_32x64 = tile_shape_config<128, 512, 16, 32, 64>;
    using wg_512x128_k16_sg_64x32 = tile_shape_config<512, 128, 16, 64, 32>;
    using wg_32x256_k32_sg_16x16 = tile_shape_config<32, 256, 32, 16, 16>;
    using wg_512x64_k32_sg_32x32 = tile_shape_config<512, 64, 32, 32, 32>;
    using wg_64x64_k32_sg_16x8 = tile_shape_config<64, 64, 32, 16, 8>;

    template <typename ref, typename... elems>
    struct find_min_elem {
      using type = ref;
      static constexpr int distance = -1;
    };

    template <typename ref, typename elem, typename... elems>
    struct find_min_elem<ref, elem, elems...> {
      using cur_type = ref;
      static constexpr int cur_distance = distance_fcn<ref, elem>();
      using nxt = find_min_elem<ref, elems...>;
      static constexpr bool use_next =
          (sizeof...(elems) > 0) && (cur_distance > nxt::distance);
      using type = typename std::
          conditional<use_next, typename nxt::type, cur_type>::type;
      static constexpr int distance = use_next ? nxt::distance : cur_distance;
    };

    template <typename ref, typename elem>
    struct find_min_elem<ref, elem> {
      using type = elem;
      static constexpr int distance = distance_fcn<ref, elem>();
    };

    template <typename T>
    struct from_dict_impl {
      using wg_tile_shape =
          typename T::template find_elem_t<tune_key::wg_tile_shape>::type;
      using sg_tile_shape =
          typename T::template find_elem_t<tune_key::sg_tile_shape>::type;
      static constexpr uint32_t wg_tile_shape_n =
          wg_tile_shape::template dim<0>();
      static constexpr uint32_t wg_tile_shape_m =
          wg_tile_shape::template dim<1>();
      static constexpr uint32_t wg_tile_k =
          T::template find_elem_v<tune_key::wg_tile_k>;
      static constexpr uint32_t sg_tile_shape_n =
          sg_tile_shape::template dim<0>();
      static constexpr uint32_t sg_tile_shape_m =
          sg_tile_shape::template dim<1>();

      using type = tile_shape_config<
          wg_tile_shape_n,
          wg_tile_shape_m,
          wg_tile_k,
          sg_tile_shape_n,
          sg_tile_shape_m>;
    };

    template <typename T>
    using from_dict = typename from_dict_impl<T>::type;

    struct update_config_impl {
      using orig = from_dict<dict_t_>;
      using type = typename find_min_elem<
          orig,
          wg_256x256_k32_sg_32x64,
          wg_256x256_k32_sg_64x32,
          wg_128x512_k16_sg_32x64,
          wg_512x128_k16_sg_64x32,
          wg_32x256_k32_sg_16x16,
          wg_512x64_k32_sg_32x32,
          wg_64x64_k32_sg_16x8>::type;
    };
  };

  using update_config = typename impl::update_config_impl::type;
  using type =
      typename dict_t_::template update_dict_t<typename update_config::to_dict>;
};

template <typename dict_t_>
struct kslicing_handler {
  struct impl {
    template <
        uint32_t global_kslicing_ratio_,
        uint32_t local_kslicing_ratio_,
        uint32_t wg_tile_shape_n_,
        uint32_t wg_tile_shape_m_,
        uint32_t wg_tile_k_,
        uint32_t sg_tile_shape_n_,
        uint32_t sg_tile_shape_m_>
    struct kslicing_config {
      static constexpr uint32_t global_kslicing_ratio = global_kslicing_ratio_;
      static constexpr uint32_t local_kslicing_ratio = local_kslicing_ratio_;
      static constexpr uint32_t wg_tile_shape_n = wg_tile_shape_n_;
      static constexpr uint32_t wg_tile_shape_m = wg_tile_shape_m_;
      static constexpr uint32_t wg_tile_k = wg_tile_k_;
      static constexpr uint32_t sg_tile_shape_n = sg_tile_shape_n_;
      static constexpr uint32_t sg_tile_shape_m = sg_tile_shape_m_;

      using this_t = kslicing_config<
          global_kslicing_ratio,
          local_kslicing_ratio,
          wg_tile_shape_n,
          wg_tile_shape_m,
          wg_tile_k,
          sg_tile_shape_n,
          sg_tile_shape_m>;

      using to_dict = dict_t<
          elem_v_t<tune_key::global_kslicing_ratio, global_kslicing_ratio>,
          elem_v_t<tune_key::local_kslicing_ratio, local_kslicing_ratio>,
          elem_t_t<
              tune_key::wg_tile_shape,
              shape<wg_tile_shape_n, wg_tile_shape_m>>,
          elem_v_t<tune_key::wg_tile_k, wg_tile_k>,
          elem_t_t<
              tune_key::sg_tile_shape,
              shape<sg_tile_shape_n, sg_tile_shape_m>>,
          elem_v_t<
              tune_key::dispatch_policy,
              tune_key_value::dispatch_policy_kslicing>>;

      template <template <typename> typename G>
      using apply = typename G<this_t>::type;
    };

    template <typename T>
    struct from_dict_impl {
      static constexpr uint32_t global_kslicing_ratio =
          T::template find_elem_v<tune_key::global_kslicing_ratio>;
      static constexpr uint32_t local_kslicing_ratio =
          T::template find_elem_v<tune_key::local_kslicing_ratio>;
      using wg_tile_shape =
          typename T::template find_elem_t<tune_key::wg_tile_shape>::type;
      using sg_tile_shape =
          typename T::template find_elem_t<tune_key::sg_tile_shape>::type;
      static constexpr uint32_t wg_tile_shape_n =
          wg_tile_shape::template dim<0>();
      static constexpr uint32_t wg_tile_shape_m =
          wg_tile_shape::template dim<1>();
      static constexpr uint32_t wg_tile_k =
          T::template find_elem_v<tune_key::wg_tile_k>;
      static constexpr uint32_t sg_tile_shape_n =
          sg_tile_shape::template dim<0>();
      static constexpr uint32_t sg_tile_shape_m =
          sg_tile_shape::template dim<1>();

      using type = kslicing_config<
          global_kslicing_ratio,
          local_kslicing_ratio,
          wg_tile_shape_n,
          wg_tile_shape_m,
          wg_tile_k,
          sg_tile_shape_n,
          sg_tile_shape_m>;
    };

    template <typename T>
    using from_dict = typename from_dict_impl<T>::type;

    struct update_config_impl {
      using orig = from_dict<dict_t_>;

      template <typename T>
      struct local_kslicing_handler {
        static constexpr uint32_t global_kslicing_ratio =
            T::global_kslicing_ratio;
        static constexpr uint32_t local_kslicing_ratio =
            T::local_kslicing_ratio;

        static constexpr uint32_t wg_tile_shape_n =
            (local_kslicing_ratio == 2) ? 128 : T::wg_tile_shape_n;
        static constexpr uint32_t wg_tile_shape_m =
            (local_kslicing_ratio == 2) ? 64 : T::wg_tile_shape_m;
        static constexpr uint32_t wg_tile_k =
            (local_kslicing_ratio == 2) ? 32 : T::wg_tile_k;
        static constexpr uint32_t sg_tile_shape_n =
            (local_kslicing_ratio == 2) ? 32 : T::sg_tile_shape_n;
        static constexpr uint32_t sg_tile_shape_m =
            (local_kslicing_ratio == 2) ? 16 : T::sg_tile_shape_m;

        using type = typename dict_t<
            elem_t_t<
                1U,
                kslicing_config<
                    global_kslicing_ratio,
                    local_kslicing_ratio,
                    wg_tile_shape_n,
                    wg_tile_shape_m,
                    wg_tile_k,
                    sg_tile_shape_n,
                    sg_tile_shape_m>>,
            elem_t_t<
                2U,
                kslicing_config<
                    global_kslicing_ratio,
                    local_kslicing_ratio,
                    128,
                    64,
                    32,
                    32,
                    16>>,
            elem_t_t<
                4U,
                kslicing_config<
                    global_kslicing_ratio,
                    local_kslicing_ratio,
                    64,
                    64,
                    32,
                    32,
                    16>>,
            elem_t_t<
                8U,
                kslicing_config<
                    global_kslicing_ratio,
                    local_kslicing_ratio,
                    64,
                    32,
                    32,
                    32,
                    16>>,
            elem_t_t<
                16U,
                kslicing_config<
                    global_kslicing_ratio,
                    local_kslicing_ratio,
                    64,
                    16,
                    32,
                    32,
                    16>>>::template find_elem_t<local_kslicing_ratio>::type;
      };

      using type = typename orig::template apply<local_kslicing_handler>;
    };
  };

  using update_config = typename impl::update_config_impl::type;
  using type = typename std::conditional<
      (dict_t_::template find_elem_v<tune_key::dispatch_policy> ==
       tune_key_value::dispatch_policy_kslicing),
      typename dict_t_::template update_dict_t<typename update_config::to_dict>,
      dict_t_>::type;
};
} // namespace decision_tree_rule

template <typename dict, typename opt_dict>
struct fallback_optimizer {
  using type = typename opt_dict::template update_t<
      elem_t_t<tune_key::data_type_a, data_type_a_t<dict>>,
      elem_t_t<tune_key::data_type_b, data_type_b_t<dict>>,
      elem_t_t<tune_key::data_type_c, data_type_c_t<dict>>,
      elem_v_t<tune_key::memory_layout_a, memory_layout_a_v<dict>>,
      elem_v_t<tune_key::memory_layout_b, memory_layout_b_v<dict>>,
      elem_v_t<tune_key::memory_layout_c, memory_layout_c_v<dict>>,
      elem_v_t<tune_key::memory_alignment_a, memory_alignment_a_v<dict>>,
      elem_v_t<tune_key::memory_alignment_b, memory_alignment_b_v<dict>>,
      elem_v_t<tune_key::memory_alignment_c, memory_alignment_c_v<dict>>,
      elem_v_t<tune_key::gpu_arch, gpu_arch_v<dict>>>;
};

template <
    param_optimizer_tag tag_,
    typename dict_t_,
    param_optimizer_level mode_,
    typename... candidates_t>
struct decision_tree_optimizer : param_optimizer_base {
  struct impl {
    template <typename T, template <typename> typename G>
    using apply_handeler = T::template update_generator_t<G>;
    static constexpr bool keep_shape =
        (mode_ == param_optimizer_level::keep_shape);

    using t0 = dict_t_;
    using t1 = apply_handeler<t0, decision_tree_rule::data_type_handler>;
    using t2_0 = apply_handeler<t1, decision_tree_rule::tile_shape_handler>;
    using t2 = std::conditional_t<keep_shape, t1, t2_0>;
    using t3 = apply_handeler<t2, decision_tree_rule::kslicing_handler>;

    using type = t3;

    // If any of data_type / mem_layout / mem_align is changed,
    // then change it back via fallback_optimizer
    using fallback_type = fallback_optimizer<dict_t_, type>;
  };
  static constexpr bool use_fallback =
      !(param_optimizer_base::
            template valid_attribute_v<dict_t_, typename impl::type>);
  using type = typename std::
      conditional_t<use_fallback, typename impl::fallback_type, impl>::type;
};

} // namespace gpu::xetla
