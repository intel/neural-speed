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
#include <memory>
#include <tuple>

#include "layers/mha_dense.h"
#include "layers/mha_dense_wrapper.h"
#include "layers/ne_test_layers_utils.hpp"

#ifndef NS_TESTS
static_assert(false, "Only compile this source file for testing!");
#endif

using namespace ne_bestla::custom::mha;  // NOLINT

#define CheckISA(ISA) \
  (bestla::device::CpuDevice::getInstance()->ISA() || (printf("Wrong Device ISA: " #ISA "\n"), false))

namespace {
bool ret_ok = true;

class test_mha_dese_t {
 public:
  test_mha_dese_t() {
    printf("Test suit: %s\n", __FUNCTION__);
    GetCPUDevice();
    static const int max_threads = std::thread::hardware_concurrency();
    ne_threading::get()->set_threads(std::min(_cd->getThreads(), max_threads));

#if CompileFP16()
    if (CheckISA(AMX_BF16)) {
      ret_ok &= test_case<float, fp16, fp16, float>({1, 1, 1, 32, 128, 64}, NE_ATTN_FLAG_NONE);
      ret_ok &= test_case<float, fp16, fp16, float>({2, 5, 5, 32, 64, 128}, NE_ATTN_FLAG_NONE);
      ret_ok &= test_case<float, fp16, fp16, float>({2, 5, 5, 80, 128, 77}, NE_ATTN_FLAG_NONE);
      ret_ok &= test_case<float, fp16, fp16, float>({1, 1, 1, 32, 63, 63}, NE_ATTN_FLAG_NONE);
      ret_ok &= test_case<float, fp16, fp16, float>({3, 4, 4, 256, 1, 384}, NE_ATTN_FLAG_NONE);
      ret_ok &= test_case<float, fp16, fp16, float>({1, 1, 1, 64, 64, 64}, NE_ATTN_FLAG_IS_CAUSAL);

      ret_ok &= test_case<fp16, fp16, fp16, fp16>({1, 1, 1, 32, 128, 64}, NE_ATTN_FLAG_NONE, true);
      ret_ok &= test_case<fp16, fp16, fp16, fp16>({2, 5, 5, 32, 64, 128}, NE_ATTN_FLAG_NONE, true);
      ret_ok &= test_case<fp16, fp16, fp16, fp16>({2, 5, 5, 80, 128, 77}, NE_ATTN_FLAG_NONE, true);
      ret_ok &= test_case<fp16, fp16, fp16, fp16>({1, 1, 1, 256, 63, 63}, NE_ATTN_FLAG_NONE, true);
      ret_ok &= test_case<fp16, fp16, fp16, fp16>({3, 4, 4, 256, 1, 384}, NE_ATTN_FLAG_NONE, true);
      ret_ok &= test_case<fp16, fp16, fp16, fp16>({1, 1, 1, 64, 64, 64}, NE_ATTN_FLAG_IS_CAUSAL, true);

      ret_ok &= test_case<float, fp16, fp16, float>({1, 1, 1, 32, 128, 64}, NE_ATTN_FLAG_NONE, true);
      ret_ok &= test_case<float, fp16, fp16, float>({2, 5, 5, 32, 64, 128}, NE_ATTN_FLAG_NONE, true);
      ret_ok &= test_case<float, fp16, fp16, float>({2, 5, 5, 80, 128, 77}, NE_ATTN_FLAG_NONE, true);
      ret_ok &= test_case<float, fp16, fp16, float>({1, 1, 1, 256, 63, 63}, NE_ATTN_FLAG_NONE, true);
      ret_ok &= test_case<float, fp16, fp16, float>({3, 4, 4, 256, 1, 384}, NE_ATTN_FLAG_NONE, true);
      ret_ok &= test_case<float, fp16, fp16, float>({1, 1, 1, 64, 64, 64}, NE_ATTN_FLAG_IS_CAUSAL, true);
    }
#endif

    if (CheckISA(AMX_BF16)) {
      const auto BA48b4a = ATTN_FWD_LAYOUT_NTILE48_ROWPACK4;
      ret_ok &= test_case<int8_t, int8_t, int8_t, int8_t>({1, 1, 1, 32, 128, 64}, NE_ATTN_FLAG_NONE, false, BA48b4a);
      ret_ok &= test_case<int8_t, int8_t, int8_t, int8_t>({2, 5, 5, 32, 64, 128}, NE_ATTN_FLAG_NONE, false, BA48b4a);
      ret_ok &= test_case<int8_t, int8_t, int8_t, int8_t>({2, 5, 5, 80, 128, 77}, NE_ATTN_FLAG_NONE, false, BA48b4a);
      ret_ok &= test_case<int8_t, int8_t, int8_t, int8_t>({1, 1, 1, 256, 63, 63}, NE_ATTN_FLAG_NONE, false, BA48b4a);
      ret_ok &= test_case<int8_t, int8_t, int8_t, int8_t>({3, 4, 4, 256, 1, 384}, NE_ATTN_FLAG_NONE, false, BA48b4a);
      ret_ok &=
          test_case<int8_t, int8_t, int8_t, int8_t>({1, 1, 1, 64, 64, 64}, NE_ATTN_FLAG_IS_CAUSAL, false, BA48b4a);
    }

    if (CheckISA(AMX_BF16)) {
      const auto BA48b2a = ATTN_FWD_LAYOUT_NTILE48_ROWPACK2;
      int flags = NE_ATTN_FLAG_NONE;
      ret_ok &= test_case<float, bf16, bf16, float>({1, 1, 1, 32, 128, 64}, flags, false, BA48b2a, 1e-3f);
      ret_ok &= test_case<float, bf16, bf16, float>({2, 5, 5, 32, 64, 128}, flags, false, BA48b2a, 1e-3f);
      ret_ok &= test_case<float, bf16, bf16, float>({2, 5, 5, 80, 128, 77}, flags, false, BA48b2a, 1e-3f);
      ret_ok &= test_case<float, bf16, bf16, float>({1, 1, 1, 256, 63, 63}, flags, false, BA48b2a, 1e-3f);
      ret_ok &= test_case<float, bf16, bf16, float>({3, 4, 4, 256, 1, 384}, flags, false, BA48b2a, 1e-3f);

      flags |= NE_ATTN_FLAG_IS_CAUSAL;
      ret_ok &= test_case<float, bf16, bf16, float>({1, 1, 1, 64, 64, 64}, flags, false, BA48b2a, 1e-3f);
    }

    if (CheckISA(AVX512F)) {  // PREFER_FP32
      const auto BA48b2a = ATTN_FWD_LAYOUT_NTILE48_ROWPACK2;
      int flags = NE_ATTN_FLAG_PREFER_FP32;
      ret_ok &= test_case<float, bf16, bf16, float>({1, 1, 1, 32, 128, 64}, flags, false, BA48b2a, 1e-3f);
      ret_ok &= test_case<float, bf16, bf16, float>({2, 5, 5, 32, 64, 128}, flags, false, BA48b2a, 1e-3f);
      ret_ok &= test_case<float, bf16, bf16, float>({2, 5, 5, 80, 128, 77}, flags, false, BA48b2a, 1e-3f);
      ret_ok &= test_case<float, bf16, bf16, float>({1, 1, 1, 256, 63, 63}, flags, false, BA48b2a, 1e-3f);
      ret_ok &= test_case<float, bf16, bf16, float>({3, 4, 4, 256, 1, 384}, flags, false, BA48b2a, 1e-3f);

      flags |= NE_ATTN_FLAG_IS_CAUSAL;
      ret_ok &= test_case<float, bf16, bf16, float>({1, 1, 1, 64, 64, 64}, flags, false, BA48b2a, 1e-3f);
    }
    if (CheckISA(AVX2)) {  // avx2
      const auto Ba24b = ATTN_FWD_LAYOUT_NTILE24_ROWPACK1;
      int flags = NE_ATTN_FLAG_PREFER_FP32;
      ret_ok &= test_case<float, fp16, fp16, float>({1, 1, 1, 32, 128, 64}, flags, false, Ba24b, 1e-3f);
      ret_ok &= test_case<float, fp16, fp16, float>({2, 5, 5, 32, 64, 128}, flags, false, Ba24b, 1e-3f);
      ret_ok &= test_case<float, fp16, fp16, float>({2, 5, 5, 80, 128, 77}, flags, false, Ba24b, 1e-3f);
      ret_ok &= test_case<float, fp16, fp16, float>({1, 1, 1, 256, 63, 63}, flags, false, Ba24b, 1e-3f);
      ret_ok &= test_case<float, fp16, fp16, float>({3, 4, 4, 256, 1, 384}, flags, false, Ba24b, 1e-3f);

      flags |= NE_ATTN_FLAG_IS_CAUSAL;
      ret_ok &= test_case<float, fp16, fp16, float>({1, 1, 1, 64, 64, 64}, flags, false, Ba24b, 1e-3f);
    }

    {  // amxbf16 => avx2 fallback
      int flags = NE_ATTN_FLAG_NONE;
      ret_ok &= test_reorder_pipe<float, float, float, float>({1, 1, 1, 32, 128, 64}, 64, flags);
      ret_ok &= test_reorder_pipe<float, float, float, float>({2, 5, 5, 32, 64, 128}, 256, flags);
      ret_ok &= test_reorder_pipe<float, float, float, float>({2, 5, 5, 80, 128, 77}, 256, flags);
      ret_ok &= test_reorder_pipe<float, float, float, float>({2, 5, 1, 80, 128, 77}, 256, flags);
      ret_ok &= test_reorder_pipe<float, float, float, float>({1, 1, 1, 256, 63, 63}, 256, flags);
      ret_ok &= test_reorder_pipe<float, float, float, float>({3, 4, 4, 256, 1, 384}, 384, flags);
      ret_ok &= test_reorder_pipe<float, float, float, float>({3, 4, 2, 256, 1, 384}, 384, flags);
      flags |= NE_ATTN_FLAG_IS_CAUSAL;
      ret_ok &= test_reorder_pipe<float, float, float, float>({1, 1, 1, 64, 64, 64}, 128, flags);
      flags |= NE_ATTN_FLAG_IS_ALIBI8;
      ret_ok &= test_reorder_pipe<float, float, float, float>({1, 8, 8, 64, 64, 64}, 128, flags);
    }
    printf("Test suit done: %s\n", __FUNCTION__);
  }

  template <class T>
  static constexpr float init_min_val = std::is_same<T, int8_t>::value    ? -127.f
                                        : std::is_same<T, uint8_t>::value ? 0.f
                                                                          : -1.f;
  template <class T>
  static constexpr float init_max_val = std::is_same<T, int8_t>::value    ? 127.f
                                        : std::is_same<T, uint8_t>::value ? 255.f
                                                                          : 1.f;
  template <class T>
  static constexpr float init_scale_val = 1.f / init_max_val<T>;

#ifdef _MSC_VER
#define __PRETTY_FUNCTION__ __FUNCSIG__
#endif

  template <class Q_T, class K_T, class V_T, class DST_T>
  bool test_case(const attn_shape_t& s, ne_attn_flags_t flags, bool k_trans = false,
                 ATTN_FWD_LAYOUT kv_layout = ATTN_FWD_LAYOUT_PLAIN, float eps = 1e-2f) {
    assert(kv_layout == ATTN_FWD_LAYOUT_PLAIN || !k_trans);
    const auto batch_size = s.batch_size;
    const auto head_num = s.head_num;
    const auto heads_kv = s.heads_kv;
    const auto head_size = s.head_size;
    const auto sl_q = s.sl_q;
    const auto sl_kv = s.sl_kv;
    assert(("GQA not supported!", s.head_num == s.heads_kv));

    const auto is_causal = flags & NE_ATTN_FLAG_IS_CAUSAL ? "maksed" : "unmask";
    const auto is_alibi8 = flags & NE_ATTN_FLAG_IS_ALIBI8 ? "alibi8" : "";
    const auto prefer_fp32 = flags & NE_ATTN_FLAG_PREFER_FP32 ? "FP32" : "";
    printf("\ntest_case: %s\t", __PRETTY_FUNCTION__);
    printf("bs_%d hn_%d hkv_%d hs_%d sl_q_%d sk_kv_%d %s %s %s\n", batch_size, head_num, heads_kv, head_size, sl_q,
           sl_kv, is_causal, is_alibi8, prefer_fp32);

    const auto NTILE = kv_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK4   ? 48
                       : kv_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2 ? 48
                       : kv_layout == ATTN_FWD_LAYOUT_NTILE24_ROWPACK1 ? 24
                                                                       : 0;
    const auto ROWPACK = kv_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK4   ? 4
                         : kv_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2 ? 2
                         : kv_layout == ATTN_FWD_LAYOUT_NTILE24_ROWPACK1 ? 1
                                                                         : 0;
    const auto ROWPAD = ROWPACK > 1 ? ROWPACK * 16 : 1;
    const auto k_rows_pad = kv_layout != ATTN_FWD_LAYOUT_PLAIN ? padto(head_size, ROWPAD) : head_size;
    const auto k_cols_pad = kv_layout != ATTN_FWD_LAYOUT_PLAIN ? padto(sl_kv, NTILE) : sl_kv;
    const auto v_rows_pad = kv_layout != ATTN_FWD_LAYOUT_PLAIN ? padto(sl_kv, ROWPAD) : sl_kv;
    const auto v_cols_pad = kv_layout != ATTN_FWD_LAYOUT_PLAIN ? padto(head_size, NTILE) : head_size;

    std::vector<Q_T> src_q(batch_size * head_num * sl_q * head_size);
    std::vector<K_T> src_k(batch_size * heads_kv * k_rows_pad * k_cols_pad);
    std::vector<V_T> src_v(batch_size * heads_kv * v_rows_pad * v_cols_pad);
    std::vector<DST_T> dst(batch_size * head_num * sl_q * head_size);
    std::vector<DST_T> ref(batch_size * head_num * sl_q * head_size);  // reference result
    std::vector<char> tmp(bestla_fusion_attn_workspace_size(&s));

    // init vector
    static std::mt19937 rng(1);
    std::uniform_int_distribution<> dist;
    init_vector(&src_q, init_min_val<Q_T>, init_max_val<Q_T>, dist(rng));
    init_vector(&src_k, init_min_val<K_T>, init_max_val<K_T>, dist(rng));
    init_vector(&src_v, init_min_val<V_T>, init_max_val<V_T>, dist(rng));

    // pad0 for padded layouts
    if (kv_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK4 || kv_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2 ||
        kv_layout == ATTN_FWD_LAYOUT_NTILE24_ROWPACK1) {
#pragma omp parallel for collapse(2)
      for (int ibs = 0; ibs < batch_size; ++ibs) {
        for (int ihn = 0; ihn < heads_kv; ++ihn) {
          // K
          const auto k_off = (ibs * heads_kv + ihn) * k_rows_pad * k_cols_pad;
          for (int i = 0; i < k_rows_pad; ++i) {
            for (int j = 0; j < k_cols_pad; ++j) {
              if (i < head_size && j < sl_kv) continue;

              const auto j_remain = j % NTILE;
              const auto j_block = j - j_remain;
              const auto i_remain = i % ROWPACK;
              const auto i_block = i - i_remain;
              src_k[k_off + j_block * k_rows_pad + i_block * NTILE + j_remain * ROWPACK + i_remain] = K_T(0);
            }
          }
          // V
          const auto v_off = (ibs * heads_kv + ihn) * v_rows_pad * v_cols_pad;
          for (int i = 0; i < v_rows_pad; ++i) {
            for (int j = 0; j < v_cols_pad; ++j) {
              if (i < sl_kv && j < head_size) continue;

              const auto j_remain = j % NTILE;
              const auto j_block = j - j_remain;
              const auto i_remain = i % ROWPACK;
              const auto i_block = i - i_remain;
              src_v[v_off + j_block * v_rows_pad + i_block * NTILE + j_remain * ROWPACK + i_remain] = V_T(0);
            }
          }
        }
      }
    }

    attn_fwd_args_t<Q_T, K_T, V_T, DST_T> args{
        /* .Q = */ src_q.data(),
        /* .K = */ src_k.data(),
        /* .V = */ src_v.data(),
        /* .dst = */ ref.data(),
        /* .Q_sc = */ init_scale_val<Q_T>,
        /* .K_sc = */ init_scale_val<K_T>,
        /* .V_sc = */ init_scale_val<V_T>,
        /* .dst_sc = */ init_scale_val<V_T>,
        /* .tmp = */ tmp.data(),
        /* .QK_scale = */ 1.f / sqrtf(static_cast<float>(head_size)),
        /* .attn_flags = */ flags,
        /* .batch_size = */ batch_size,
        /* .head_num = */ head_num,
        /* .heads_kv = */ heads_kv,
        /* .head_size = */ head_size,
        /* .sl_q = */ sl_q,
        /* .sl_kv = */ sl_kv,
        /* .Q_layout = */ ATTN_FWD_LAYOUT_PLAIN,
        /* .K_layout = */ kv_layout,
        /* .V_layout = */ kv_layout,
        /* .dst_layout = */ ATTN_FWD_LAYOUT_PLAIN,
        /* .step_q_bs = */ sl_q * head_num * head_size,
        /* .step_q_head_num = */ head_size,
        /* .step_q_sl = */ head_num * head_size,
        /* .step_k_bs = */ sl_kv * heads_kv * head_size,
        /* .step_k_head_num = */ k_trans ? head_size * sl_kv : head_size,
        /* .step_k_sl = */ k_trans ? 1 : heads_kv * head_size,
        /* .step_k_head_size = */ k_trans ? sl_kv : 1,
        /* .step_v_bs = */ sl_kv * heads_kv * head_size,
        /* .step_v_head_num = */ head_size,
        /* .step_v_sl = */ heads_kv * head_size,
        /* .step_v_head_size = */ 1,
        /* .step_dst_bs = */ sl_q * head_num * head_size,
        /* .step_dst_head_num = */ head_size,
        /* .step_dst_sl = */ head_num * head_size,
    };
    if (kv_layout != ATTN_FWD_LAYOUT_PLAIN) {
      args.step_k_bs = heads_kv * k_rows_pad * k_cols_pad;
      args.step_k_head_num = k_rows_pad * k_cols_pad;
      args.step_k_sl = k_rows_pad;
      args.step_k_head_size = NTILE;
      args.step_v_bs = heads_kv * v_rows_pad * v_cols_pad;
      args.step_v_head_num = v_rows_pad * v_cols_pad;
      args.step_v_sl = NTILE;
      args.step_v_head_size = v_rows_pad;
    }

    bestla_fusion_attn_forward_ref(args);

    args.dst = dst.data();
    bestla_fusion_attn_forward(args);

    // Check result
    return compare_data(dst.data(), ref.data(), dst.size(), eps);
  }

  template <class Q_T, class K_T, class V_T, class DST_T>
  bool test_reorder_pipe(const attn_shape_t& s, int sl_kv_max, ne_attn_flags_t flags) {
    const auto batch_size = s.batch_size;
    const auto head_num = s.head_num;
    const auto heads_kv = s.heads_kv;
    const auto head_size = s.head_size;
    const auto sl_q = s.sl_q;
    const auto sl_kv = s.sl_kv;
    assert(("head_num must be a multiple of heads_kv!", head_num % heads_kv == 0));

    const auto is_causal = flags & NE_ATTN_FLAG_IS_CAUSAL ? "maksed" : "unmask";
    const auto is_alibi8 = flags & NE_ATTN_FLAG_IS_ALIBI8 ? "alibi8" : "";
    const auto prefer_fp32 = flags & NE_ATTN_FLAG_PREFER_FP32 ? "FP32" : "";
    printf("\ntest_case: %s\t", __PRETTY_FUNCTION__);
    printf("bs_%d hn_%d hkv_%d hs_%d sl_q_%d sk_kv_%d %s %s %s\n", batch_size, head_num, heads_kv, head_size, sl_q,
           sl_kv, is_causal, is_alibi8, prefer_fp32);

    assert(sl_kv_max >= sl_kv);

    kv_shape_t kv_shape = {
        /* .heads_kv */ static_cast<uint32_t>(heads_kv),
        /* .head_size */ static_cast<uint32_t>(head_size),
        /* .sl_kv_max */ static_cast<uint32_t>(sl_kv_max),
    };
    kv_cache_info_t kv_cache_info;
    bestla_reordered_attn_fp32_batch_kv_info(&kv_shape, &kv_cache_info);
    assert(kv_cache_info.k_layout >= kv_cache_info.v_layout);
    const auto kv_layout = kv_cache_info.k_layout;
    const auto NTILE = kv_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK4   ? 48
                       : kv_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2 ? 48
                       : kv_layout == ATTN_FWD_LAYOUT_NTILE24_ROWPACK1 ? 24
                                                                       : 0;
    const auto ROWPACK = kv_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK4   ? 4
                         : kv_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2 ? 2
                         : kv_layout == ATTN_FWD_LAYOUT_NTILE24_ROWPACK1 ? 1
                                                                         : 0;
    const auto ROWPAD = ROWPACK > 1 ? ROWPACK * 16 : 1;
    const auto k_rows_pad = kv_layout != ATTN_FWD_LAYOUT_PLAIN ? padto(head_size, ROWPAD) : head_size;
    const auto k_cols_pad = kv_layout != ATTN_FWD_LAYOUT_PLAIN ? padto(sl_kv, NTILE) : sl_kv;
    const auto v_rows_pad = kv_layout != ATTN_FWD_LAYOUT_PLAIN ? padto(sl_kv, ROWPAD) : sl_kv;
    const auto v_cols_pad = kv_layout != ATTN_FWD_LAYOUT_PLAIN ? padto(head_size, NTILE) : head_size;

    std::vector<Q_T> src_q(batch_size * head_num * sl_q * head_size);
    std::vector<K_T> src_k(batch_size * heads_kv * sl_kv * head_size);
    std::vector<V_T> src_v(batch_size * heads_kv * sl_kv * head_size);
    std::vector<char> k_cache(batch_size * kv_cache_info.k_bytes);
    std::vector<char> v_cache(batch_size * kv_cache_info.v_bytes);
    std::vector<DST_T> dst(batch_size * head_num * sl_q * head_size);
    std::vector<DST_T> ref(batch_size * head_num * sl_q * head_size);  // reference result
    std::vector<char> tmp(bestla_fusion_attn_workspace_size(&s));

    // init vector
    static std::mt19937 rng(1);
    std::uniform_int_distribution<> dist;
    init_vector(&src_q, init_min_val<Q_T>, init_max_val<Q_T>, dist(rng));
    init_vector(&src_k, init_min_val<K_T>, init_max_val<K_T>, dist(rng));
    init_vector(&src_v, init_min_val<V_T>, init_max_val<V_T>, dist(rng));

    // undefined values
    init_vector(&k_cache, INT8_MIN, INT8_MAX, dist(rng));
    init_vector(&v_cache, INT8_MIN, INT8_MAX, dist(rng));

    int step_src_k_bs = sl_kv * heads_kv * head_size;
    int step_src_k_head_num = head_size;
    int step_src_k_sl = heads_kv * head_size;
    int step_src_k_head_size = 1;
    int step_src_v_bs = sl_kv * heads_kv * head_size;
    int step_src_v_head_num = head_size;
    int step_src_v_sl = heads_kv * head_size;
    int step_src_v_head_size = 1;
    attn_fwd_args_t<Q_T, K_T, V_T, DST_T> ref_args{
        /* .Q = */ src_q.data(),
        /* .K = */ src_k.data(),
        /* .V = */ src_v.data(),
        /* .dst = */ ref.data(),
        /* .Q_sc = */ init_scale_val<Q_T>,
        /* .K_sc = */ init_scale_val<K_T>,
        /* .V_sc = */ init_scale_val<V_T>,
        /* .dst_sc = */ init_scale_val<V_T>,
        /* .tmp = */ tmp.data(),
        /* .QK_scale = */ 1.f / sqrtf(static_cast<float>(head_size)),
        /* .attn_flags = */ flags,
        /* .batch_size = */ batch_size,
        /* .head_num = */ head_num,
        /* .heads_kv = */ heads_kv,
        /* .head_size = */ head_size,
        /* .sl_q = */ sl_q,
        /* .sl_kv = */ sl_kv,
        /* .Q_layout = */ ATTN_FWD_LAYOUT_PLAIN,
        /* .K_layout = */ ATTN_FWD_LAYOUT_PLAIN,
        /* .V_layout = */ ATTN_FWD_LAYOUT_PLAIN,
        /* .dst_layout = */ ATTN_FWD_LAYOUT_PLAIN,
        /* .step_q_bs = */ sl_q * head_num * head_size,
        /* .step_q_head_num = */ head_size,
        /* .step_q_sl = */ head_num * head_size,

        /* .step_k_bs = */ step_src_k_bs,
        /* .step_k_head_num = */ step_src_k_head_num,
        /* .step_k_sl = */ step_src_k_sl,
        /* .step_k_head_size = */ step_src_k_head_size,
        /* .step_v_bs = */ step_src_v_bs,
        /* .step_v_head_num = */ step_src_v_head_num,
        /* .step_v_sl = */ step_src_v_sl,
        /* .step_v_head_size = */ step_src_v_head_size,

        /* .step_dst_bs = */ sl_q * head_num * head_size,
        /* .step_dst_head_num = */ head_size,
        /* .step_dst_sl = */ head_num * head_size,
    };
    bestla_fusion_attn_forward_ref(ref_args);

    if (std::is_same<std::tuple<Q_T, K_T, V_T, DST_T>, std::tuple<float, float, float, float>>::value) {
      assert(kv_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2 || kv_layout == ATTN_FWD_LAYOUT_NTILE24_ROWPACK1);
      // for testing, first reorder sl_kv - 1 and than concat the last 1 line
      const auto seq_size_first = sl_kv - 1;
      const auto seq_size_next = 1;
      bestla_fusion_attn_fp32_update_kv_args_t update_k_args = {
          /* .src = */ src_k.data(),
          /* .cache = */ k_cache.data(),
          /* .batch_size = */ batch_size,
          /* .heads_kv = */ heads_kv,
          /* .head_size = */ head_size,
          /* .seq_off = */ 0,
          /* .seq_size = */ seq_size_first,
          /* .seq_max = */ sl_kv_max,
          /* .step_bs = */ step_src_k_bs,
          /* .step_head_num = */ step_src_k_head_num,
          /* .step_seq = */ step_src_k_sl,
          /* .step_head_size = */ step_src_k_head_size,
      };
      bestla_reordered_attn_fp32_update_k(&update_k_args);

      bestla_fusion_attn_fp32_update_kv_args_t update_v_args = {
          /* .src = */ src_v.data(),
          /* .cache = */ v_cache.data(),
          /* .batch_size = */ batch_size,
          /* .heads_kv = */ heads_kv,
          /* .head_size = */ head_size,
          /* .seq_off = */ 0,
          /* .seq_size = */ seq_size_first,
          /* .seq_max = */ sl_kv_max,
          /* .step_bs = */ step_src_v_bs,
          /* .step_head_num = */ step_src_v_head_num,
          /* .step_seq = */ step_src_v_sl,
          /* .step_head_size = */ step_src_v_head_size,
      };
      bestla_reordered_attn_fp32_update_v(&update_v_args);

      update_k_args.seq_off = seq_size_first;
      update_k_args.seq_size = seq_size_next;
      update_k_args.src = src_k.data() + seq_size_first * step_src_k_sl;
      bestla_reordered_attn_fp32_update_k(&update_k_args);

      update_v_args.seq_off = seq_size_first;
      update_v_args.seq_size = seq_size_next;
      update_v_args.src = src_v.data() + seq_size_first * step_src_v_sl;
      bestla_reordered_attn_fp32_update_v(&update_v_args);

      bestla_reordered_attn_fp32_fp32_fwd_args_t kern_args{
          /* .Q = */ reinterpret_cast<float*>(src_q.data()),
          /* .K = */ k_cache.data(),
          /* .V = */ v_cache.data(),
          /* .dst = */ reinterpret_cast<float*>(dst.data()),
          /* .Q_sc = */ init_scale_val<Q_T>,
          /* .K_sc = */ init_scale_val<K_T>,
          /* .V_sc = */ init_scale_val<V_T>,
          /* .dst_sc = */ init_scale_val<V_T>,
          /* .tmp = */ tmp.data(),
          /* .QK_scale = */ 1.f / sqrtf(static_cast<float>(head_size)),
          /* .attn_flags = */ flags,
          /* .batch_size = */ batch_size,
          /* .head_num = */ head_num,
          /* .heads_kv = */ heads_kv,
          /* .head_size = */ head_size,
          /* .sl_q = */ sl_q,
          /* .sl_kv = */ sl_kv,
          /* .Q_layout = */ ATTN_FWD_LAYOUT_PLAIN,
          /* .K_layout = */ kv_layout,
          /* .V_layout = */ kv_layout,
          /* .dst_layout = */ ATTN_FWD_LAYOUT_PLAIN,
          /* .step_q_bs = */ sl_q * head_num * head_size,
          /* .step_q_head_num = */ head_size,
          /* .step_q_sl = */ head_num * head_size,

          /* .stride_k_bs = */ static_cast<int>(kv_cache_info.k_bytes),
          /* .stride_k_head_num = */ kv_cache_info.stride_k_head_num,
          /* .stride_k_sl = */ kv_cache_info.stride_k_sl,
          /* .stride_k_head_size = */ kv_cache_info.stride_k_head_size,
          /* .stride_v_bs = */ static_cast<int>(kv_cache_info.v_bytes),
          /* .stride_v_head_num = */ kv_cache_info.stride_v_head_num,
          /* .stride_v_sl = */ kv_cache_info.stride_v_sl,
          /* .stride_v_head_size = */ kv_cache_info.stride_v_head_size,

          /* .step_dst_bs = */ sl_q * head_num * head_size,
          /* .step_dst_head_num = */ head_size,
          /* .step_dst_sl = */ head_num * head_size,
      };
      bestla_reordered_attn_fp32_forward(&kern_args);
    }

    // Check result
    return compare_data(dst.data(), ref.data(), dst.size(), 1e-2f);
  }
};
const test_mha_dese_t inst_;

}  // namespace

int main() {
  printf("NS_TESTS: mha_dense ");
  printf(ret_ok ? "OK\n" : "FAILED\n");
  return ret_ok ? 0 : -1;
}
