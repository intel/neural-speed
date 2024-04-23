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
#include <utils/utils.hpp>
#include <memory>
#include "fmha_forward.hpp"
#include "fmha_forward_policy.h"

#include "xetla.hpp"

using FMHA_T = fp16;
using policy_t = stage0<fmha_policy_32x128x128>;
// using policy_t = stage0<fmha_policy_1x512x128>;

constexpr uint32_t num_batches = 1;
constexpr uint32_t num_heads = 32;
constexpr uint32_t head_size = 128;
constexpr uint32_t num_queries = 33;
// constexpr uint32_t num_queries = 1;
constexpr uint32_t num_keys = 33;
constexpr float softmax_scale = 0.125;

// Q: [FxBxNxH] or [BxFxMxH]
// similar for K/V/O
constexpr bool kSeqLast = true;

template <typename accum_t>
int fma_result_validate(
    FMHA_T* q_device,
    FMHA_T* k_device,
    FMHA_T* v_device,
    FMHA_T* DST_device,
    sycl::queue& queue) {
  auto Q_ptr = alloc_host_and_copy<FMHA_T>(
      q_device, num_batches * num_heads * head_size * num_queries, queue);
  auto K_ptr = alloc_host_and_copy<FMHA_T>(
      k_device, num_batches * num_heads * head_size * num_keys, queue);
  auto V_ptr = alloc_host_and_copy<FMHA_T>(
      v_device, num_batches * num_heads * head_size * num_keys, queue);
  auto DST_ptr = alloc_host_and_copy<FMHA_T>(
      DST_device, num_batches * num_heads * head_size * num_queries, queue);

  std::vector<accum_t> gold_SP(
      num_batches * num_heads * num_queries * num_keys, 0);
  for (uint32_t gid = 0; gid < num_batches * num_heads; gid++) {
    uint32_t batch_id = gid / num_heads; // get batch idx
    uint32_t head_id = gid % num_heads; // get head idx

    const auto Q_cur = kSeqLast
        ? Q_ptr + batch_id * head_size * num_heads + head_size * head_id
        : Q_ptr + batch_id * num_queries * head_size * num_heads +
            head_size * head_id;
    const auto K_cur = kSeqLast
        ? K_ptr + batch_id * head_size * num_heads + head_size * head_id
        : K_ptr + batch_id * num_keys * head_size * num_heads +
            head_size * head_id;
    const auto gold_cur = gold_SP.data() + gid * num_queries * num_keys;

    auto Q_tmp = std::unique_ptr<FMHA_T[]>(new FMHA_T[num_queries * head_size]);
    for (uint32_t i = 0; i < num_queries; ++i)
      std::copy_n(
          Q_cur + i * head_size * num_heads * (kSeqLast ? num_batches : 1),
          head_size,
          Q_tmp.get() + i * head_size);
    auto K_tmp = std::unique_ptr<FMHA_T[]>(new FMHA_T[num_keys * head_size]);
    for (uint32_t i = 0; i < num_keys; ++i)
      for (uint32_t j = 0; j < head_size; ++j)
        K_tmp[j * num_keys + i] =
            K_cur[i * head_size * num_heads * (kSeqLast ? num_batches : 1) + j];

    get_gemm_gold<FMHA_T, FMHA_T, accum_t>(
        num_queries,
        num_keys,
        head_size,
        mem_layout::row_major,
        mem_layout::row_major,
        Q_tmp.get(),
        K_tmp.get(),
        gold_cur);

    for (uint32_t i = 0; i < num_queries; i++)
      for (uint32_t j = 0; j < num_keys; j++)
        gold_cur[i * num_keys + j] *=
            softmax_scale; // TODO(Yi): pass scale + mask
    for (uint32_t i = 0; i < num_queries; i++) {
      accum_t row_max = -INFINITY;
      accum_t exp_sum = 0;
      for (uint32_t j = 0; j < num_keys; j++)
        row_max = max(row_max, gold_cur[i * num_keys + j]);
      for (uint32_t j = 0; j < num_keys; j++) {
        gold_cur[i * num_keys + j] =
            std::exp(gold_cur[i * num_keys + j] - row_max);
        exp_sum += gold_cur[i * num_keys + j];
      }
      for (uint32_t j = 0; j < num_keys; j++)
        gold_cur[i * num_keys + j] /= exp_sum;
    }
  }

  std::vector<accum_t> gold_DST(
      num_batches * num_queries * num_heads * head_size, 0);
  // second gemm on host
  for (uint32_t gid = 0; gid < num_batches * num_heads; gid++) {
    uint32_t batch_id = gid / num_heads; // get batch idx
    uint32_t head_id = gid % num_heads; // get head idx

    // TODO
    const auto V_cur = kSeqLast
        ? V_ptr + batch_id * head_size * num_heads + head_size * head_id
        : V_ptr + batch_id * num_keys * head_size * num_heads +
            head_size * head_id;
    const auto P_cur = gold_SP.data() + gid * num_queries * num_keys;
    auto dst_cur =
        std::unique_ptr<accum_t[]>(new accum_t[num_queries * head_size]);
    std::fill_n(dst_cur.get(), num_queries * head_size, 0);
    auto V_tmp = std::unique_ptr<FMHA_T[]>(new FMHA_T[num_keys * head_size]);
    for (uint32_t i = 0; i < num_keys; ++i)
      std::copy_n(
          V_cur + i * head_size * num_heads * (kSeqLast ? num_batches : 1),
          head_size,
          V_tmp.get() + i * head_size);
    get_gemm_gold(
        num_queries,
        head_size,
        num_keys,
        mem_layout::row_major,
        mem_layout::row_major,
        P_cur,
        V_tmp.get(),
        dst_cur.get());

    // permute 0213
    const auto gold_cur = gold_DST.data() +
        batch_id * num_queries * num_heads * head_size + head_id * head_size;
    for (uint32_t i = 0; i < num_queries; ++i)
      std::copy_n(
          dst_cur.get() + i * head_size,
          head_size,
          gold_cur + i * num_heads * head_size * (kSeqLast ? num_batches : 1));
  }
  buff_cmp::buff_vals<FMHA_T> data( //
      DST_ptr,
      num_queries * num_heads * num_batches,
      head_size,
      head_size);
  buff_cmp::buff_vals<FMHA_T, accum_t> other(
      gold_DST.data(),
      num_queries * num_heads * num_batches,
      head_size,
      head_size);
  bool result = buff_cmp::xetla_buff_cmp(data, other, "fmha validation");

  free(Q_ptr);
  free(K_ptr);
  free(V_ptr);
  free(DST_ptr);

  std::cout << ((!result) ? "FAILED\n" : "PASSED\n");
  return result ? 0 : 1;
}

void fmha_run(uint32_t iter, uint32_t warmup = 10) {
  using fmha_forward_op_t = gpu::xetla::fmha::fmha_forward_t<
      policy_t,
      FMHA_T,
      gpu_arch::XeHpg,
      false,
      false,
      false,
      kSeqLast,
      false,
      false>;
  using accum_t = typename fmha_forward_op_t::accum_t;

  // Define SYCL queue, context and device
  sycl::property_list properties{sycl::property::queue::enable_profiling()};
  auto queue = sycl::queue(properties);
  auto context = queue.get_info<info::queue::context>();
  auto device = queue.get_info<info::queue::device>();

  print_device_details(device);

  auto Q = alloc_device_and_init<FMHA_T>(
      num_batches * num_heads * head_size * num_queries,
      [](FMHA_T* data, size_t idx) {
        data[idx] = static_cast<FMHA_T>(idx % 17);
      },
      queue,
      device,
      context);
  auto K = alloc_device_and_init<FMHA_T>(
      num_batches * num_heads * head_size * num_keys,
      [](FMHA_T* data, size_t idx) {
        data[idx] = static_cast<FMHA_T>(idx % 17);
      },
      queue,
      device,
      context);
  auto V = alloc_device_and_init<FMHA_T>(
      num_batches * num_heads * head_size * num_keys,
      [](FMHA_T* data, size_t idx) {
        data[idx] = static_cast<FMHA_T>(random_float());
      },
      queue,
      device,
      context);
  auto DST = alloc_device_and_init<FMHA_T>(
      num_batches * num_heads * head_size * num_queries,
      [](FMHA_T* data, size_t idx) { data[idx] = static_cast<FMHA_T>(9999); },
      queue,
      device,
      context);
  auto L = alloc_device_and_init<accum_t>( // log sum exp
      num_batches * num_heads * num_keys,
      [](accum_t* data, size_t idx) { data[idx] = static_cast<accum_t>(9999); },
      queue,
      device,
      context);

  sycl::nd_range<3> nd_range =
      fmha_forward_op_t::get_nd_range(num_batches * num_heads, num_queries);
  fmha_forward_op_t::check_slm_size(queue.get_info<info::queue::device>());
  std::cout << "slm_size:\t" << fmha_forward_op_t::get_slm_size() << std::endl;
  std::cout << "global_size:\t" << nd_range.get_global_range()[0] << ",\t"
            << nd_range.get_global_range()[1] << ",\t"
            << nd_range.get_global_range()[2] << std::endl;
  std::cout << "local_size:\t" << nd_range.get_local_range()[0] << ",\t"
            << nd_range.get_local_range()[1] << ",\t"
            << nd_range.get_local_range()[2] << std::endl;
  constexpr int64_t qk_ops = static_cast<int64_t>(2) * num_batches * num_heads *
      head_size * num_queries * num_keys;
  constexpr int64_t pv_ops = static_cast<int64_t>(2) * num_batches * num_heads *
      head_size * num_queries * num_keys;

  int64_t ops = qk_ops + pv_ops;
  profiling_helper prof("gemm_universal", ops, "gflops");
  for (uint32_t i = 0; i < iter + warmup; i++) {
    if (i >= warmup) {
      prof.cpu_start();
    }
    auto gpu_event = queue.submit([&](handler& cgh) {
      cgh.parallel_for(nd_range, [=](sycl::nd_item<3> item) KERNEL_MAIN {
        typename fmha_forward_op_t::arguments_t kern_args(
            Q,
            K,
            V,
            nullptr,
            nullptr,
            nullptr,
            DST,
            L,
            num_batches,
            num_heads,
            num_heads, // num_kv_heads
            head_size,
            num_queries,
            num_keys,
            -1,
            -1,
            -1,
            softmax_scale,
            0,
            0,
            0,
            (uint64_t)0,
            (uint64_t)0);
        fmha_forward_op_t{}(item, kern_args);
      });
    });
    gpu_event.wait();

    if (i >= warmup) {
      prof.cpu_end();
      prof.add_gpu_event(gpu_event);
    }
  }
  // performance
  prof.print_profiling_result(profiling_selector::GPU);

  ASSERT_EQ(0, fma_result_validate<accum_t>(Q, K, V, DST, queue));

  free(Q, context);
  free(K, context);
  free(V, context);
  free(DST, context);
}

int main() {
  fmha_run(5, 2);
}
