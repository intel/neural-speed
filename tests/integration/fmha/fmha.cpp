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
#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include "fmha_forward.hpp"
#include "fmha_forward_policy.h"

#include "xetla.hpp"

struct test_params_t {
  // Q: [FxBxNxH] or [BxFxMxH] ; similar for K/V/O
  bool kSeqLast;
  uint32_t bs;
  uint32_t hn;
  uint32_t hs;
  uint32_t qlen;
  uint32_t klen;

  static std::vector<test_params_t> cases() {
    return {
        {false, 1, 32, 128, 1, 33},
        {true, 1, 32, 128, 1, 33},
        {false, 1, 32, 128, 33, 33},
        {true, 1, 32, 128, 33, 33},
        {false, 1, 32, 128, 1023, 1023},
        {true, 1, 32, 128, 1023, 1023},
    };
  }

  std::string to_string() const {
    std::vector<std::string> params;
    params.push_back("bs" + std::to_string(bs));
    params.push_back("hn" + std::to_string(hn));
    params.push_back("hs" + std::to_string(hs));
    params.push_back("qlen" + std::to_string(qlen));
    params.push_back("klen" + std::to_string(klen));
    params.push_back(std::string("kSeqLast") + (kSeqLast ? "ON" : "OFF"));
    return std::accumulate(
        std::next(params.begin()),
        params.end(),
        params[0],
        [](std::string a, std::string b) { return a + '_' + b; });
  }
};

using FMHA_T = fp16;

template <bool kSeqLast, typename accum_t>
int fma_result_validate(
    const test_params_t& p,
    FMHA_T* q_device,
    FMHA_T* k_device,
    FMHA_T* v_device,
    FMHA_T* DST_device,
    sycl::queue& queue) {
  const auto bs = p.bs;
  const auto hn = p.hn;
  const auto hs = p.hs;
  const auto qlen = p.qlen;
  const auto klen = p.klen;
  const float softmax_scale = 1.f / std::sqrt(p.hs);
  auto Q_ptr =
      alloc_host_and_copy<FMHA_T>(q_device, bs * hn * hs * qlen, queue);
  auto K_ptr =
      alloc_host_and_copy<FMHA_T>(k_device, bs * hn * hs * klen, queue);
  auto V_ptr =
      alloc_host_and_copy<FMHA_T>(v_device, bs * hn * hs * klen, queue);
  auto DST_ptr =
      alloc_host_and_copy<FMHA_T>(DST_device, bs * hn * hs * qlen, queue);

  std::vector<accum_t> gold_SP(bs * hn * qlen * klen, 0);
  for (uint32_t gid = 0; gid < bs * hn; gid++) {
    uint32_t batch_id = gid / hn; // get batch idx
    uint32_t head_id = gid % hn; // get head idx

    const auto Q_cur = kSeqLast
        ? Q_ptr + batch_id * hs * hn + hs * head_id
        : Q_ptr + batch_id * qlen * hs * hn + hs * head_id;
    const auto K_cur = kSeqLast
        ? K_ptr + batch_id * hs * hn + hs * head_id
        : K_ptr + batch_id * klen * hs * hn + hs * head_id;
    const auto gold_cur = gold_SP.data() + gid * qlen * klen;

    auto Q_tmp = std::unique_ptr<FMHA_T[]>(new FMHA_T[qlen * hs]);
    for (uint32_t i = 0; i < qlen; ++i)
      std::copy_n(
          Q_cur + i * hs * hn * (kSeqLast ? bs : 1), hs, Q_tmp.get() + i * hs);
    auto K_tmp = std::unique_ptr<FMHA_T[]>(new FMHA_T[klen * hs]);
    for (uint32_t i = 0; i < klen; ++i)
      for (uint32_t j = 0; j < hs; ++j)
        K_tmp[j * klen + i] = K_cur[i * hs * hn * (kSeqLast ? bs : 1) + j];

    get_gemm_gold<FMHA_T, FMHA_T, accum_t>(
        qlen,
        klen,
        hs,
        mem_layout::row_major,
        mem_layout::row_major,
        Q_tmp.get(),
        K_tmp.get(),
        gold_cur);

    for (uint32_t i = 0; i < qlen; i++)
      for (uint32_t j = 0; j < klen; j++)
        gold_cur[i * klen + j] *= softmax_scale; // TODO(Yi): pass scale + mask
    for (uint32_t i = 0; i < qlen; i++) {
      accum_t row_max = -INFINITY;
      accum_t exp_sum = 0;
      for (uint32_t j = 0; j < klen; j++)
        row_max = max(row_max, gold_cur[i * klen + j]);
      for (uint32_t j = 0; j < klen; j++) {
        gold_cur[i * klen + j] = std::exp(gold_cur[i * klen + j] - row_max);
        exp_sum += gold_cur[i * klen + j];
      }
      for (uint32_t j = 0; j < klen; j++)
        gold_cur[i * klen + j] /= exp_sum;
    }
  }

  std::vector<accum_t> gold_DST(bs * qlen * hn * hs, 0);
  // second gemm on host
  for (uint32_t gid = 0; gid < bs * hn; gid++) {
    uint32_t batch_id = gid / hn; // get batch idx
    uint32_t head_id = gid % hn; // get head idx

    // TODO
    const auto V_cur = kSeqLast
        ? V_ptr + batch_id * hs * hn + hs * head_id
        : V_ptr + batch_id * klen * hs * hn + hs * head_id;
    const auto P_cur = gold_SP.data() + gid * qlen * klen;
    auto dst_cur = std::unique_ptr<accum_t[]>(new accum_t[qlen * hs]);
    std::fill_n(dst_cur.get(), qlen * hs, 0);
    auto V_tmp = std::unique_ptr<FMHA_T[]>(new FMHA_T[klen * hs]);
    for (uint32_t i = 0; i < klen; ++i)
      std::copy_n(
          V_cur + i * hs * hn * (kSeqLast ? bs : 1), hs, V_tmp.get() + i * hs);
    get_gemm_gold(
        qlen,
        hs,
        klen,
        mem_layout::row_major,
        mem_layout::row_major,
        P_cur,
        V_tmp.get(),
        dst_cur.get());

    // permute 0213
    const auto gold_cur =
        gold_DST.data() + batch_id * qlen * hn * hs + head_id * hs;
    for (uint32_t i = 0; i < qlen; ++i)
      std::copy_n(
          dst_cur.get() + i * hs,
          hs,
          gold_cur + i * hn * hs * (kSeqLast ? bs : 1));
  }
  buff_cmp::buff_vals<FMHA_T> data( //
      DST_ptr,
      qlen * hn * bs,
      hs,
      hs);
  buff_cmp::buff_vals<FMHA_T, accum_t> other(
      gold_DST.data(), qlen * hn * bs, hs, hs);
  bool result = buff_cmp::xetla_buff_cmp(data, other, "fmha validation");

  free(Q_ptr);
  free(K_ptr);
  free(V_ptr);
  free(DST_ptr);

  std::cout << ((!result) ? "FAILED\n" : "PASSED\n");
  return result ? 0 : 1;
}

template <bool kSeqLast, typename policy_t>
void fmha_run_(const test_params_t& p, uint32_t iter, uint32_t warmup) {
  const auto bs = p.bs;
  const auto hn = p.hn;
  const auto hs = p.hs;
  const auto qlen = p.qlen;
  const auto klen = p.klen;
  const float softmax_scale = 1.f / std::sqrt(p.hs);
  using fmha_forward_op_t = gpu::xetla::fmha::fmha_forward_t<
      policy_t,
      FMHA_T,
      gpu_arch::XeLpg,
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
      bs * hn * hs * qlen,
      [](FMHA_T* data, size_t idx) {
        data[idx] = static_cast<FMHA_T>(idx % 11);
      },
      queue,
      device,
      context);
  auto K = alloc_device_and_init<FMHA_T>(
      bs * hn * hs * klen,
      [](FMHA_T* data, size_t idx) {
        data[idx] = static_cast<FMHA_T>(idx % 11);
      },
      queue,
      device,
      context);
  auto V = alloc_device_and_init<FMHA_T>(
      bs * hn * hs * klen,
      [](FMHA_T* data, size_t idx) {
        data[idx] = static_cast<FMHA_T>(random_float());
      },
      queue,
      device,
      context);
  auto DST = alloc_device_and_init<FMHA_T>(
      bs * hn * hs * qlen,
      [](FMHA_T* data, size_t idx) { data[idx] = static_cast<FMHA_T>(9999); },
      queue,
      device,
      context);
  auto L = alloc_device_and_init<accum_t>( // log sum exp
      bs * hn * klen,
      [](accum_t* data, size_t idx) { data[idx] = static_cast<accum_t>(9999); },
      queue,
      device,
      context);

  sycl::nd_range<3> nd_range = fmha_forward_op_t::get_nd_range(bs * hn, qlen);
  fmha_forward_op_t::check_slm_size(queue.get_info<info::queue::device>());
  std::cout << "slm_size:\t" << fmha_forward_op_t::get_slm_size() << std::endl;
  std::cout << "global_size:\t" << nd_range.get_global_range()[0] << ",\t"
            << nd_range.get_global_range()[1] << ",\t"
            << nd_range.get_global_range()[2] << std::endl;
  std::cout << "local_size:\t" << nd_range.get_local_range()[0] << ",\t"
            << nd_range.get_local_range()[1] << ",\t"
            << nd_range.get_local_range()[2] << std::endl;
  const int64_t qk_ops = static_cast<int64_t>(2) * bs * hn * hs * qlen * klen;
  const int64_t pv_ops = static_cast<int64_t>(2) * bs * hn * hs * qlen * klen;

  const int64_t ops = qk_ops + pv_ops;
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
            bs,
            hn,
            hn, // num_kv_heads
            hs,
            qlen,
            klen,
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

  ASSERT_EQ(
      0, (fma_result_validate<kSeqLast, accum_t>(p, Q, K, V, DST, queue)));

  free(Q, context);
  free(K, context);
  free(V, context);
  free(DST, context);
}
void fmha_run(const test_params_t& p, uint32_t iter, uint32_t warmup = 10) {
  ASSERT_TRUE(p.hs == 128);
  if (p.qlen == 1) {
    using policy_t = stage0<fmha_policy_1x512x128>;
    return p.kSeqLast ? fmha_run_<true, policy_t>(p, iter, warmup)
                      : fmha_run_<false, policy_t>(p, iter, warmup);
  } else {
    using policy_t = stage0<fmha_policy_32x128x128>;
    return p.kSeqLast ? fmha_run_<true, policy_t>(p, iter, warmup)
                      : fmha_run_<false, policy_t>(p, iter, warmup);
  }
}

using ::testing::TestParamInfo;
using ::testing::TestWithParam;
using ::testing::ValuesIn;

class FMHATest : public TestWithParam<test_params_t> {
 protected:
  FMHATest() {}
  ~FMHATest() {}
  void SetUp() override {}
  void TearDown() override {}
};
TEST_P(FMHATest, ) {
  test_params_t p = TestWithParam<test_params_t>::GetParam();
  fmha_run(p, 1, 0);
}
INSTANTIATE_TEST_SUITE_P(
    XeTLA,
    FMHATest,
    ValuesIn(test_params_t::cases()),
    [](TestParamInfo<test_params_t> tpi) { return tpi.param.to_string(); });
