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

#include <gtest/gtest.h>
#include <utils/utils.hpp>
#include "common.hpp"

using namespace gpu::xetla;
using namespace cl::sycl;

template <class Test>
static void col_major_shuf_run() {
  size_t matrix_m = Test::mat_m;
  size_t matrix_n = Test::mat_n;
  size_t size_gidx = matrix_n;

  constexpr uint32_t wg_tile_m = Test::wg_m;
  constexpr uint32_t wg_tile_n = Test::wg_n;
  constexpr uint32_t sg_tile_m = Test::sg_m;
  constexpr uint32_t sg_tile_n = Test::sg_n;

  static_assert(
      // Test::mat_m % Test::sg_m == 0 && Test::mat_n % Test::sg_n == 0,
      Test::mat_n % Test::sg_n == 0,
      "Matrix size should be multiple of subgroup size");

  using data_type_in = typename Test::data_type_in;
  using data_type_out = typename Test::data_type_out;

  // Turn on the enable_profiling property to facilitate subsequent profiling
  sycl::property_list properties{sycl::property::queue::enable_profiling()};
  auto queue = sycl::queue(properties);
  auto context = queue.get_info<info::queue::context>();
  auto device = queue.get_info<info::queue::device>();

  std::cout << "Running on " << device.get_info<info::device::name>() << "\n";

  int size = matrix_m * matrix_n;

  auto buffer_in = alloc_device_and_init<data_type_in>(
      size,
      [](data_type_in* data, size_t idx) {
        data[idx] = static_cast<data_type_in>(idx);
      },
      queue,
      device,
      context);
  auto buffer_out = alloc_device_and_init<data_type_out>(
      size,
      [](data_type_out* data, size_t idx) {
        data[idx] = static_cast<data_type_out>(0);
      },
      queue,
      device,
      context);

  uint32_t* gidx_h = static_cast<uint32_t*>(
      malloc_host(size_gidx * sizeof(uint32_t), context));
  auto* gidx_d = static_cast<uint32_t*>(aligned_alloc_device(
      DEVICE_MEM_ALIGNMENT, size_gidx * sizeof(uint32_t), device, context));

  for (uint32_t i = 0; i < matrix_n; i++) {
    gidx_h[i] = i * sizeof(data_type_in);
  }
  for (int i = matrix_n - 1; i >= 0; i--) {
    int j = rand() % (i + 1);
    std::swap(gidx_h[i], gidx_h[j]);
  }
  queue.memcpy((void*)gidx_d, (void*)gidx_h, size_gidx * sizeof(uint32_t))
      .wait();

  cl::sycl::range<3> group_range{
      1,
      (matrix_m + wg_tile_m - 1) / wg_tile_m,
      (matrix_n + wg_tile_n - 1) / wg_tile_n};
  cl::sycl::range<3> local_range{
      1,
      (wg_tile_m + sg_tile_m - 1) / sg_tile_m,
      (wg_tile_n + sg_tile_n - 1) / sg_tile_n};
  cl::sycl::nd_range<3> nd_range(group_range * local_range, local_range);
  std::vector<kernel_id> kernelId = {get_kernel_id<Test>()};
  auto inputBundle = get_kernel_bundle<bundle_state::input>(context, kernelId);
  static const std::string env_set_str =
      "SYCL_PROGRAM_COMPILE_OPTIONS= -vc-codegen -Xfinalizer ' -printregusage -enableBCR -DPASTokenReduction '";
  putenv(const_cast<char*>(env_set_str.c_str()));
  kernel_bundle<bundle_state::executable> exeBundle = build(inputBundle);
  static const std::string env_unset_str = "SYCL_PROGRAM_COMPILE_OPTIONS=";
  putenv(const_cast<char*>(env_unset_str.c_str()));

  try {
    auto e_esimd = queue.submit([&](handler& cgh) {
      cgh.use_kernel_bundle(exeBundle);
      cgh.parallel_for<Test>(nd_range, [=](nd_item<3> item) KERNEL_MAIN {
        // cgh.parallel_for<Test>(nd_range, [=](nd_item<3> item)
        // SYCL_ESIMD_KERNEL {
        using col_major_shuf_attr = gpu::xetla::kernel::col_major_shuf_attr_t<
            wg_tile_n,
            wg_tile_m,
            sg_tile_n,
            sg_tile_m,
            Test::load_block_size>;
        using col_major_shuf = gpu::xetla::kernel::col_major_shuf_t<
            data_type_in,
            data_type_out,
            uint32_t,
            mem_layout::row_major,
            col_major_shuf_attr,
            gpu_arch::XeHpg>;

        typename col_major_shuf::arguments_t args;
        args.mat_in_ptr = buffer_in;
        args.mat_out_ptr = buffer_out;
        args.gidx_ptr = gidx_d;
        args.matrix_x = matrix_n;
        args.matrix_y = matrix_m;
        col_major_shuf::call(item, args);
      });
    });
    e_esimd.wait();
  } catch (cl::sycl::exception const& e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    FAIL();
  }

  // validation
  ASSERT_EQ(
      0,
      (col_major_shuf_result_validate<data_type_in, data_type_out>(
          buffer_in, buffer_out, gidx_h, Test::mat_m, Test::mat_n, queue)));

  free(buffer_in, context);
  free(buffer_out, context);
  free(gidx_d, context);
  free(gidx_h, context);
}

TEST(TestBase, esimd) {
  col_major_shuf_run<TestBase>();
}
