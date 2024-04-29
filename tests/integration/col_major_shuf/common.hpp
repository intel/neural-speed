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

#include <utils/common.hpp>
#include "xetla.hpp"

using namespace gpu::xetla;
using namespace cl::sycl;

template <typename data_type_in, typename data_type_out>
int col_major_shuf_result_validate(
    data_type_in* in_device,
    data_type_out* out_device,
    uint32_t* gidx_h,
    size_t m,
    size_t n,
    sycl::queue& queue) {
  auto in = alloc_host_and_copy<data_type_in>(in_device, m * n, queue);
  auto out = alloc_host_and_copy<data_type_out>(out_device, m * n, queue);

  int err_num = 0;

  for (uint32_t i = 0; i < m; i++) {
    for (uint32_t j = 0; j < n; j++) {
      if (out[i * n + j] != in[i * n + gidx_h[j] / sizeof(data_type_in)]) {
        std::cout << i * n + j << " " << out[i * n + j] << " "
                  << in[i * n + gidx_h[j] / sizeof(data_type_in)] << std::endl;
        err_num++;
      }
    }
  }

  free(in);
  free(out);
  if (err_num == 0) {
    std::cout << "Test Passed!!!" << std::endl;
  }
  return err_num;
}

class Test1 {
 public:
  static constexpr size_t mat_m = 1024;
  static constexpr size_t mat_n = 11008;

  static constexpr uint32_t wg_m = 16;
  static constexpr uint32_t wg_n = 64;
  static constexpr uint32_t sg_m = 8;
  static constexpr uint32_t sg_n = 16;

  static constexpr uint32_t load_block_size = 16;

  using data_type_in = fp16;
  using data_type_out = fp16;
};

class Test2 {
 public:
  static constexpr size_t mat_m = 1;
  static constexpr size_t mat_n = 4096;

  static constexpr size_t wg_n = 128;
  static constexpr size_t wg_m = 1;
  static constexpr size_t sg_n = 16;
  static constexpr size_t sg_m = 1;

  static constexpr uint32_t load_block_size = 16;

  using data_type_in = fp16;
  using data_type_out = fp16;
};
