//  Copyright (c) 2022 Intel Corporation
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
#include "matmul_vnni_noperm_p2031_p1302.hpp"

#include <functional>
#include <utility>

#include "benchmark_utils.hpp"
#include "common_utils.hpp"

namespace bench {

bool matmul_vnni_noperm_p2031_p1302_bench::check_result() {
  const auto& p = args.first;
  const auto& q = args.second;
  get_true_data();
  auto buf1 = p.rt_data[io::DST0];
  auto size1 = p.op_desc.tensor_descs()[io::DST0].size();
  auto buf2 = q.rt_data[io::DST0];
  auto size2 = q.op_desc.tensor_descs()[io::DST0].size();
  const auto& dst_type = p.op_desc.tensor_descs()[io::DST0].dtype();
  if (dst_type == jd::data_type::u8) {
    return compare_data<uint8_t>(buf1, size1, buf2, size2, 8e-3);  // tolerance of 2
  } else {
    LOG(WARNING) << "unsupported dst type";
    return false;
  }
  return false;
}
std::pair<const void*, const void*> make_data_obj_matmul_vnni_noperm_p2031_p1302(  //
    const std::vector<int64_t>& a_shape, const jd::data_type& a_dt, bool is_clear = false,
    const std::vector<float>& ranges = {-10, 10}) {
  if (a_shape.empty()) {
    return std::pair<const void*, const void*>{nullptr, nullptr};
  }
  int elem_num = std::accumulate(a_shape.begin(), a_shape.end(), 1, std::multiplies<dim_t>());
  int bytes_size = elem_num * jd::type_size[a_dt];
  void* data_ptr = nullptr;
  if (is_clear) {
    data_ptr = aligned_allocator_t<char>::allocate(bytes_size, true);
  } else {
    if (a_dt == jd::data_type::fp32) {
      data_ptr = aligned_allocator_t<float>::allocate(elem_num);
      init_vector(static_cast<float*>(data_ptr), elem_num, ranges[0], ranges[1]);
    } else if (a_dt == jd::data_type::s32) {
      data_ptr = aligned_allocator_t<int32_t>::allocate(elem_num);
      init_vector(static_cast<int32_t*>(data_ptr), elem_num, ranges[0], ranges[1]);
    } else if (a_dt == jd::data_type::u8) {
      data_ptr = aligned_allocator_t<uint8_t>::allocate(elem_num);
      init_vector(static_cast<uint8_t*>(data_ptr), elem_num, ranges[0], ranges[1]);
    } else if (a_dt == jd::data_type::s8) {
      data_ptr = aligned_allocator_t<int8_t>::allocate(elem_num);
      init_vector(static_cast<int8_t*>(data_ptr), elem_num, ranges[0], ranges[1]);
    }
  }
  void* data_ptr_copy = aligned_allocator_t<char>::allocate(bytes_size);
  memcpy(data_ptr_copy, data_ptr, bytes_size);
  return std::pair<const void*, const void*>{data_ptr, data_ptr_copy};
}
void matmul_vnni_noperm_p2031_p1302_bench::gen_case() {
  /**
   * Step 1: Construct operator config
   *
   * Dimension details:
   *   src0: bs0 bs1 m   k ===========> bs0 bs1 m k
   *   src1: bs1 n   bs0 k ==perm2031=> bs0 bs1 k n
   *   dst:  bs1 n   bs0 m <=perm1302== bs0 bs1 m n
   */
  jd::tensor_desc src0_desc = {{bs0, bs1, M, K}, jd::data_type::u8, jd::format_type::ab};
  jd::tensor_desc src1_desc = {{bs1, N, bs0, K}, jd::data_type::s8, jd::format_type::ab};
  jd::tensor_desc dst_desc = {{bs1, N, bs0, M}, jd::data_type::u8, jd::format_type::ab};
  jd::tensor_desc src2_desc = {{}, jd::data_type::fp32, jd::format_type::ab};  // binary postop not supported
  jd::tensor_desc scale_desc = {{1}, jd::data_type::fp32, jd::format_type::a};
  jd::tensor_desc zp_desc = {{1}, jd::data_type::fp32, jd::format_type::a};
  ts_descs = {src0_desc, src1_desc, dst_desc, src2_desc, scale_desc, zp_desc};
  // Step 2: Construct runtime data
  std::vector<const void*> rt_data1;
  std::vector<const void*> rt_data2;
  int tensor_num = ts_descs.size();
  for (int index = 0; index < tensor_num; ++index) {
    auto& tsd = ts_descs[index];
    bool is_clear = (index == io::DST0);
    std::vector<float> ranges;
    if (index < io::SRC2) {
      ranges = {-10, 10};
    } else if (index == io::SCALE0) {
      ranges = {.003f, .003f};
    } else if (index == io::ZP0) {
      ranges = {113, 113};
    }

    auto data_pair = make_data_obj_matmul_vnni_noperm_p2031_p1302(tsd.shape(), tsd.dtype(), is_clear, ranges);
    rt_data1.emplace_back(data_pair.first);
    rt_data2.emplace_back(data_pair.second);
  }

  jd::operator_desc op_desc(jd::kernel_kind::transpose_matmul, jd::kernel_prop::forward_inference, jd::engine_kind::cpu,
                            ts_descs, op_attrs);
  // Step 3: op_args_t testcase pair
  op_args_t op_args = {op_desc, rt_data1};
  op_args_t op_args_copy = {op_desc, rt_data2};
  args = {op_args, op_args_copy};
}
bench_res_t matmul_vnni_noperm_p2031_p1302_bench::set_config(int argc, char** argv) {
  LOG(INFO) << "matmul_vnni_noperm_p2031_p1302";
  if (argc < MATMUL_VNNI_NOPERM_P2031_P1302_ARG_NUM) {
    LOG(ERROR) << "No enough arguments passed";
    return {bench_status::wrong_input};
  }
  M = str_to_num<int64_t>(argv[0]);    // M
  K = str_to_num<int64_t>(argv[1]);    // K
  N = str_to_num<int64_t>(argv[2]);    // N
  bs0 = str_to_num<int64_t>(argv[3]);  // bs0
  bs1 = str_to_num<int64_t>(argv[4]);  // bs1
  return {bench_status::success};
}
}  // namespace bench
