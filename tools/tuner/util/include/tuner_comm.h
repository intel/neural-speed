
/*******************************************************************************
 * Copyright 2022 Intel Corporation
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
#ifndef TUNER_COMMON_H
#define TUNER_COMMON_H
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace tuner_ns {

using namespace std;

typedef enum {
    MK_GEMM = 0,
    MK_BATCH_GEMM,
    MK_BRGEMM,
    MK_GELU,
    MK_RELU,
    MK_TILE_STORE,
    MK_MHA,
    MK_SOFT_MAX,
    MK_GRU,
    MK_SELF_DEFINE_KERNEL,
    MK_NOT_SUPPORT
} micro_kernel_type;

typedef enum {
    TL_TUNE_OUTPUT = 0,
    TL_TUNE_RESULT,
    TL_DBG_INFO,
    TL_NOT_SUPPORT
} tuner_log_type;

typedef enum {
    CG_STATIC_SIZET = 0,
    CG_STATIC_UINT32,
    CG_STATIC_USING,
    CG_STATIC_MEM_LAYOUT,
    CG_STATIC_STR,
    CG_NO_NEED_CODE_GEN,
} code_gen_info_type;
extern std::map<code_gen_info_type, std::string> code_gen_info_map;
extern map<string, string> tuner_support_dtypes;
extern std::map<micro_kernel_type, string> micro_kernel_type_map;
typedef struct runner_record {
    std::string kernel_path;
    bool accuracy;
    double kernel_time;
    double kernel_flops;
} runner_record;

using smap_t = std::unordered_map<std::string, std::string>;

typedef struct {
    string data_type;
    string mem_layout;
} shape_info;

typedef struct {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t batch_count;
    shape_info a_shape;
    shape_info b_shape;
    shape_info c_shape;
} gemm_cmd_info;

typedef struct {
    std::string micro_kernel_name;
    std::string attribute_file_name;
    std::string micro_kernel_path; // full path
    micro_kernel_type mk_type;
} micro_kernel_info;

typedef struct {
    int32_t start_value;
    int32_t end_value;
    int32_t stride;
} range_element_attr;

typedef struct {
    std::string attr_name;
    range_element_attr attr;
} tune_attr_cfg;

typedef std::vector<tune_attr_cfg> tune_attr_vector;

typedef struct {
    micro_kernel_info micro_kernel_desc;
    tune_attr_vector tune_attr;
} one_micro_kernel_tune_info;

typedef struct {
    map<string, string> kernel_attr; //("m","1024")
    map<string, code_gen_info_type> code_gen_info; //("mat_m",CG_STATIC_SIZET)
} kernel_cfg_input_info;

typedef struct {
    vector<micro_kernel_type> mk_list;
    kernel_cfg_input_info usr_cfg_kernel_attr;
} cmdline_input_info;

typedef struct {
    micro_kernel_type mk_type;
    string mk_path;
} mk_info;

typedef struct kernel_perf_test_info {
    string kernel_path;
    string test_bin_full_path;
    string run_env;
} kernel_perf_test_info;

typedef struct run_kernel_result {
    bool run_succ;
    runner_record perf_data;
} run_kernel_result;

// size_t wg_m;
// size_t wg_n;
// size_t sg_m;
// size_t sg_n;
// size_t sg_k;
// size_t local_kslicing;
// size_t global_kslicing;
using gemm_tune_cfg = std::map<std::string, std::string>;

typedef struct build_kernel_perf_test_result {
    bool build_succ;
    kernel_perf_test_info build_res;
    smap_t tuning_params;
    run_kernel_result run_res;
} build_kernel_perf_test_result;

typedef struct optimal_tune_result {
    runner_record perf_data;
    std::string tune_cfg;
    bool tune_succ;
} optimal_tune_result;
using mk_name_to_mk_attr_map = map<string, one_micro_kernel_tune_info>;

} // namespace tuner_ns

#endif
