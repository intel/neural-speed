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
#include "tune_in_existed_result.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>

using namespace std;
namespace tuner_ns {
std::unordered_map<std::string, gemm_tune_cfg>
        tuned_result::tuned_result_cache_matrix_B_row_major = {
                //                       wg_m,wg_n,sg_m,sg_n,sg_k,local_kslicing,global_kslicing
                {"m = 1, n = 5120, k = 5120",
                        {{"wg_m", "8"}, {"wg_n", "128"}, {"sg_m", "8"},
                                {"sg_n", "16"}, {"sg_k", "16"},
                                {"local_kslicing", "4"},
                                {"global_kslicing", "1"}}},
                {"m = 1, n = 13824, k = 5120",
                        {{"wg_m", "8"}, {"wg_n", "512"}, {"sg_m", "8"},
                                {"sg_n", "16"}, {"sg_k", "16"},
                                {"local_kslicing", "1"},
                                {"global_kslicing", "1"}}},
                {"m = 1, n = 5120, k = 13824",
                        {{"wg_m", "8"}, {"wg_n", "128"}, {"sg_m", "8"},
                                {"sg_n", "16"}, {"sg_k", "16"},
                                {"local_kslicing", "4"},
                                {"global_kslicing", "1"}}},
                {"m = 1, n = 4096, k = 4096",
                        {{"wg_m", "32"}, {"wg_n", "64"}, {"sg_m", "8"},
                                {"sg_n", "16"}, {"sg_k", "16"},
                                {"local_kslicing", "2"},
                                {"global_kslicing", "1"}}},
                {"m = 1, n = 11008, k = 4096",
                        {{"wg_m", "16"}, {"wg_n", "256"}, {"sg_m", "8"},
                                {"sg_n", "16"}, {"sg_k", "16"},
                                {"local_kslicing", "1"},
                                {"global_kslicing", "1"}}},
                {"m = 1, n = 4096, k = 11008",
                        {{"wg_m", "32"}, {"wg_n", "64"}, {"sg_m", "8"},
                                {"sg_n", "16"}, {"sg_k", "16"},
                                {"local_kslicing", "2"},
                                {"global_kslicing", "1"}}},
                {"m = 1, n = 32000, k = 4096",
                        {{"wg_m", "16"}, {"wg_n", "1024"}, {"sg_m", "16"},
                                {"sg_n", "32"}, {"sg_k", "16"},
                                {"local_kslicing", "1"},
                                {"global_kslicing", "1"}}},
                {"m = 1, n = 16384, k = 4096",
                        {{"wg_m", "8"}, {"wg_n", "512"}, {"sg_m", "8"},
                                {"sg_n", "16"}, {"sg_k", "16"},
                                {"local_kslicing", "1"},
                                {"global_kslicing", "1"}}},
                {"m = 1, n = 4096, k = 16384",
                        {{"wg_m", "32"}, {"wg_n", "64"}, {"sg_m", "8"},
                                {"sg_n", "16"}, {"sg_k", "16"},
                                {"local_kslicing", "2"},
                                {"global_kslicing", "1"}}},
                {"m = 100, n = 4096, k = 4096",
                        {{"wg_m", "128"}, {"wg_n", "64"}, {"sg_m", "32"},
                                {"sg_n", "16"}, {"sg_k", "32"},
                                {"local_kslicing", "2"},
                                {"global_kslicing", "1"}}},
                {"m = 152, n = 4096, k = 4096",
                        {{"wg_m", "128"}, {"wg_n", "128"}, {"sg_m", "32"},
                                {"sg_n", "32"}, {"sg_k", "32"},
                                {"local_kslicing", "2"},
                                {"global_kslicing", "1"}}},
                {"m = 216, n = 4096, k = 4096",
                        {{"wg_m", "128"}, {"wg_n", "128"}, {"sg_m", "32"},
                                {"sg_n", "32"}, {"sg_k", "32"},
                                {"local_kslicing", "2"},
                                {"global_kslicing", "1"}}},
                {"m = 256, n = 4096, k = 4096",
                        {{"wg_m", "128"}, {"wg_n", "128"}, {"sg_m", "32"},
                                {"sg_n", "32"}, {"sg_k", "32"},
                                {"local_kslicing", "2"},
                                {"global_kslicing", "1"}}}

};
std::unordered_map<std::string, gemm_tune_cfg>
        tuned_result::tuned_result_cache_matrix_B_col_major = {
                //                       wg_m,wg_n,sg_m,sg_n,sg_k,local_kslicing,global_kslicing
                {"m = 1, n = 5120, k = 5120",
                        {{"wg_m", "64"}, {"wg_n", "16"}, {"sg_m", "16"},
                                {"sg_n", "16"}, {"sg_k", "64"},
                                {"local_kslicing", "8"},
                                {"global_kslicing", "1"}}},
                {"m = 1, n = 13824, k = 5120",
                        {{"wg_m", "32"}, {"wg_n", "16"}, {"sg_m", "16"},
                                {"sg_n", "16"}, {"sg_k", "64"},
                                {"local_kslicing", "16"},
                                {"global_kslicing", "1"}}},
                {"m = 1, n = 5120, k = 13824",
                        {{"wg_m", "32"}, {"wg_n", "16"}, {"sg_m", "8"},
                                {"sg_n", "16"}, {"sg_k", "64"},
                                {"local_kslicing", "1"},
                                {"global_kslicing", "1"}}},
                {"m = 1, n = 32000, k = 5120",
                        {{"wg_m", "32"}, {"wg_n", "16"}, {"sg_m", "8"},
                                {"sg_n", "16"}, {"sg_k", "64"},
                                {"local_kslicing", "8"},
                                {"global_kslicing", "1"}}},
                {"m = 1, n = 4096, k = 4096",
                        {{"wg_m", "32"}, {"wg_n", "16"}, {"sg_m", "8"},
                                {"sg_n", "16"}, {"sg_k", "64"},
                                {"local_kslicing", "8"},
                                {"global_kslicing", "1"}}},
                {"m = 1, n = 11008, k = 4096",
                        {{"wg_m", "32"}, {"wg_n", "16"}, {"sg_m", "8"},
                                {"sg_n", "16"}, {"sg_k", "64"},
                                {"local_kslicing", "8"},
                                {"global_kslicing", "1"}}},
                {"m = 1, n = 4096, k = 11008",
                        {{"wg_m", "32"}, {"wg_n", "16"}, {"sg_m", "8"},
                                {"sg_n", "16"}, {"sg_k", "64"},
                                {"local_kslicing", "8"},
                                {"global_kslicing", "1"}}},
                {"m = 1, n = 32000, k = 4096",
                        {{"wg_m", "32"}, {"wg_n", "16"}, {"sg_m", "16"},
                                {"sg_n", "16"}, {"sg_k", "64"},
                                {"local_kslicing", "16"},
                                {"global_kslicing", "1"}}},
                {"m = 1, n = 16384, k = 4096",
                        {{"wg_m", "128"}, {"wg_n", "16"}, {"sg_m", "64"},
                                {"sg_n", "16"}, {"sg_k", "32"},
                                {"local_kslicing", "16"},
                                {"global_kslicing", "1"}}},
                {"m = 1, n = 4096, k = 16384",
                        {{"wg_m", "64"}, {"wg_n", "16"}, {"sg_m", "8"},
                                {"sg_n", "16"}, {"sg_k", "64"},
                                {"local_kslicing", "4"},
                                {"global_kslicing", "1"}}},
                {"m = 1, n = 50400, k = 4096",
                        {{"wg_m", "128"}, {"wg_n", "16"}, {"sg_m", "64"},
                                {"sg_n", "16"}, {"sg_k", "32"},
                                {"local_kslicing", "16"},
                                {"global_kslicing", "1"}}},
};

std::string tuned_result::get_cfg_key(size_t m, size_t n, size_t k) {
    std::ostringstream traits;
    traits << "m = " << m << ", n = " << n << ", k = " << k;
    std::string traits_str = traits.str();
    return traits_str;
}
std::unordered_map<std::string, gemm_tune_cfg> &tuned_result::get_cache_data(
        bool is_row_major) {
    return (is_row_major ? tuned_result_cache_matrix_B_row_major
                         : tuned_result_cache_matrix_B_col_major);
}

bool tuned_result::get_tuned_result(size_t m, size_t n, size_t k,
        bool is_row_major, gemm_tune_cfg &cfg_res) {
    auto key = get_cfg_key(m, n, k);
    auto cache_data = get_cache_data(is_row_major);
    auto res = cache_data.find(key);
    if (res != cache_data.end()) {
        cfg_res = res->second;

        return true;
    }

    return false;
};

optimal_tune_result tune_in_cache_result::run_tuning(
        cmdline_input_info &cmd_info, vector<vector<string>> &all_op_cart_proc,
        mk_name_to_mk_attr_map &mk_attr_map) {

    auto M = cmd_info.usr_cfg_kernel_attr.kernel_attr["mat_m"];
    auto N = cmd_info.usr_cfg_kernel_attr.kernel_attr["mat_n"];
    auto K = cmd_info.usr_cfg_kernel_attr.kernel_attr["mat_k"];
    auto layout_b = cmd_info.usr_cfg_kernel_attr.kernel_attr["layout_b"];

    tuned_result cache_result;
    bool is_row_major = layout_b != "mem_layout::column_major";

    optimal_tune_result optimal_res = {{"", 0, 65535}, "", false};
    gemm_tune_cfg cfg_res;

    auto res = cache_result.get_tuned_result(
            stoi(M), stoi(N), stoi(K), is_row_major, cfg_res);
    if (!res) {
        tune_result_rec->write_log(
                "No cache data is found during tuning in tuned result. M:" + M
                + ",N:" + N + ",K:" + K + ",B layout:" + layout_b + "\n");
        optimal_res.tune_succ = false;
        return optimal_res;
    }
    smap_t tuning_params;
    for (auto tune_para : tunning_para_name_list) {
        if (cfg_res.find(tune_para) != cfg_res.end()) {
            tuning_params[tune_para] = cfg_res.at(tune_para);
        } else {
            std::cout << tune_para << " is not found in existed tune result!"
                      << std::endl;
            tuning_params[tune_para] = "1"; //
        }
    }
    for (auto one_op_fuse_elem : all_op_cart_proc) {
        vector<mk_info> mk_sequence;
        for (auto mk_name : one_op_fuse_elem) {
            mk_info cur_mk_info {mk_attr_map[mk_name].micro_kernel_desc.mk_type,
                    mk_attr_map[mk_name].micro_kernel_desc.micro_kernel_path
                            + "/" + mk_name};
            mk_sequence.push_back(cur_mk_info);
            tune_one_cfg_combination(
                    cmd_info, mk_sequence, tuning_params, optimal_res);
        }
    }

    return optimal_res;
}
} // namespace tuner_ns