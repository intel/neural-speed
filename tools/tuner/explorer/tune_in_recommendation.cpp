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
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include "tune_in_recommendation.h"
#include <unordered_map>

namespace tuner_ns {
using namespace std;
std::vector<gemm_tune_cfg>
        tune_cfg_recommendation::tune_cfg_recommendation_matrix_B_row_major = {
                //wg_m,wg_n,sg_m,sg_n,sg_k,local_kslicing,global_kslicing
                {{"wg_m", "256"}, {"wg_n", "256"}, {"sg_m", "32"},
                        {"sg_n", "64"}, {"sg_k", "32"}, {"local_kslicing", "1"},
                        {"global_kslicing", "1"}},
                {{"wg_m", "8"}, {"wg_n", "512"}, {"sg_m", "8"}, {"sg_n", "16"},
                        {"sg_k", "16"}, {"local_kslicing", "1"},
                        {"global_kslicing", "1"}},
                {{"wg_m", "8"}, {"wg_n", "128"}, {"sg_m", "8"}, {"sg_n", "16"},
                        {"sg_k", "16"}, {"local_kslicing", "4"},
                        {"global_kslicing", "1"}},
                {{"wg_m", "32"}, {"wg_n", "64"}, {"sg_m", "8"}, {"sg_n", "16"},
                        {"sg_k", "16"}, {"local_kslicing", "2"},
                        {"global_kslicing", "1"}},
                {{"wg_m", "16"}, {"wg_n", "256"}, {"sg_m", "8"}, {"sg_n", "16"},
                        {"sg_k", "16"}, {"local_kslicing", "1"},
                        {"global_kslicing", "1"}},
                {{"wg_m", "16"}, {"wg_n", "1024"}, {"sg_m", "16"},
                        {"sg_n", "32"}, {"sg_k", "16"}, {"local_kslicing", "1"},
                        {"global_kslicing", "1"}},
                {{"wg_m", "128"}, {"wg_n", "64"}, {"sg_m", "32"},
                        {"sg_n", "16"}, {"sg_k", "32"}, {"local_kslicing", "2"},
                        {"global_kslicing", "1"}},
                {{"wg_m", "128"}, {"wg_n", "128"}, {"sg_m", "32"},
                        {"sg_n", "32"}, {"sg_k", "32"}, {"local_kslicing", "2"},
                        {"global_kslicing", "1"}}};
std::vector<gemm_tune_cfg>
        tune_cfg_recommendation::tune_cfg_recommendation_matrix_B_col_major
        = {{{"wg_m", "256"}, {"wg_n", "256"}, {"sg_m", "32"}, {"sg_n", "64"},
                   {"sg_k", "32"}, {"local_kslicing", "1"},
                   {"global_kslicing", "1"}},
                {{"wg_m", "64"}, {"wg_n", "16"}, {"sg_m", "16"}, {"sg_n", "16"},
                        {"sg_k", "64"}, {"local_kslicing", "8"},
                        {"global_kslicing", "1"}},
                {{"wg_m", "32"}, {"wg_n", "16"}, {"sg_m", "16"}, {"sg_n", "16"},
                        {"sg_k", "64"}, {"local_kslicing", "16"},
                        {"global_kslicing", "1"}},
                {{"wg_m", "32"}, {"wg_n", "16"}, {"sg_m", "8"}, {"sg_n", "16"},
                        {"sg_k", "64"}, {"local_kslicing", "1"},
                        {"global_kslicing", "1"}},
                {{"wg_m", "32"}, {"wg_n", "16"}, {"sg_m", "8"}, {"sg_n", "16"},
                        {"sg_k", "64"}, {"local_kslicing", "8"},
                        {"global_kslicing", "1"}},
                {{"wg_m", "128"}, {"wg_n", "16"}, {"sg_m", "64"},
                        {"sg_n", "16"}, {"sg_k", "32"},
                        {"local_kslicing", "16"}, {"global_kslicing", "1"}},
                {{"wg_m", "64"}, {"wg_n", "16"}, {"sg_m", "8"}, {"sg_n", "16"},
                        {"sg_k", "64"}, {"local_kslicing", "4"},
                        {"global_kslicing", "1"}}};

std::vector<gemm_tune_cfg> &
tune_cfg_recommendation::get_tune_cfg_recommendation(
        size_t m, size_t n, size_t k, bool is_row_major) {
    return (is_row_major ? tune_cfg_recommendation_matrix_B_row_major
                         : tune_cfg_recommendation_matrix_B_col_major);
}
bool tune_in_recommendation_cfg::tune_one_mk_sequence(
        cmdline_input_info &cmd_info, vector<mk_info> &mk_sequence,
        optimal_tune_result &optimal_res) {

    auto M = cmd_info.usr_cfg_kernel_attr.kernel_attr["mat_m"];
    auto N = cmd_info.usr_cfg_kernel_attr.kernel_attr["mat_n"];
    auto K = cmd_info.usr_cfg_kernel_attr.kernel_attr["mat_k"];
    auto layout_b = cmd_info.usr_cfg_kernel_attr.kernel_attr["layout_b"];

    tune_cfg_recommendation cache_result;
    bool is_row_major = layout_b != "mem_layout::column_major";

    std::vector<gemm_tune_cfg> &recommendation_cfg
            = cache_result.get_tune_cfg_recommendation(
                    stoi(M), stoi(N), stoi(K), is_row_major);
    auto kernel_gen = cur_explorer->get_cur_kernel_gen();
    kernel_gen->gen_kernel_prepare();

    for (auto recom_cfg : recommendation_cfg) {
        smap_t tuning_params;
        for (auto tune_para : tunning_para_name_list) {
            if (recom_cfg.find(tune_para) != recom_cfg.end()) {
                tuning_params[tune_para] = recom_cfg.at(tune_para);
            } else {
                std::cout << tune_para
                          << " is not found in recommendation tune result!"
                          << std::endl;
                tuning_params[tune_para] = "1"; //
            }
        }
        tune_one_cfg_combination(
                cmd_info, mk_sequence, tuning_params, optimal_res);
    }

    return true;
}
optimal_tune_result tune_in_recommendation_cfg::run_tuning(
        cmdline_input_info &cmd_info, vector<vector<string>> &all_op_cart_proc,
        mk_name_to_mk_attr_map &mk_attr_map) {
    optimal_tune_result optimal_res = {{"", 0, 65535}, "", false};
    for (auto one_op_fuse_elem : all_op_cart_proc) {
        vector<mk_info> mk_sequence;
        for (auto mk_name : one_op_fuse_elem) {
            mk_info cur_mk_info {mk_attr_map[mk_name].micro_kernel_desc.mk_type,
                    mk_attr_map[mk_name].micro_kernel_desc.micro_kernel_path
                            + "/" + mk_name};
            mk_sequence.push_back(cur_mk_info);

            tune_one_mk_sequence(cmd_info, mk_sequence, optimal_res);
        }
    }

    return optimal_res;
}

} // namespace tuner_ns

// recommendation cfg: https://github.com/xytintel/syclbench/blob/main/gemm/hgemm_xetla.cpp#L212
