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
#include "tune_strategy.h"
#include <filesystem>

namespace tuner_ns {
tune_strategy::tune_strategy(explorer *explorer_ptr)
    : cur_explorer(explorer_ptr) {
    set_tune_para_name_list(cur_explorer->get_tune_para_name_list());
}
bool tune_strategy::tune_one_cfg_combination(cmdline_input_info &cmd_info,
        vector<mk_info> &mk_sequence, smap_t &tuning_params,
        optimal_tune_result &optimal_res) {
    auto &best_kernel_perf = optimal_res.perf_data;

    auto kernel_gen = cur_explorer->get_cur_kernel_gen();
    kernel_gen->gen_kernel_prepare();

    bool find_new = false;
    bool need_save_kernel = false;
    string exec_reslut = "NULL";
    string exec_time_str = to_string(0xFFFFFFFF);
    string flops_str = to_string(0xFFFFFFFF);
    string tune_cfg = "Tuning para: ";

    for (auto cur_tune_para_name : cur_explorer->get_tune_para_name_list()) {
        tune_cfg.append(cur_tune_para_name + ": "
                + tuning_params[cur_tune_para_name] + ", ");
    }
    tune_cfg = tune_cfg.substr(0, tune_cfg.size() - 2);
    string out_path {""};
    build_kernel_perf_test_result build_res;
    build_res.build_succ = false;
    if (kernel_gen->gen_kernel(out_path, mk_sequence,
                cmd_info.usr_cfg_kernel_attr, tuning_params)) {
        if (cur_explorer->get_cur_runner()->build_kernel_perf_test(out_path,
                    cmd_info.usr_cfg_kernel_attr, build_res.build_res)) {
            build_res.build_succ = true;
        }
    }

    if (build_res.build_succ) {
        if (std::filesystem::exists(build_res.build_res.test_bin_full_path)) {
            build_res.build_res.run_env = cur_explorer->get_run_tile_evn();
            run_kernel_result_in_thread run_kernel_res;
            auto run_succ
                    = cur_explorer->get_cur_runner()->run_kernel_perf_test(
                            build_res.build_res,
                            run_kernel_res.run_res.perf_data);
            run_kernel_res.run_res.run_succ = run_succ;
            build_res.run_res = run_kernel_res.run_res;

            //parse run result
            if (build_res.run_res.run_succ) {
                auto perf_data = build_res.run_res.perf_data;
                if (perf_data.accuracy) {
                    if (perf_data.kernel_time < best_kernel_perf.kernel_time) {
                        if (best_kernel_perf.kernel_path != "") {
                            kernel_gen->clean_generated_info(
                                    best_kernel_perf.kernel_path, false);
                        }
                        optimal_res.perf_data = perf_data;
                        optimal_res.tune_cfg = tune_cfg;
                        optimal_res.tune_succ = true;
                        find_new = true;
                    }
                    exec_reslut = "Success";
                    exec_time_str = to_string(perf_data.kernel_time);
                    flops_str = to_string(perf_data.kernel_flops);
                } else {
                    exec_reslut = "Accuracy fail";
                }
            } else {
                exec_reslut = "Run perf test fail";
            }
        } else {
            exec_reslut = "Run fail. kernle is not exist";
        }

    } else {
        exec_reslut = "Build kernel fail";
    }
    string tune_para = "";
    for (auto cur_tune_para_name : cur_explorer->get_tune_para_name_list()) {
        tune_para += tuning_params[cur_tune_para_name] + ", ";
    }
    tune_para = tune_para.substr(0, tune_para.size() - 2);
    tune_output_rec->write_log(tune_para + "," + exec_time_str + "," + flops_str
            + "," + exec_reslut + "\n");
    if (find_new) {
        tune_dbg_rec->write_log("Find better kernel. Perf: "
                + to_string(best_kernel_perf.kernel_time)
                + ". Path: " + best_kernel_perf.kernel_path + ".\n");
    }
    return true;
}

void tune_strategy::set_tune_para_name_list(std::vector<string> &tuning_names) {
    tunning_para_name_list.assign(tuning_names.begin(), tuning_names.end());
}
} // namespace tuner_ns
