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
#ifndef EXPLORER_H
#define EXPLORER_H
#include <algorithm>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "codegen.hpp"
#include "runner.hpp"
#include "tuner_comm.h"
#include "tuner_log_factory.h"
#include "tuner_thread_pool.h"

namespace tuner_ns {

typedef struct run_kernel_result_in_thread {
    uint32_t src_pos;
    run_kernel_result run_res;
} run_kernel_result_in_thread;
class explorer {
public:
    explorer();
    ~explorer();
    void tune_kernel(cmdline_input_info &cmd_info,
            vector<vector<one_micro_kernel_tune_info>> &tune_mk_set);
    shared_ptr<kernel_generator> get_cur_kernel_gen() { return kernel_gen; };
    shared_ptr<runner> get_cur_runner() { return kernel_runner; };
    string &get_run_tile_evn() { return run_kernel_tile_env; };
    string get_cur_time();
    std::vector<std::string> &get_tune_para_name_list() {
        return tunning_para_name_list;
    };
    void set_validate(bool enable) { kernel_runner->set_validate_res(enable); };
    void set_tune_para_name_list(std::vector<string> &tuning_names);
    void set_tune_para_code_gen_info(
            std::map<string, code_gen_info_type> &code_gen);

private:
    bool fast_tune(cmdline_input_info &cmd_info,
            vector<vector<string>> &all_op_cart_proc,
            mk_name_to_mk_attr_map &mk_attr_map,
            vector<vector<one_micro_kernel_tune_info>> &tune_mk_set);
    bool tune_kernel_one_by_one(cmdline_input_info &cmd_info,
            vector<vector<string>> &all_op_cart_proc,
            mk_name_to_mk_attr_map &mk_attr_map,
            map<string, vector<vector<string>>> &mk_tune_para_set);
    void print_cart_proc(vector<vector<string>>);
    template <typename T>
    void cartesian(vector<vector<T>> &v, vector<vector<T>> &res);
    void range_to_vector(tune_attr_cfg &tune_cfg, vector<string> &res);
    void save_better_kernel_info(
            build_kernel_perf_test_result &one_build_result,
            runner_record &perf_data);
    void parse_thread_pool_exec_result(
            std::vector<std::future<build_kernel_perf_test_result>>
                    &tune_results,
            runner_record &best_kernel_perf, tuner_thread_pool &runner_pool,
            int &cur_run_thread_num,
            std::vector<std::future<run_kernel_result_in_thread>> &run_results);
    void find_optimal_kernel(cmdline_input_info &cmd_info);
    bool check_is_valid_cfg(
            cmdline_input_info &cmd_info, smap_t &input_tune_para_set);

    std::string get_env(const char *env_name) {
        std::string env_value = "";
        char *value = getenv(env_name);
        if (value != NULL) {
            env_value = value;
        } else {
            std::cerr << "environment variable " << env_name << " is not set\n";
        }
        return env_value;
    };

    constexpr static int better_kernel_num = 10;
    constexpr static int max_validate_kernel_num = 3;
    constexpr static int tune_thread_pool_size = 50;
    constexpr static int max_run_kernel_num = 1; //4;
    int cur_build_thread_num = 0;
    std::map<double, build_kernel_perf_test_result> better_kernel_map;
    shared_ptr<kernel_generator> kernel_gen = make_shared<kernel_generator>();
    shared_ptr<runner> kernel_runner = make_shared<runner>();
    //     string run_kernel_tile_env = "export ZE_AFFINITY_MASK=1.0;";
    string run_kernel_tile_env = "";

    shared_ptr<tuner_log> tune_result_rec
            = tuner_log_factory::get_instance().get_log(TL_TUNE_RESULT);
    shared_ptr<tuner_log> tune_output_rec
            = tuner_log_factory::get_instance().get_log(TL_TUNE_OUTPUT);
    shared_ptr<tuner_log> tune_dbg_rec
            = tuner_log_factory::get_instance().get_log(TL_DBG_INFO);
    std::vector<std::string> tunning_para_name_list;
    std::map<string, code_gen_info_type> code_gen_info;
};
} // namespace tuner_ns

#endif
