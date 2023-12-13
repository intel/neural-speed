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
#ifndef RUNNER_HPP
#define RUNNER_HPP

#include <fstream>
#include <vector>
#include "tuner_comm.h"
#include <unordered_map>

namespace tuner_ns {

class runner {
public:
    runner();
    ~runner();
    bool run(const std::string &kernel_path,
            const kernel_cfg_input_info &input_info, runner_record &r);
    bool run_kernel_perf_test(
            kernel_perf_test_info &perf_test_info, runner_record &run_res);
    bool build_kernel_perf_test(const std::string &kernel_path,
            const kernel_cfg_input_info &input_info,
            kernel_perf_test_info &perf_test_info);
    void set_validate_res(bool validate_res) {
        need_validate_res = validate_res;
    }

private:
    bool build(std::string &out_path, const std::string kernel_path,
            const kernel_cfg_input_info &input_info);
    bool build_unfused_kernel(std::string &out_path,
            const std::string kernel_path,
            const kernel_cfg_input_info &input_info);
    std::string exec(const std::string &cmd);
    std::string get_env(const char *env_name);
    string get_value_in_result(
            string &result, string &key, const string &value_end);

    bool need_validate_res = false;
    int max_cmd_time_out = 30; //unit: s
    bool multi_micro_kernel_fuse = false;
};
} // namespace tuner_ns

#endif
