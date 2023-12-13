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
#include "runner.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <unistd.h>
#include "../util/include/tuner_comm.h"

namespace tuner_ns {

runner::runner() {
    std::string env = "XETLA_TUNER_CMD_TIME_OUT";
    auto time_out = get_env(env.c_str());
    if (time_out != "") {
        max_cmd_time_out = stoi(time_out);
    } else {
        max_cmd_time_out = 30;
    }
};

runner::~runner() {

};

bool runner::build_unfused_kernel(std::string &out_path,
        const std::string kernel_path,
        const kernel_cfg_input_info &input_info) {
    std::string tune_root_dir = kernel_path.substr(0, kernel_path.rfind('/'));
    if (!std::filesystem::exists(tune_root_dir)) {
        std::cout << "tune kernel dir " << tune_root_dir << " is not exist!\n";
        return false;
    }
    out_path = tune_root_dir + "/build/tune_kernel";

    std::string cmdline = "cd " + tune_root_dir
            + ";mkdir -p build;cd build;rm -rf *;cmake ..;make -j";
    exec(cmdline);
    return std::filesystem::exists(out_path);
}

bool runner::build(std::string &out_path, const std::string kernel_path,
        const kernel_cfg_input_info &input_info) {

    if (false == multi_micro_kernel_fuse) {
        return build_unfused_kernel(out_path, kernel_path, input_info);
    } else {
        std::cout << "Tuner does not support to fuse kernel!\n";
        return false;
    }

    return true;
}

std::string runner::exec(const std::string &cmd) {
    char buffer[128];
    std::string result;
    FILE *pipe = popen(cmd.c_str(), "r");
    if (!pipe) throw std::runtime_error("\n   popen() failed!");
    try {
        while (fgets(buffer, sizeof buffer, pipe) != NULL) {
            result += buffer;
        }
    } catch (...) {
        pclose(pipe);
        throw;
    }
    pclose(pipe);
    return result;
}

bool runner::build_kernel_perf_test(const std::string &kernel_path,
        const kernel_cfg_input_info &input_info,
        kernel_perf_test_info &perf_test_info) {
    perf_test_info.test_bin_full_path = "";
    auto res
            = build(perf_test_info.test_bin_full_path, kernel_path, input_info);
    perf_test_info.kernel_path = kernel_path;
    return res;
}

string runner::get_value_in_result(
        string &result, string &key, const string &value_end) {
    auto start_pos = result.find(key, 0);
    if (start_pos == std::string::npos) { return ""; }

    auto end_pos = result.find(value_end, start_pos);
    start_pos += key.length();
    if (end_pos < start_pos) { return ""; }
    auto value_len = end_pos - start_pos;
    return result.substr(start_pos, value_len);
}

bool runner::run_kernel_perf_test(
        kernel_perf_test_info &perf_test_info, runner_record &perf_data) {
    perf_data.accuracy = false;
    if (access(perf_test_info.test_bin_full_path.c_str(), F_OK) == -1) {
        return false;
    }
    std::string cmdline {
            perf_test_info.run_env + "timeout " + to_string(max_cmd_time_out)};
    cmdline.append(" " + perf_test_info.test_bin_full_path);
    if (this->need_validate_res) { cmdline.append(" 1"); }
    cout << "cmd:" << cmdline << endl;
    cout << endl;
    cout << endl;
    std::string result = exec(cmdline);
    cout << "\n================\n" << result;
    // check if the kernel is skipped
    // [  SKIPPED ] 1 test, listed below:
    string skip_keyword {"[  SKIPPED ]"};
    auto start_pos = result.find(skip_keyword, 0);
    if (start_pos != std::string::npos) {
        std::cout << "The kernel is skipped.\n";
        return false;
    }

    string exec_time_keyword {
            "The mean(exclude the first trial) running(GPU_time) time is "};
    auto time_str = get_value_in_result(result, exec_time_keyword, " ms");
    if (time_str == "") {
        std::cout << "Can not get performance data! The build result of the "
                     "kernel is failed or the kernel execution time is larger "
                     "than "
                  << to_string(max_cmd_time_out) << "s.\n";
        return false;
    }
    perf_data.kernel_time = std::stod(time_str);

    string flops_keyword {"The median  gflops(GPU_time) is "};
    auto flops_str = get_value_in_result(result, flops_keyword, "\n");
    if (flops_str == "") {
        std::cout << "Can not get performance data! The build result of the "
                     "kernel is failed or the kernel execution time is larger "
                     "than "
                  << to_string(max_cmd_time_out) << "s.\n";
        return false;
    }
    perf_data.kernel_flops = std::stod(flops_str);

    perf_data.kernel_path = perf_test_info.kernel_path;
    perf_data.accuracy = true;
    if (need_validate_res) {
        auto start_pos = result.find("[  PASSED  ] 1 test.", 0);
        if (start_pos == std::string::npos) {
            std::cout << "accuracy is fail!!" << std::endl;
            perf_data.accuracy = false;
        }
    }
    return true;
}

bool runner::run(const std::string &kernel_path,
        const kernel_cfg_input_info &input_info, runner_record &r) {
    kernel_perf_test_info perf_test_info;
    r.accuracy = false;
    r.kernel_path = "";

    auto res = build_kernel_perf_test(kernel_path, input_info, perf_test_info);
    if (false == res) { return false; }
    res = run_kernel_perf_test(perf_test_info, r);
    return res;
}

std::string runner::get_env(const char *env_name) {
    std::string env_value = "";
    char *value = getenv(env_name);
    if (value != NULL) {
        env_value = value;
    } else {
        std::cerr << "environment variable " << env_name << " is not set\n";
    }
    return env_value;
};

} // namespace tuner_ns
