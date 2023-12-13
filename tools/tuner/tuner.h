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
#ifndef TUNER_H
#define TUNER_H
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "explorer.h"
#include "selector.h"
#include "tuner_comm.h"

namespace tuner_ns {
class tuner {
public:
    tuner();
    ~tuner() {};
    bool on_receive_cmdline(int argc, const char **argv);

private:
    void get_mk_set_in_each_operation(cmdline_input_info &cmd_info,
            vector<vector<one_micro_kernel_tune_info>> &total_search_space);
    void init();
    void print_kernel_tune_attr(
            vector<vector<one_micro_kernel_tune_info>> &total_search_space);
    void print_micro_kernel_info(
            map<string, vector<micro_kernel_info>> &mk_operation);
    void init_default_log_list();
    string get_output_file(cmdline_input_info &cmd_info) {
        return cur_selector->get_cmd_option("output");
    }

    void get_tuning_para_code_gen_info(std::vector<string> &tuning_name_list,
            std::map<string, code_gen_info_type> &code_gen_info);
    void sort_csv();

private:
    std::shared_ptr<selector> cur_selector;
    shared_ptr<explorer> cur_explored = nullptr;
};
} // namespace tuner_ns

#endif