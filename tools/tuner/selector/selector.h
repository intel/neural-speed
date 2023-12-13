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
#ifndef SELCETOR_H
#define SELCETOR_H
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "cfg_attribute_parser.h"
#include "cmdline_parser.h"
#include "micro_kernel_repo.h"

namespace tuner_ns {
using namespace std;
class selector {
public:
    selector();
    bool parse_cmdline(
            int argc, const char **argv, cmdline_input_info &cmd_info);
    void get_micro_kernel_info(
            micro_kernel_type mk_type, std::vector<micro_kernel_info> &info);
    void get_micro_kernel_tune_attr(
            string &cfg_path, tune_attr_vector &tune_attr);
    void get_tune_parameter_names(
            std::string &cfg_path, std::vector<std::string> &tune_para_names);

    string get_cmd_option(const string &option_key);

private:
    std::shared_ptr<cfg_attribute_parser> get_cfg_parser(std::string &cfg_path);
    shared_ptr<cmdline_parser> cmd_parser = make_shared<cmdline_parser>();
    shared_ptr<micro_kernel_info_mng> mk_manager;
    std::map<std::string, std::shared_ptr<cfg_attribute_parser>>
            kernel_cfg; //(cfg_path,cfg_attribute_parser_ptr)
};
} // namespace tuner_ns

#endif