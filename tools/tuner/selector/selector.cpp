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
#include "selector.h"

namespace tuner_ns {
selector::selector() {}
bool selector::parse_cmdline(
        int argc, const char **argv, cmdline_input_info &cmd_info) {

    if (false == cmd_parser->init_cmdline_parser(argc, argv)) { return false; }

    cmd_parser->get_usr_cmd_info(cmd_info);

    mk_manager = make_shared<micro_kernel_info_mng>();
    return true;
}

void selector::get_micro_kernel_info(
        micro_kernel_type mk_type, std::vector<micro_kernel_info> &info) {
    mk_manager->get_micro_kernel_info(mk_type, info);
}

std::shared_ptr<cfg_attribute_parser> selector::get_cfg_parser(
        std::string &cfg_path) {
    if (kernel_cfg.find(cfg_path) != kernel_cfg.end()) {
        return kernel_cfg[cfg_path];
    }

    kernel_cfg[cfg_path] = make_shared<cfg_attribute_parser>(cfg_path);

    return kernel_cfg[cfg_path];
}

void selector::get_micro_kernel_tune_attr(
        string &cfg_path, tune_attr_vector &tune_attr) {
    auto cfg_parser = get_cfg_parser(cfg_path);
    cfg_parser->get_tune_parameters(tune_attr);
}

void selector::get_tune_parameter_names(
        std::string &cfg_path, std::vector<std::string> &tune_para_names) {
    auto cfg_parser = get_cfg_parser(cfg_path);
    cfg_parser->get_tune_parameter_names(tune_para_names);
}

string selector::get_cmd_option(const string &option_key) {
    return cmd_parser->get_cmd_option(option_key);
}

} // namespace tuner_ns