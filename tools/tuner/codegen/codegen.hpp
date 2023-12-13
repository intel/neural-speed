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
#ifndef TUNNER_CODEGEN_HPP
#define TUNNER_CODEGEN_HPP

#include <vector>
#include "tuner_comm.h"
#include "tuner_log_factory.h"
#include <unordered_map>

namespace tuner_ns {
class kernel_generator {
public:
    kernel_generator() {
        mk_detail_dir = false;
        detail_dir = "";
    };

    bool gen_kernel(std::string &out_path, const vector<mk_info> &mk_sequence,
            const kernel_cfg_input_info &input_info,
            const smap_t &tuning_params);

    bool gen_un_fuse_kernel_para_cfg_file(std::string &out_path,
            const vector<mk_info> &mk_sequence,
            const kernel_cfg_input_info &input_info,
            const smap_t &tuning_params);

    void gen_kernel_path(std::string &out_path,
            const vector<mk_info> &mk_sequence,
            const kernel_cfg_input_info &input_info,
            const smap_t &tuning_params);
    bool gen_kernel_prepare();
    bool clean_generated_info(string &kernel_pat, bool reserve = true);
    void set_tune_para_name_list(std::vector<string> &tuning_names);
    void set_tune_para_code_gen_info(
            std::map<string, code_gen_info_type> &code_gen);

private:
    std::string exec(const std::string &cmd);
    std::vector<std::string> tunning_para_name_list;
    std::map<std::string, code_gen_info_type> tunning_para_code_gen_info;
    bool mk_detail_dir = false;
    std::string detail_dir = "";
    uint32_t reserved_generated_kernel = 5;
};
} // namespace tuner_ns

#endif
