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
#ifndef TUNE_STRATEGY_H
#define TUNE_STRATEGY_H

#include "explorer.h"
#include "tuner_comm.h"

namespace tuner_ns {

class tune_strategy {
public:
    virtual optimal_tune_result run_tuning(cmdline_input_info &cmd_info,
            vector<vector<string>> &all_op_cart_proc,
            mk_name_to_mk_attr_map &mk_attr_map)
            = 0;
    tune_strategy(explorer *explorer_ptr);
    virtual ~tune_strategy() {};
    void set_tune_para_name_list(std::vector<string> &tuning_names);

protected:
    bool tune_one_cfg_combination(cmdline_input_info &cmd_info,
            vector<mk_info> &mk_sequence, smap_t &tuning_params,
            optimal_tune_result &optimal_res);

    explorer *cur_explorer = nullptr;
    shared_ptr<tuner_log> tune_result_rec
            = tuner_log_factory::get_instance().get_log(TL_TUNE_RESULT);
    shared_ptr<tuner_log> tune_output_rec
            = tuner_log_factory::get_instance().get_log(TL_TUNE_OUTPUT);
    shared_ptr<tuner_log> tune_dbg_rec
            = tuner_log_factory::get_instance().get_log(TL_DBG_INFO);

    std::vector<std::string> tunning_para_name_list;
};

} // namespace tuner_ns

#endif