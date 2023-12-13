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
#ifndef TUNE_IN_RECOMMENDATION_H
#define TUNE_IN_RECOMMENDATION_H

#include <vector>
#include "explorer.h"
#include "tune_strategy.h"
#include "tuner_comm.h"

namespace tuner_ns {
class tune_cfg_recommendation {
public:
    std::vector<gemm_tune_cfg> &get_tune_cfg_recommendation(
            size_t m, size_t n, size_t k, bool is_row_major);

private:
    static std::vector<gemm_tune_cfg>
            tune_cfg_recommendation_matrix_B_row_major;
    static std::vector<gemm_tune_cfg>
            tune_cfg_recommendation_matrix_B_col_major;
};

class tune_in_recommendation_cfg : public tune_strategy {
public:
    tune_in_recommendation_cfg(explorer *cur_explorer_ptr)
        : tune_strategy(cur_explorer_ptr) {

        };
    optimal_tune_result run_tuning(cmdline_input_info &cmd_info,
            vector<vector<string>> &all_op_cart_proc,
            mk_name_to_mk_attr_map &mk_attr_map);

private:
    bool tune_one_mk_sequence(cmdline_input_info &cmd_info,
            vector<mk_info> &mk_sequence, optimal_tune_result &optimal_res);
};

} // namespace tuner_ns

#endif