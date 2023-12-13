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
#include "tuner.h"
#include <unistd.h>
#include "tuner_log_factory.h"
namespace tuner_ns {
tuner::tuner() {
    init();
}
bool tuner::on_receive_cmdline(int argc, const char **argv) {
    cmdline_input_info cmd_info;
    if (false == cur_selector->parse_cmdline(argc, argv, cmd_info)) {
        return false;
    }

    if (0 == cmd_info.mk_list.size()) {
        cout << "no kernel is need to tune.\n";
        return false;
    }

    vector<vector<one_micro_kernel_tune_info>> tune_mk_set;
    get_mk_set_in_each_operation(cmd_info, tune_mk_set);

    // We require that the tuning sets of the same type of kernel are the same,
    // or we cannot record the output in one file

    std::vector<string> tuning_name_list;
    auto tune_attr = tune_mk_set.at(0).at(0).tune_attr;
    string tune_para_cfg = "";
    for (auto attr : tune_attr) {
        tuning_name_list.push_back(attr.attr_name);
        tune_para_cfg += attr.attr_name + ",";
    }

    // print_cmd_info(cmd_info);
    auto output_file = get_output_file(cmd_info);
    auto tune_output_rec = make_shared<tuner_log>(output_file);
    tuner_log_factory::get_instance().add_log_elem(
            TL_TUNE_OUTPUT, tune_output_rec);

    tune_output_rec->write_log(tune_para_cfg + "exec_time,GFlops,run_result\n");

    // print_kernel_tune_attr(total_search_space);
    cout << "\n---start tune\n";
    cur_explored = make_shared<explorer>();
    cur_explored->set_tune_para_name_list(tuning_name_list);

    std::map<string, code_gen_info_type> code_gen_info;
    get_tuning_para_code_gen_info(tuning_name_list, code_gen_info);
    cur_explored->set_tune_para_code_gen_info(code_gen_info);

    auto enable = cur_selector->get_cmd_option("verification-enabled");
    if (enable == "true") {
        cur_explored->set_validate(true);
    } else {
        cur_explored->set_validate(false);
    }

    cur_explored->tune_kernel(cmd_info, tune_mk_set);
    cout << "---end tune\n";

    sort_csv();

    return true;
}

void tuner::get_mk_set_in_each_operation(cmdline_input_info &cmd_info,
        vector<vector<one_micro_kernel_tune_info>> &total_search_space) {
    map<micro_kernel_type, vector<micro_kernel_info>> mk_operation;
    for (auto mk : cmd_info.mk_list) {
        cur_selector->get_micro_kernel_info(mk, mk_operation[mk]);
    }
    // print_micro_kernel_info(mk_operation);

    for (auto oper : mk_operation) {
        vector<one_micro_kernel_tune_info> one_type_mk;
        for (auto mk : oper.second) {
            one_micro_kernel_tune_info mk_tune;
            mk_tune.micro_kernel_desc = mk;
            string mk_cfg_path
                    = mk.micro_kernel_path + "/" + mk.attribute_file_name;
            cur_selector->get_micro_kernel_tune_attr(
                    mk_cfg_path, mk_tune.tune_attr);
            one_type_mk.push_back(mk_tune);
        }
        total_search_space.push_back(one_type_mk);
    }
}

void tuner::print_kernel_tune_attr(
        vector<vector<one_micro_kernel_tune_info>> &total_search_space) {
    for (auto op : total_search_space) {
        for (auto mk : op) {
            cout << "mk name:" << mk.micro_kernel_desc.micro_kernel_name;
            cout << "\nmk tune para:";
            for (auto tune_para : mk.tune_attr) {
                cout << "----para_name: " << tune_para.attr_name;
                cout << " start value: " << tune_para.attr.start_value;
                cout << " end value:   " << tune_para.attr.end_value;
                cout << " stride:      " << tune_para.attr.stride;
                cout << endl;
            }
        }
    }
}
void tuner::print_micro_kernel_info(
        map<string, vector<micro_kernel_info>> &mk_operation) {
    cout << "micro kernel info:\n";
    for (auto ops : mk_operation) {
        cout << "micro kernel type:" << ops.first << "\n";
        for (auto mk : ops.second) {
            cout << "----micro_kernel_name:" << mk.micro_kernel_name << "\n";
            cout << "----attribute_file_name:" << mk.attribute_file_name
                 << "\n";
            cout << "----micro_kernel_path:" << mk.micro_kernel_path << "\n";
            cout << "--------\n";
        }
    }
}

void tuner::init() {
    init_default_log_list();
    cur_selector = make_shared<selector>();
}

void tuner::init_default_log_list() {
    string cur_time = tuner_log_factory::get_instance().get_cur_time();
    stringstream sstream;
    sstream << "./tune_result_" << cur_time << "_" << to_string(getpid());
    string file_name = sstream.str();
    auto tune_result_rec = make_shared<tuner_log>(file_name);
    tuner_log_factory::get_instance().add_log_elem(
            TL_TUNE_RESULT, tune_result_rec);

    sstream.clear();
    sstream.str("");
    sstream << "./tune_dbg_info_" << cur_time << "_" << to_string(getpid());
    file_name = sstream.str();
    auto tune_dbg_rec = make_shared<tuner_log>(file_name);
    tuner_log_factory::get_instance().add_log_elem(TL_DBG_INFO, tune_dbg_rec);
}

void tuner::sort_csv() {
    auto output_log = tuner_log_factory::get_instance().get_log(TL_TUNE_OUTPUT);
    output_log->sort_log("exec_time");
}

void tuner::get_tuning_para_code_gen_info(std::vector<string> &tuning_name_list,
        std::map<string, code_gen_info_type> &code_gen_info) {
    // we should get code gen information of tuning parameters from yaml file in future.
    for (auto para : tuning_name_list) {
        code_gen_info[para] = CG_STATIC_SIZET;
    }
}
} // namespace tuner_ns