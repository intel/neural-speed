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
#ifndef CMDLINE_PARSER_H
#define CMDLINE_PARSER_H
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "tuner_comm.h"
#include "tuner_log_factory.h"

namespace tuner_ns {
using namespace std;
class cmdline_parser {
public:
    cmdline_parser() {};
    bool init_cmdline_parser(int argc, const char **argv);
    // to do: suppport default parameter or partial defalut parameter
    void print_help() {
        //  "./tuner --operation=GEMM  --m=4096 --n=4096 --k=2048 "
        //  "--A-shape=bf16:row --B-shape=bf16:row --C-shape=f32:row [--batch_size=1]\n"
        //  "./tuner --operation=MHA  --B=64 --N=16 --F=384 --T=384 --H=64 "
        //  "--data-type=bf16  --layout=row\n"
        //  "./tuner --operation=SELF_DEFINE_KERNEL "
        //  "--cfg-para-list=m:4096,n:4096,k:2048,batch_num:32 "
        //  "--shape-list=bf16:row:input:268435456,bf16:row:input:"
        //  "268435456,f32:row:output:16777216  "
        //  "--ops=2*m*n*k*batch_num+m*n*batch_num\n"
        std::cout << "============Tuner supports cmd=========\n"
                     "./tuner --operation=GEMM  --m=4096 --n=4096 --k=2048 "
                     "--A-shape=bf16:row --B-shape=bf16:row --C-shape=f32:row "
                     "--output=tune_output.csv --verification-enabled=true\n"
                     "=======================================\n";
    };
    void get_usr_cmd_info(cmdline_input_info &cmd_Info);
    string get_cmd_option(const string &option_key);
    bool check_cmd();
    void print_user_cmd_info();

private:
    bool parse_user_cmdline(map<string, string> &cmdline_info);
    void get_cmdline_argument(map<string, string> &cmdline_info,
            const string &arg_name, string &value);
    bool parse_common_format_usr_cmd_info(map<string, string> &cmdline_info);
    bool parse_gemm_usr_cmd_info(map<string, string> &cmdline_info);
    bool parse_mha_usr_cmd_info(map<string, string> &cmdline_info);
    void string_split_by_char(
            const string &str, const char delimiter, vector<string> &res);
    bool get_output_cmd_info(map<string, string> &cmdline_info);
    bool get_verification_enabled(map<string, string> &cmdline_info);

private:
    cmdline_input_info usr_cmdline_info;
    static vector<string> suported_cmd_option_name;
    bool is_number(const std::string &s);
};
} // namespace tuner_ns

#endif