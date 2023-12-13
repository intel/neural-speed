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
#include "cmdline_parser.h"
#include <algorithm>

namespace tuner_ns {
using namespace std;
map<string, string> tuner_support_dtypes
        = {{"bf16", "bf16"}, {"fp16", "fp16"}, {"f32", "float"}};
map<micro_kernel_type, std::string> micro_kernel_type_map = {
        {MK_GEMM, "GEMM"},
        {MK_MHA, "MHA"},
        {MK_SELF_DEFINE_KERNEL, "SELF_DEFINE_KERNEL"},
        {MK_NOT_SUPPORT, "NOT_SUPPORT"},
};

vector<string> cmdline_parser::suported_cmd_option_name = {
        // GEMM
        "operation",
        "m",
        "n",
        "k",
        "batch_size",
        "A-shape",
        "B-shape",
        "C-shape",
        // MHA
        "B",
        "N",
        "F",
        "T",
        "H",
        "data-type",
        "layout",
        // Self define
        "cfg-para-list",

        //all support
        "output",
        "verification-enabled",
};

std::map<code_gen_info_type, std::string> code_gen_info_map = {
        {CG_STATIC_SIZET, "static constexpr size_t"},
        {CG_STATIC_UINT32, "static constexpr uint32_t"},
        {CG_STATIC_USING, "using"},
        {CG_STATIC_MEM_LAYOUT, "static constexpr mem_layout"},
        {CG_STATIC_MEM_LAYOUT, "static constexpr string"},
        {CG_NO_NEED_CODE_GEN, ""},
};

bool cmdline_parser::init_cmdline_parser(int argc, const char **argv) {
    if (1 == argc) {
        cout << "no user input cmd;\n";
        return false;
    }

    map<string, string> cmdline_info;
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];

        if ((arg[0] != '-') || (arg[1] != '-')) {
            cout << "The input parameter " << arg << " is not supported!\n";
            continue;
        }

        string help_str = arg;
        transform(
                help_str.begin(), help_str.end(), help_str.begin(), ::tolower);
        if (help_str == "--help") {
            print_help();
            return false;
        }

        string::size_type pos;
        string key, val;
        if ((pos = arg.find('=')) == string::npos) {
            key = string(arg, 2, arg.length() - 2);
            val = "";
        } else {
            key = string(arg, 2, pos - 2);
            val = string(arg, pos + 1, arg.length() - 1);
        }

        cmdline_info[key] = val;
    }

    // get micro kernel list
    if (!parse_user_cmdline(cmdline_info)) {
        cout << "Parse cmd line failed\n";
        return false;
    }

    if (!check_cmd()) {
        print_help();
        return false;
    }

    return true;
}

bool cmdline_parser::parse_user_cmdline(map<string, string> &cmdline_info) {
    string cur_res;
    vector<string> operation_list;
    get_cmdline_argument(cmdline_info, "operation", cur_res);
    string_split_by_char(cur_res, ',', operation_list);

    if (0 == operation_list.size()) {
        cout << "Invalid cmdline. No opration!\n";
        return false;
    }

    if (operation_list.size() > 1) {
        cout << "Currently only support one operation!\n";
        return false;
    }

    for (auto mk_name : operation_list) {
        auto mt = micro_kernel_type_map.begin();
        for (; mt != micro_kernel_type_map.end(); ++mt) {
            if (mt->second == mk_name) {
                usr_cmdline_info.mk_list.push_back(mt->first);
                break;
            }
        }
        if (mt == micro_kernel_type_map.end()) {
            cout << "Invalid operation:" << mk_name << "!\n";
            return false;
        }
    }

    if (!get_output_cmd_info(cmdline_info)) { return false; }
    if (!get_verification_enabled(cmdline_info)) { return false; }

    bool res = true;
    switch (usr_cmdline_info.mk_list[0]) {
        case MK_GEMM: {
            res = parse_gemm_usr_cmd_info(cmdline_info);
            break;
        }
        case MK_MHA: {
            res = parse_mha_usr_cmd_info(cmdline_info);
            break;
        }
        default: res = parse_common_format_usr_cmd_info(cmdline_info); break;
    }

    return res;
}

bool cmdline_parser::check_cmd() {
    return true;
    if (usr_cmdline_info.mk_list[0] == MK_SELF_DEFINE_KERNEL) {
        cout << "Self define micro kernel. Please check parameter by "
                "yourself.\n";
        return true;
    }
    for (auto option : usr_cmdline_info.usr_cfg_kernel_attr.kernel_attr) {
        if (find(suported_cmd_option_name.begin(),
                    suported_cmd_option_name.end(), option.first)
                == suported_cmd_option_name.end()) {
            cout << "check_cmd " << option.first << " is not supported!\n";
            return false;
        }
    }

    return true;
}
void cmdline_parser::get_cmdline_argument(map<string, string> &cmdline_info,
        const string &arg_name, string &value) {

    if (cmdline_info.find(arg_name) != cmdline_info.end()) {
        value = cmdline_info[arg_name];
    } else {
        value = "";
    }
}

bool cmdline_parser::is_number(const std::string &s) {
    return !s.empty() && std::find_if(s.begin(), s.end(), [](unsigned char c) {
        return !std::isdigit(c);
    }) == s.end();
}

bool cmdline_parser::parse_common_format_usr_cmd_info(
        map<string, string> &cmdline_info) {
    auto &kernel_cfg_attr = usr_cmdline_info.usr_cfg_kernel_attr.kernel_attr;
    auto &code_gen_info = usr_cmdline_info.usr_cfg_kernel_attr.code_gen_info;
    string res {""};
    get_cmdline_argument(cmdline_info, "cfg-para-list", res);
    char cur_delimit = ',';
    vector<string> cfg_para_list;
    string_split_by_char(res, cur_delimit, cfg_para_list);

    cur_delimit = ':';
    for (auto cfg_attr : cfg_para_list) {
        vector<string> attr;
        string_split_by_char(cfg_attr, cur_delimit, attr);
        if (attr.size() < 2) {
            cout << "Invalid configuration parameter: " << cfg_attr << endl;
            return false;
        }
        kernel_cfg_attr[attr[0]] = attr[1];
        if (is_number(attr[0])) {
            code_gen_info[attr[0]] = CG_STATIC_SIZET;
        } else if (attr[0].find("layout") != string::npos) {
            code_gen_info[attr[0]] = CG_STATIC_MEM_LAYOUT;
        } else if (attr[0].find("data_type") != string::npos) {
            code_gen_info[attr[0]] = CG_STATIC_USING;
        } else {
            code_gen_info[attr[0]] = CG_STATIC_STR;
        }
    }

    return true;
}

string cmdline_parser::get_cmd_option(const string &option_key) {
    string value = "";
    get_cmdline_argument(usr_cmdline_info.usr_cfg_kernel_attr.kernel_attr,
            option_key, value);
    return value;
}
bool cmdline_parser::get_output_cmd_info(map<string, string> &cmdline_info) {
    auto &kernel_cfg_attr = usr_cmdline_info.usr_cfg_kernel_attr.kernel_attr;
    string output_name("");
    string output_key("output");
    string suffix = (".csv");
    get_cmdline_argument(cmdline_info, "output", output_name);

    if (output_name != "") {
        if ((suffix.length() < output_name.length())
                && (output_name.rfind(suffix)
                        == (output_name.length() - suffix.length()))) {
            kernel_cfg_attr[output_key] = output_name;
        } else {
            std::cout << "The value of output can only be filename.csv. output "
                         "value is: "
                      << output_name << std::endl;
            return false;
        }
    } else {
        kernel_cfg_attr[output_key] = "default_tune_output_"
                + tuner_log_factory::get_instance().get_cur_time() + ".csv";
    }
    return true;
}

bool cmdline_parser::get_verification_enabled(
        map<string, string> &cmdline_info) {
    auto &kernel_cfg_attr = usr_cmdline_info.usr_cfg_kernel_attr.kernel_attr;
    string verify_value = "";
    string key = "verification-enabled";
    get_cmdline_argument(cmdline_info, key, verify_value);
    if (verify_value != "") {
        if ((verify_value == "true") || (verify_value == "false")) {
            kernel_cfg_attr["verification-enabled"] = verify_value;
        } else {
            std::cout << "The value of " << key
                      << " can only be true or false. The input "
                         "value is: "
                      << verify_value << std::endl;
            return false;
        }
    } else {
        kernel_cfg_attr["verification-enabled"] = "true";
    }
    return true;
}
bool cmdline_parser::parse_gemm_usr_cmd_info(
        map<string, string> &cmdline_info) {
    auto &kernel_cfg_attr = usr_cmdline_info.usr_cfg_kernel_attr.kernel_attr;
    auto &code_gen_info = usr_cmdline_info.usr_cfg_kernel_attr.code_gen_info;
    string res {""};

    for (auto key : {"m", "n", "k"}) {
        get_cmdline_argument(cmdline_info, key, res);

        kernel_cfg_attr["mat_" + string(key)] = (res != "") ? res : "0";
        code_gen_info["mat_" + string(key)] = CG_STATIC_SIZET;
    }

    int batch_size = 1;
    get_cmdline_argument(cmdline_info, "batch_size", res);
    if (res != "") { batch_size = stoi(res); }
    kernel_cfg_attr["batch_size"] = to_string(batch_size);
    code_gen_info["batch_size"] = CG_STATIC_SIZET;

    vector<string> all_shape {"A-shape", "B-shape", "C-shape"};
    typedef struct shape_info_stru {
        string layout;
        string data_type;
    } shape_info;
    vector<shape_info> shape_info_list;
    for (auto shape : all_shape) {
        vector<string> res_vec;
        get_cmdline_argument(cmdline_info, shape, res);
        string_split_by_char(res, ':', res_vec);
        string data_type = res_vec[0];
        data_type = (tuner_support_dtypes.find(data_type)
                            != tuner_support_dtypes.end())
                ? tuner_support_dtypes[data_type]
                : "NULL";
        string data_layout = res_vec[1];

        shape_info_list.push_back({data_layout, data_type});
    }
    kernel_cfg_attr["layout_a"]
            = "mem_layout::" + shape_info_list[0].layout + "_major";
    kernel_cfg_attr["data_type_a"] = shape_info_list[0].data_type;

    kernel_cfg_attr["layout_b"]
            = "mem_layout::" + shape_info_list[1].layout + "_major";
    kernel_cfg_attr["data_type_b"] = shape_info_list[1].data_type;

    kernel_cfg_attr["layout_c"]
            = "mem_layout::" + shape_info_list[2].layout + "_major";
    kernel_cfg_attr["data_type_c"] = shape_info_list[2].data_type;

    code_gen_info["layout_a"] = CG_STATIC_MEM_LAYOUT;
    code_gen_info["layout_b"] = CG_STATIC_MEM_LAYOUT;
    code_gen_info["layout_c"] = CG_STATIC_MEM_LAYOUT;

    code_gen_info["data_type_a"] = CG_STATIC_USING;
    code_gen_info["data_type_b"] = CG_STATIC_USING;
    code_gen_info["data_type_c"] = CG_STATIC_USING;

    return true;
}

bool cmdline_parser::parse_mha_usr_cmd_info(map<string, string> &cmdline_info) {
    auto &kernel_cfg_attr = usr_cmdline_info.usr_cfg_kernel_attr.kernel_attr;
    auto &code_gen_info = usr_cmdline_info.usr_cfg_kernel_attr.code_gen_info;
    string res;
    for (auto key : {"B", "N", "F", "T", "H"}) {
        get_cmdline_argument(cmdline_info, key, res);
        kernel_cfg_attr[key] = (res != "") ? res : "0";
        code_gen_info[key] = CG_STATIC_SIZET;
    }
    return true;
}
void cmdline_parser::get_usr_cmd_info(cmdline_input_info &cmd_info) {
    cmd_info = usr_cmdline_info;
}

void cmdline_parser::string_split_by_char(
        const string &src_str, const char delimiter, vector<string> &res) {
    if (src_str == "") { return; }

    string strs = src_str + delimiter;
    size_t pos = strs.find(delimiter);

    while (pos != strs.npos) {
        string temp = strs.substr(0, pos);
        res.push_back(temp);

        strs = strs.substr(pos + 1);
        pos = strs.find(delimiter);
    }
}

void cmdline_parser::print_user_cmd_info() {
    cout << "user cmd:\noperation:\n";
    for (auto oper : usr_cmdline_info.mk_list) {
        cout << micro_kernel_type_map[oper] << ",";
    }
    cout << "\n";

    cout << "cfg attr:\n";
    for (auto cfg : usr_cmdline_info.usr_cfg_kernel_attr.kernel_attr) {
        cout << cfg.first << ":" << cfg.second << "\n";
    }
}
} // namespace tuner_ns
