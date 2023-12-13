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
#include <algorithm>
#include <assert.h>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <numeric>
#include <sstream>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "codegen.hpp"
#include <sys/stat.h>

namespace tuner_ns {

std::string kernel_generator::exec(const std::string &cmd) {
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
bool kernel_generator::gen_un_fuse_kernel_para_cfg_file(std::string &out_path,
        const vector<mk_info> &mk_sequence,
        const kernel_cfg_input_info &input_info, const smap_t &tuning_params) {

    std::string micor_kernel_root_dir = mk_sequence.at(0).mk_path.substr(
            0, mk_sequence.at(0).mk_path.rfind("/"));
    std::string ori_test_hpp_path = micor_kernel_root_dir + "/test.hpp";
    std::string cmd = "cp -r " + micor_kernel_root_dir + "/* " + out_path + "/"
            + ";" + "rm -rf " + out_path + "/test.hpp";
    exec(cmd);

    std::string para_cfg_file = out_path + "/" + "test.hpp";
    ofstream os(para_cfg_file);

    auto pos = mk_sequence.at(0).mk_path.rfind("micro_kernel_repo");
    string kernel_rel_path = "";
    if (pos != std::string::npos) {
        kernel_rel_path = mk_sequence.at(0).mk_path.substr(pos);
    }
    os << "//kernel template: " << kernel_rel_path << "\n";

    ifstream infile(ori_test_hpp_path);
    std::string line;
    while (getline(infile, line)) {
        std::string str(line);
        if (str.rfind("using tests", 0) == 0) { break; }
        os << line << std::endl;
    }

    auto file_name = mk_sequence.at(0).mk_path.substr(
            mk_sequence.at(0).mk_path.rfind("/") + 1);
    auto base_name = file_name.substr(0, file_name.rfind("."));
    std::string new_case_name = "xetla_tune_cfg_para_" + base_name + "_"
            + tuner_log_factory::get_instance().get_cur_time("_") + "_"
            + to_string(rand() % 65535) + "_" + std::to_string(getpid());

    os << "class " << new_case_name << " : public TestBase {\n";
    os << "public:\n";
    // generate cfg attr
    auto &code_gen_info = input_info.code_gen_info;
    for (auto cfg : input_info.kernel_attr) {
        if (code_gen_info.find(cfg.first) != code_gen_info.end()) {
            os << code_gen_info_map.at(code_gen_info.at(cfg.first)) << " ";
            os << cfg.first + " = " << cfg.second << ";\n";
        } else {
            // os << "static constexpr size_t ";
            continue;
        }
    }
    // generate tune parameters list. we want to keep the parameter sequence.
    for (auto tune_para : tunning_para_name_list) {
        if (tuning_params.find(tune_para) != tuning_params.end()) {
            os << code_gen_info_map.at(tunning_para_code_gen_info.at(tune_para))
               << +" " << tune_para << +" = " << tuning_params.at(tune_para)
               << ";\n";
        } else {
            std::cout << tune_para << " is not in tuning parameter list."
                      << std::endl;
        }
    }
    os << "};\n";

    os << "using tests = ::testing::Types<" << new_case_name << ">;\n";
    os.close();

    return true;
}

bool kernel_generator::gen_kernel_prepare() {
    if (false == mk_detail_dir) {
        detail_dir = "./tune_detail_"
                + tuner_log_factory::get_instance().get_cur_time("_");
        auto ret = mkdir(
                detail_dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if (ret && errno == EEXIST) {
            cout << "dir: " << detail_dir << " aleardy exist" << endl;
            mk_detail_dir = true;
        } else if (ret) {
            cout << "create dir error: " << ret << ", errno:" << (errno)
                 << endl;
        } else {
            mk_detail_dir = true;
        }
    }

    if ((false == mk_detail_dir)) {
        std::cout << "Error: need to make root dir to store tune details!\n";
        return false;
    }

    return true;
}
bool kernel_generator::clean_generated_info(string &kernel_path, bool reserve) {
    if (reserve && (reserved_generated_kernel > 0)) {
        --reserved_generated_kernel;
        return true;
    }

    // ./tune_detail_2023-08-08-08-00-09/xetla_tune_mat_m_1_mat_k_4096_mat_n_8192_layout_a_row_major_layout_b_row_major_data_type_a_fp16_data_type_b_fp16_data_type_c_fp16_wg_m_256_wg_n_256_sg_m_64_sg_n_32_sg_k_64_global_kslicing_1_local_kslicing_1_42989_1268451/
    if (std::filesystem::exists(kernel_path)) {
        std::string cmd = "rm -rf " + kernel_path;
        exec(cmd);
        return true;
    } else {
        cout << kernel_path << " is not exist! It Can not be removed." << endl;
        return false;
    }
    return true;
}
void kernel_generator::gen_kernel_path(std::string &out_path,
        const vector<mk_info> &mk_sequence,
        const kernel_cfg_input_info &input_info, const smap_t &tuning_params) {
    uint32_t rand_postfix = rand() % 65535;
    auto file_name = mk_sequence.at(0).mk_path.substr(
            mk_sequence.at(0).mk_path.rfind("/") + 1);
    auto base_name = file_name.substr(0, file_name.rfind("."));
    string new_tmp_dir = "xetla_tune_case_" + base_name + "_"
            + tuner_log_factory::get_instance().get_cur_time("_") + "_"
            + to_string(rand_postfix) + "_" + std::to_string(getpid());
    out_path = detail_dir + "/" + new_tmp_dir + "/";
    auto res = mkdir(out_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    if (res && errno == EEXIST) {
        cout << "dir: " << out_path << " aleardy exist" << endl;
    } else if (res) {
        cout << "create dir error: " << res << ", errno:" << (errno) << endl;
    } else {
        //success
    }
}
bool kernel_generator::gen_kernel(std::string &out_path,
        const vector<mk_info> &mk_sequence,
        const kernel_cfg_input_info &input_info, const smap_t &tuning_params) {

    gen_kernel_path(out_path, mk_sequence, input_info, tuning_params);
    if (1 == mk_sequence.size()) {
        gen_un_fuse_kernel_para_cfg_file(
                out_path, mk_sequence, input_info, tuning_params);
    } else {
        std::cout << "Tuner does not support to fuse kernel!\n";
        return false;
    }

    return true;
}
void kernel_generator::set_tune_para_name_list(
        std::vector<string> &tuning_names) {
    tunning_para_name_list.assign(tuning_names.begin(), tuning_names.end());
}
void kernel_generator::set_tune_para_code_gen_info(
        std::map<string, code_gen_info_type> &code_gen) {
    tunning_para_code_gen_info = code_gen;
}
} // namespace tuner_ns
