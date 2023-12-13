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
#include "micro_kernel_repo.h"
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include "cfg_attribute_parser.h"

namespace tuner_ns {

namespace fs = std::filesystem;
using namespace std;

micro_kernel_info_mng::micro_kernel_info_mng()
    : micro_kernel_info_mng(MICRO_KERNEL_REPO_ROOT_PATH) {}
micro_kernel_info_mng::micro_kernel_info_mng(
        std::string micro_kernel_repo_location) {
    micro_kernel_root_dir = micro_kernel_repo_location;
    init_all_micro_kernel_info();
}
void micro_kernel_info_mng::get_micro_kernel_info(
        micro_kernel_type mk_type, std::vector<micro_kernel_info> &info) {
    if (all_micro_kernel_info.find(mk_type) != all_micro_kernel_info.end()) {
        info.assign(all_micro_kernel_info[mk_type].begin(),
                all_micro_kernel_info[mk_type].end());
        return;
    }
    std::cout << "get_micro_kernel_info--Micro kernel type " << mk_type
              << " is not supported." << std::endl;
}

void micro_kernel_info_mng::print_all_directory() {
    for (auto it = fs::recursive_directory_iterator(micro_kernel_root_dir);
            it != fs::recursive_directory_iterator(); ++it) {
        const string spacer(it.depth() * 2, ' ');
        auto &entry = *it;
        if (filesystem::is_regular_file(entry)) {
            cout << spacer << "File:" << entry;
            cout << "(" << filesystem::file_size(entry) << " bytes )" << endl;
        } else if (filesystem::is_directory(entry)) {
            cout << spacer << "Dir:" << entry << endl;
        }
    }
}
void micro_kernel_info_mng::print_micro_kernel_info() {
    cout << "print all micro info:" << endl;
    for (auto &cur : all_micro_kernel_info) {
        cout << cur.first << endl;
    }
}
void micro_kernel_info_mng::init_all_micro_kernel_info() {
    if (!std::filesystem::exists(micro_kernel_root_dir)) {
        cout << micro_kernel_root_dir << "is not exit!" << endl;
        return;
    }

    for (const auto entry :
            fs::recursive_directory_iterator(micro_kernel_root_dir)) {
        std::string path = micro_kernel_root_dir;
        std::string yaml_cfg_file_extension = ".yaml";
        if ((filesystem::is_regular_file(entry))
                && (yaml_cfg_file_extension == entry.path().extension())) {
            std::string curr_abs_path
                    = fs::absolute(entry.path().parent_path()).string();
            std::string yaml_file
                    = curr_abs_path + "/" + entry.path().filename().string();
            std::shared_ptr<cfg_attribute_parser> parser(
                    new cfg_attribute_parser(yaml_file));

            // file_type: micro_kernel_attri_cfg
            std::string yaml_attr_file_key_word = "micro_kernel_attri_cfg";
            std::string file_type = parser->get_cfg_attr_by_name("file_type");
            if (yaml_attr_file_key_word != file_type) {
                std::cout << entry << " is not micro kernel cfg file."
                          << " file type:" << file_type << std::endl;
                continue;
            }

            micro_kernel_info mk_Info;
            mk_Info.micro_kernel_path = curr_abs_path;
            mk_Info.attribute_file_name = entry.path().filename().string();
            // micro_kernel_name: mha_micro_kernel.cpp
            mk_Info.micro_kernel_name
                    = parser->get_cfg_attr_by_name("micro_kernel_name");

            // micro_kernel_type: MHA
            std::string mk_type_str
                    = parser->get_cfg_attr_by_name("micro_kernel_type");
            if (("" == mk_type_str) || ("" == mk_Info.micro_kernel_name)) {
                std::cout
                        << "Micro kernel type or micro kernel file is not exit."
                        << std::endl;
                continue;
            }
            auto mt = micro_kernel_type_map.begin();
            for (; mt != micro_kernel_type_map.end(); ++mt) {
                if (mt->second == mk_type_str) { break; }
            }
            if (mt == micro_kernel_type_map.end()) {
                std::cout << "Micro kernel type " << mk_type_str
                          << " is not supported. File path: " << yaml_file
                          << std::endl;
                continue;
            }

            auto mk_type = mt->first;
            mk_Info.mk_type = mk_type;
            if (all_micro_kernel_info.find(mk_type)
                    == all_micro_kernel_info.end()) {
                std::vector<micro_kernel_info> mk_info_vec = {mk_Info};
                all_micro_kernel_info[mk_type] = mk_info_vec;
            } else {
                all_micro_kernel_info[mk_type].push_back(mk_Info);
            }
        } else if (filesystem::is_directory(entry)) {
            //dir;
        }
    }
}
} // namespace tuner_ns
