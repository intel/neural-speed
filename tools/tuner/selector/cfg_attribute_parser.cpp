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

#include "cfg_attribute_parser.h"
#include <map>

using namespace YAML;

namespace tuner_ns {
cfg_attribute_parser::cfg_attribute_parser(std::string config_path) {
    try {
        config_info = YAML::LoadFile(config_path);
    } catch (BadFile e) {
        std::cout << "Invalid file: " << config_path << std::endl;
        return;
    } catch (...) { std::cout << "load yaml error!" << std::endl; }
}
void cfg_attribute_parser::print_one_node(std::string attr_name) {
    //"tune_parameter"
    if (config_info[attr_name]) {
        std::cout << attr_name << " size: " << config_info[attr_name].size()
                  << std::endl;
        for (auto it = config_info[attr_name].begin();
                it != config_info[attr_name].end(); ++it) {
            std::cout << it->first.as<std::string>() << std::endl;
            range_element_attr attr = it->second.as<range_element_attr>();
            std::cout << "start: " << attr.start_value << std::endl;
            std::cout << "end: " << attr.end_value << std::endl;
            std::cout << "stride: " << attr.stride << std::endl << std::endl;
        }
    } else {
        std::cout << attr_name << " is not found in cfg yaml file!"
                  << std::endl;
    }
}
void cfg_attribute_parser::get_tune_parameters(tune_attr_vector &cfg_attr) {
    std::string attr_name = "tune_parameter";
    if (config_info[attr_name]) {
        for (auto it = config_info[attr_name].begin();
                it != config_info[attr_name].end(); ++it) {
            tune_attr_cfg attr = {it->first.as<std::string>(),
                    it->second.as<range_element_attr>()};
            cfg_attr.push_back(attr);
        }
    } else {
        std::cout << attr_name << " is not found in cfg yaml file!"
                  << std::endl;
    }
}

void cfg_attribute_parser::get_tune_parameter_names(
        std::vector<std::string> &tune_para_names) {
    std::string attr_name = "tune_parameter";
    if (config_info[attr_name]) {
        for (auto it = config_info[attr_name].begin();
                it != config_info[attr_name].end(); ++it) {
            tune_para_names.push_back(it->first.as<std::string>());
        }
    } else {
        std::cout << attr_name << " is not found! in cfg yaml file!"
                  << std::endl;
    }
}

// map<string,string>
// file_type: micro_kernel_attri_cfg
std::string cfg_attribute_parser::get_cfg_attr_by_name(std::string attr_name) {
    if (config_info[attr_name]) {
        return config_info[attr_name].as<std::string>();
    } else {
        return "";
    }
}

} // namespace tuner_ns
