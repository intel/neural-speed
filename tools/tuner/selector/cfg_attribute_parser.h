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
#ifndef CFG_ATTRIBUTE_PARSER_H
#define CFG_ATTRIBUTE_PARSER_H
#include <iostream>
#include <string>
#include "tuner_comm.h"

#include <yaml-cpp/node/parse.h>
#include <yaml-cpp/yaml.h>

namespace YAML {
template <>
struct convert<tuner_ns::range_element_attr> {
    static Node encode(const tuner_ns::range_element_attr &rhs) {
        Node node;
        node.push_back(rhs.start_value);
        node.push_back(rhs.end_value);
        node.push_back(rhs.stride);
        return node;
    }

    static bool decode(const Node &node, tuner_ns::range_element_attr &rhs) {
        rhs.start_value = node["start_value"].as<int32_t>();
        rhs.end_value = node["end_value"].as<int32_t>();
        rhs.stride = node["stride"].as<int32_t>();
        return true;
    }
};
} // namespace YAML

namespace tuner_ns {

class cfg_attribute_parser {
public:
    cfg_attribute_parser(std::string config_path);
    void get_tune_parameters(tune_attr_vector &cfg_attr);
    void print_one_node(std::string attr_name);
    std::string get_cfg_attr_by_name(std::string attr_name);
    void get_tune_parameter_names(std::vector<std::string> &tune_para_names);

private:
    YAML::Node config_info;
};
} // namespace tuner_ns

#endif
