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
#ifndef MICRO_KERNEL_REPO_H
#define MICRO_KERNEL_REPO_H
#include <iostream>
#include <map>
#include <vector>
#include "tuner_comm.h"
namespace tuner_ns {

#define MICRO_KERNEL_REPO_ROOT_PATH "../resource_file/micro_kernel_repo/"

class micro_kernel_info_mng {
public:
    micro_kernel_info_mng();
    micro_kernel_info_mng(std::string micro_kernel_repo_location);
    void get_micro_kernel_info(
            micro_kernel_type mk_type, std::vector<micro_kernel_info> &info);
    void print_all_directory();
    void print_micro_kernel_info();

private:
    void init_all_micro_kernel_info();
    std::string micro_kernel_root_dir;
    std::map<micro_kernel_type, std::vector<micro_kernel_info>>
            all_micro_kernel_info;
};
} // namespace tuner_ns

#endif