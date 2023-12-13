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
#include <map>
#include <numeric>
#include <thread>
#include <unistd.h>
#include "cfg_attribute_parser.h"
#include "cmdline_parser.h"
#include "micro_kernel_repo.h"
#include "selector.h"
#include "tuner.h"
#include "tuner_thread_pool.h"
using namespace tuner_ns;
using namespace std;

void run_tuner(int argc, const char **argv) {
    std::shared_ptr<tuner> cur_tuner(new tuner());
    cur_tuner->on_receive_cmdline(argc, argv);
}
int main(int argc, const char **argv) {
    run_tuner(argc, argv);
}