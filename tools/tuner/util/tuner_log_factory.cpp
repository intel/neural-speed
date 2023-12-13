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
#include "tuner_log_factory.h"
#include <chrono>
#include <ctime>
#include <iomanip>

namespace tuner_ns {

tuner_log_factory &tuner_log_factory::get_instance() {
    static tuner_log_factory instance;
    return instance;
}
void tuner_log_factory::add_log_elem(
        tuner_log_type log_type, shared_ptr<tuner_log> log_ptr) {
    log_map[log_type] = log_ptr;
}
shared_ptr<tuner_log> tuner_log_factory::get_log(tuner_log_type log_type) {
    if (log_map.find(log_type) != log_map.end()) { return log_map[log_type]; }
    return nullptr;
}
tuner_log_factory::tuner_log_factory() {}
tuner_log_factory::~tuner_log_factory() {
    for (auto [log_type, log] : log_map) {
        log->flush();
    }
}

string tuner_log_factory::get_cur_time(const string &delimiter) {
    std::time_t t = std::time(NULL);
    std::tm *tm = std::localtime(&t);
    stringstream sstream;
    string time_format = "%Y" + delimiter + "%m" + delimiter + "%d" + delimiter
            + "%H" + delimiter + "%M" + delimiter + "%S";
    sstream << std::put_time(tm, time_format.c_str());

    return sstream.str();
}

} // namespace tuner_ns
