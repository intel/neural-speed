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
#ifndef TUNER_LOG_FACTORY_H
#define TUNER_LOG_FACTORY_H

#include "../tuner_log.h"
#include "tuner_comm.h"

namespace tuner_ns {
class tuner_log_factory {
public:
    tuner_log_factory(const tuner_log_factory &) = delete;
    tuner_log_factory &operator=(const tuner_log_factory &) = delete;
    static tuner_log_factory &get_instance();
    void add_log_elem(tuner_log_type log_type, shared_ptr<tuner_log> log_ptr);
    shared_ptr<tuner_log> get_log(tuner_log_type log_type);
    string get_cur_time(const string &delimiter = "-");

private:
    tuner_log_factory();
    ~tuner_log_factory();

private:
    std::map<tuner_log_type, shared_ptr<tuner_log>> log_map;
};

} // namespace tuner_ns
#endif