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
#ifndef TUNER_LOG_H
#define TUNER_LOG_H

#include <fstream>
#include <iostream>
#include "tuner_comm.h"

namespace tuner_ns {
class tuner_log {
private:
    /// Operation file name
    std::string output_file_name;
    /// Output file containing results
    std::ofstream output_file;

public:
    tuner_log(std::string &log_file_name);
    ~tuner_log();
    void write_log(std::string const &log_info);
    void flush() { output_file.flush(); };
    string get_log_file_name() { return output_file_name; };
    void sort_log(const string &colum_name);
};
} // namespace tuner_ns
#endif