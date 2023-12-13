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
#include "tuner_log.h"
#include "tuner_comm.h"

using namespace std;

namespace tuner_ns {
tuner_log::tuner_log(std::string &log_file_name)
    : output_file_name(log_file_name) {
    output_file_name = log_file_name;
    output_file = ofstream(output_file_name);
}
tuner_log::~tuner_log() {
    if (output_file.is_open()) { output_file.close(); }
}
void tuner_log::write_log(std::string const &log_info) {
    output_file << log_info;
}
void tuner_log::sort_log(const string &colum_name) {
    output_file.close();

    ifstream in(output_file_name);
    string title_line = "";
    getline(in, title_line);
    in.close();
    if (title_line == "") {
        std::cout << "log header is incorrect!\n";
        return;
    }

    if (title_line.find(colum_name) == title_line.npos) {
        std::cout << "sorted colum is not found!\n";
        return;
    }
    string delimiter = ",";
    string strs = title_line;
    size_t pos = strs.find(delimiter);
    int i = 0;
    while (pos != strs.npos) {
        ++i;
        string temp = strs.substr(0, pos);
        if (temp == colum_name) { break; };

        strs = strs.substr(pos + 1);
        pos = strs.find(delimiter);
    }

    auto cmd = "sort -n -k " + to_string(i) + " -t \",\" " + output_file_name
            + " > new_" + output_file_name + "; mv new_" + output_file_name
            + " " + output_file_name;
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
}

} // namespace tuner_ns
