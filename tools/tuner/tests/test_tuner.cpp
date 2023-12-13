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
#include "codegen.hpp"
#include "micro_kernel_repo.h"
#include "runner.hpp"
#include "selector.h"
#include "tuner.h"
#include "tuner_thread_pool.h"

using namespace std;
using namespace tuner_ns;

void test_cfg_attribute_parser() {
    std::string root_dir
            = "/home/huaiyuzh/00_code/tuner/"
              "libraries.performance.math.gpu-graph-compiler/tools/tuner/";
    std::string cfg_file = root_dir
            + "resource_file/micro_kernel_repo/gemm/gemm1/gemm1.yaml";
    std::shared_ptr<cfg_attribute_parser> parser(
            new cfg_attribute_parser(cfg_file));

    std::string tune_parameter = "tune_parameter";
    parser->print_one_node(tune_parameter);
#if 0
    std::map<std::string, range_element_attr> cfg_attr;
    parser->get_tune_parameters(cfg_attr);
    std::cout<<"tune parameters:"<<std::endl;
    for (auto elem : cfg_attr){
        std::cout<<elem.first<<":  "<<elem.second.start_value<<"  "<<elem.second.end_value<<std::endl;
    }
#endif
}

void test_selector() {

    std::shared_ptr<selector> sel(new selector());
    // sel->set_counter(1357911);
    // sel->print_counter();
}

void test_micro_kernel_info_mng() {
    std::shared_ptr<micro_kernel_info_mng> mk_manager(
            new micro_kernel_info_mng());
    std::vector<micro_kernel_info> mk_info;

    mk_manager->get_micro_kernel_info(MK_GEMM, mk_info);
    for (auto &elem : mk_info) {
        cout << "micro_kernel_name: " << elem.micro_kernel_name << endl;
        cout << "attribute_file_name: " << elem.attribute_file_name << endl;
        cout << "micro_kernel_path: " << elem.micro_kernel_path << endl;
    }

    mk_manager->get_micro_kernel_info(MK_GEMM, mk_info);
    cout << "MK_GEMM -----" << std::endl;
    for (auto &elem : mk_info) {
        cout << "micro_kernel_name: " << elem.micro_kernel_name << endl;
        cout << "attribute_file_name: " << elem.attribute_file_name << endl;
        cout << "micro_kernel_path: " << elem.micro_kernel_path << endl;
    }

    mk_manager->get_micro_kernel_info(MK_MHA, mk_info);
    cout << "MK_MHA -----" << std::endl;
    for (auto &elem : mk_info) {
        cout << "micro_kernel_name: " << elem.micro_kernel_name << endl;
        cout << "attribute_file_name: " << elem.attribute_file_name << endl;
        cout << "micro_kernel_path: " << elem.micro_kernel_path << endl;
    }
}
void test_cmd_parser() {
#if 0
    int argc = 9;
    const char *argv[] = {
            "./tuner",
            //"--operation=GEMM",
            "--operation=MHA",
            //"--operation=GEMM,MHA",
            "--B=64",
            "--N=16",
            "--F=384",
            "--T=384",
            "--H=64",
            "--data-type=bf16",
            "--layout=row",
    };
#endif
    int argc = 8;
    const char *argv[] = {
            "./tuner",
            //"--operation=GEMM",
            //"--operation=MHA",
            "--operation=GEMM,MHA",
            "--m=4096",
            "--n=4096",
            "--k=2048",
            "--A-shape=bf16:row",
            "--B-shape=float:row",
            "--C-shape=f32:column",
    };
    std::shared_ptr<cmdline_parser> cmd_parser(new cmdline_parser());
    if (false == cmd_parser->init_cmdline_parser(argc, argv)) { return; }
    cmd_parser->print_user_cmd_info();
}

template <typename T>
vector<vector<T>> cartiain(vector<vector<T>> &v) {
    auto product = [](long long a, vector<T> &b) { return a * b.size(); };
    const long long N = accumulate(v.begin(), v.end(), 1LL, product);
    vector<T> u(v.size());
    vector<vector<T>> res;
    for (long long n = 0; n < N; ++n) {
        lldiv_t q {n, 0};
        for (long long i = v.size() - 1; 0 <= i; --i) {
            q = div(q.quot, v[i].size());
            u[i] = v[i][q.rem];
            cout << "i:" << i << "q.rem: " << q.rem << endl;
            for (T x : u)
                cout << x << ' ';
            cout << '\n';
        }
        // Do what you want here with u.
        res.push_back(u);
        cout << "\ncomplete one loop n:" << n << endl;
        for (T x : u)
            cout << x << ' ';
        cout << '\n';
    }

    return res;
}
void test_cart() {
    vector<vector<int>> input = {{11, 12, 13}, {21, 22, 23}, {31, 32, 33}};
    auto res = cartiain(input);
    for (auto aa : res) {
        for (auto bb : aa) {
            cout << bb << ",";
        }
        cout << endl;
    }
}

void test_tuner_gemm() {
    std::shared_ptr<tuner> cur_tuner(new tuner());
    int argc = 9;
    const char *argv[] = {
            "./tuner",
            //"--operation=GEMM",
            //"--operation=MHA",
            "--operation=GEMM",
            "--m=4096",
            "--n=4096",
            "--k=1024",
            "--A-shape=bf16:row",
            "--B-shape=bf16:row",
            "--C-shape=f32:row",
            "--global_kslicing=1",
    };

    cur_tuner->on_receive_cmdline(argc, argv);
    // cur_tuner->print_cmd_info();
}

void test_tuner_mha() {
    //--operation=MHA  --B=64 ,--N=16 ,--F=384 ,--T=384 ,--H=64 ,
    //--data-type=bf16
    std::shared_ptr<tuner> cur_tuner(new tuner());
    int argc = 9;
    const char *argv[] = {"./tuner",
            //"--operation=GEMM",
            "--operation=MHA",
            //"--operation=GEMM,MHA",
            "--B=64", "--N=16", "--F=384", "--T=384", "--H=64",
            "--data-type=bf16", "--layout=row"};

    cur_tuner->on_receive_cmdline(argc, argv);
}
void test_self_define_kernel() {
    //--operation==SELF_DEFINE_KERNEL --cfg-para-list=key1:value1,key:value
    //--shape-list==bf16:row:output:size, bf16:row:output:size
    //--micro-kernel-path=micro-kernel.hpp –ops=
    std::shared_ptr<tuner> cur_tuner(new tuner());
    int argc = 5;
    const char *argv[] = {"./tuner",
            //"--operation=GEMM",
            "--operation=SELF_DEFINE_KERNEL",
            //"--operation=GEMM,MHA",
            "--cfg-para-list=m:4096,n:4096,k:2048,batch_num:32",
            //"--shape-list=bf16:row:input:268435456,bf16:row:input:268435456," "f32:row:output:16777216",
            "--shape-list=bf16:row:input:4096*2048*32,bf16:row:input:1024*4096*"
            "batch_num+1024*4096*32,"
            "f32:row:output:m*n",

            "--ops=2*m*n*k*batch_num+m*n*batch_num"};

    cur_tuner->on_receive_cmdline(argc, argv);
}

void print_cmd_para(int argc, const char **argv) {
    cout << "argc in main:" << argc << endl;
    for (auto i = 1; i < argc; ++i) {
        cout << "cmd in main:" << argv[i] << endl;
    }
}

string thread_task(int index, int j, vector<string> &a) {
    // std::this_thread::sleep_for(std::chrono::seconds(1));
    a[index] = "success get result " + std::to_string(index * j);
    return ("success get result " + std::to_string(index));
    // return index*index;
}

typedef struct input_test {
    uint32_t a;
    uint32_t b;
} input_test;

std::vector<input_test> input_vec;
std::map<string, string> map_info;
string thread_task_stl_input(input_test &inputa,
        std::vector<input_test> input_vec, std::map<string, string> map_info) {
    uint32_t index = 0;
    // std::this_thread::sleep_for(std::chrono::seconds(1));

    return ("success get result " + std::to_string(index));
    // return index*index;
}

void test_thread_pool() {
    tuner_thread_pool pool(4);
    std::vector<std::future<string>> results;
    input_test input {1, 2};
    input_vec.emplace_back(input);
    map_info["aa"] = "bb";
    vector<string> a;
    for (int i = 0; i < 8; ++i) {
        a.emplace_back(to_string(i));
    }
    for (int i = 0; i < 8; ++i) {
        results.emplace_back(pool.add_task(thread_task, i, i, a)
                // pool.add_task(thread_task_stl_input, input,input_vec,map_info)
        );
    }
    int i = 0;
    for (auto &&result : results) {
        result.get();
        std::cout << "a[" << i << "] is:" << a[i++] << std::endl;
        std::cout << std::endl;
    }
}

int main() {
    // test_micro_kernel_info_mng();
    // test_selector();
    // test_cmd_parser();
    // test_search_alg();
    // test_cfg_attribute_parser();
    // test_cart();
    // test_tuner_gemm();
    // test_tuner_mha();
    // test_time_out();
    //  test_self_define_kernel();
    // test_thread_pool();
    // print_cmd_para(argc,argv);
    // test_MK_GEMM();
    // test_MK_BRGEMM_GELU_STORE();
    return 0;
}
