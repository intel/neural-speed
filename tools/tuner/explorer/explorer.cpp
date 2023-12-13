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
#include "explorer.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <time.h>
#include <unistd.h>
#include "tune_in_existed_result.h"
#include "tune_in_recommendation.h"

namespace tuner_ns {

build_kernel_perf_test_result build_one_kernel_perf_test_wrapper(
        shared_ptr<kernel_generator> kernel_gen,
        shared_ptr<runner> kernel_runner, cmdline_input_info &cmd_info,
        const vector<mk_info> &mk_sequence, const smap_t &tuning_params) {
    string out_path {""};
    build_kernel_perf_test_result build_res;
    build_res.build_succ = false;
    if (kernel_gen->gen_kernel(out_path, mk_sequence,
                cmd_info.usr_cfg_kernel_attr, tuning_params)) {
        if (kernel_runner->build_kernel_perf_test(out_path,
                    cmd_info.usr_cfg_kernel_attr, build_res.build_res)) {
            build_res.build_succ = true;
        }
    }

    build_res.tuning_params = tuning_params;
    return build_res;
}

run_kernel_result_in_thread run_one_kernel_perf_test_wrapper(
        shared_ptr<runner> kernel_runner, kernel_perf_test_info &perf_test_info,
        uint32_t src_pos) {
    run_kernel_result_in_thread res;
    auto run_succ = kernel_runner->run_kernel_perf_test(
            perf_test_info, res.run_res.perf_data);
    res.run_res.run_succ = run_succ;
    res.src_pos = src_pos;

    return res;
}

explorer::explorer() {
    // std::string env = "ZE_AFFINITY_MASK";
    // auto tile_num = get_env(env.c_str());
    // if (tile_num != "") {
    //     run_kernel_tile_env = "export ZE_AFFINITY_MASK=" + tile_num + ";";
    // }
}

explorer::~explorer() {}

string explorer::get_cur_time() {
    return tuner_log_factory::get_instance().get_cur_time();
}

bool explorer::tune_kernel_one_by_one(cmdline_input_info &cmd_info,
        vector<vector<string>> &all_op_cart_proc,
        mk_name_to_mk_attr_map &mk_attr_map,
        map<string, vector<vector<string>>> &mk_tune_para_set) {
    runner_record best_kernel_perf {"", 0, 0xFFFFFFFF, 0.0};

    tuner_thread_pool pool(tune_thread_pool_size);
    std::vector<std::future<build_kernel_perf_test_result>> tune_build_results;

    tuner_thread_pool run_pool(max_run_kernel_num);
    std::vector<std::future<run_kernel_result_in_thread>> run_results;
    int cur_run_thread_num = 0;
    std::tm *tm;
    for (auto one_op_fuse_elem : all_op_cart_proc) {

        tune_dbg_rec->write_log("---Start one round tune kernel: "
                + get_cur_time() + "---\n" + "micro_kernel:");
        for (auto mk_name : one_op_fuse_elem) {
            tune_dbg_rec->write_log(mk_name + ", ");
        }
        tune_dbg_rec->write_log("\n");
        tune_dbg_rec->write_log(
                "build thread num size:" + to_string(tune_thread_pool_size));
        tune_dbg_rec->write_log(", run kernel thread num size:"
                + to_string(max_run_kernel_num));
        tune_dbg_rec->write_log("\n");
        vector<vector<string>> cur_tune_para_set;
        vector<mk_info> mk_sequence;
        vector<vector<string>> tune_para_set;
        vector<string> tune_para_name_list;
        // auto s0 = clock();
        for (auto mk_name : one_op_fuse_elem) {
            auto cur_mk_para = mk_tune_para_set[mk_name];
            cur_tune_para_set.insert(cur_tune_para_set.end(),
                    cur_mk_para.begin(), cur_mk_para.end());
            mk_info cur_mk_info {mk_attr_map[mk_name].micro_kernel_desc.mk_type,
                    mk_attr_map[mk_name].micro_kernel_desc.micro_kernel_path
                            + "/" + mk_name};
            mk_sequence.push_back(cur_mk_info);

            for (auto para_name : mk_attr_map[mk_name].tune_attr) {
                tune_para_name_list.push_back(para_name.attr_name);
            }
        }

        /// auto s1 = clock();
        cartesian(cur_tune_para_set, tune_para_set);
        kernel_gen->gen_kernel_prepare();
        smap_t input_tune_para_set;
        for (auto one_piece_of_tune_para : tune_para_set) {
            // tune_para_rec<<"tune para: ";
            for (auto i = 0; i < one_piece_of_tune_para.size(); ++i) {
                // tune_para_rec<<tune_para_name_list[i]<<":"<<one_piece_of_tune_para[i]<<",";
                input_tune_para_set[tune_para_name_list[i]]
                        = one_piece_of_tune_para[i];
            }
            if (false == check_is_valid_cfg(cmd_info, input_tune_para_set)) {
                continue;
            }
            if (cur_build_thread_num >= tune_thread_pool_size) {
                parse_thread_pool_exec_result(tune_build_results,
                        best_kernel_perf, run_pool, cur_run_thread_num,
                        run_results);
                cur_build_thread_num = 0;
                tune_build_results.clear();
                tune_output_rec->flush();
            }
            tune_build_results.emplace_back(pool.add_task(
                    build_one_kernel_perf_test_wrapper, kernel_gen,
                    kernel_runner, cmd_info, mk_sequence, input_tune_para_set));
            ++cur_build_thread_num;
        }
        if (cur_build_thread_num > 0) {
            parse_thread_pool_exec_result(tune_build_results, best_kernel_perf,
                    run_pool, cur_run_thread_num, run_results);
            cur_build_thread_num = 0;
            tune_build_results.clear();
        }

        tune_dbg_rec->write_log(
                "---End one round tune kernel: " + get_cur_time() + "\n");
        tune_dbg_rec->flush();
    }
    find_optimal_kernel(cmd_info);
    return true;
}

void explorer::parse_thread_pool_exec_result(
        std::vector<std::future<build_kernel_perf_test_result>> &build_results,
        runner_record &best_kernel_perf, tuner_thread_pool &runner_pool,
        int &cur_run_thread_num,
        std::vector<std::future<run_kernel_result_in_thread>> &run_results) {
    vector<build_kernel_perf_test_result> final_build_res;
    // std::map<uint32_t,string> gpu_card={{0,"0.0"},{1,"0.1",},{2,"1.0"},{3,"1.1"}};
    for (auto &&cur_res : build_results) {
        final_build_res.push_back(cur_res.get());
        auto &one_build_result = final_build_res[final_build_res.size() - 1];
        // one_build_result.build_res.run_env = "export ZE_AFFINITY_MASK="+ gpu_card.at(cur_run_thread_num % max_run_kernel_num);
        one_build_result.build_res.run_env = "";
        // = this->run_kernel_tile_env; //"export ZE_AFFINITY_MASK=1.0;";

        if (one_build_result.build_succ) {
            if (std::filesystem::exists(
                        one_build_result.build_res.test_bin_full_path)) {
                if (cur_run_thread_num >= max_run_kernel_num) {
                    for (auto &&cur_run_res : run_results) {
                        auto one_run_res = cur_run_res.get();
                        final_build_res[one_run_res.src_pos].run_res
                                = one_run_res.run_res;
                    }
                    cur_run_thread_num = 0;
                    run_results.clear();
                }
                run_results.emplace_back(
                        runner_pool.add_task(run_one_kernel_perf_test_wrapper,
                                kernel_runner, one_build_result.build_res,
                                final_build_res.size() - 1));
                ++cur_run_thread_num;
            }
        }
    }
    if (cur_run_thread_num > 0) {
        for (auto &&cur_run_res : run_results) {
            auto one_run_res = cur_run_res.get();
            final_build_res[one_run_res.src_pos].run_res = one_run_res.run_res;
        }
        cur_run_thread_num = 0;
        run_results.clear();
    }

    for (auto &&cur_res : final_build_res) {
        bool find_new = false;
        bool need_save_kernel = false;
        string exec_reslut = "NULL";
        string exec_time_str = to_string(0xFFFFFFFF);
        string flops_str = to_string(0xFFFFFFFF);
        if (cur_res.build_succ) {
            if (std::filesystem::exists(cur_res.build_res.test_bin_full_path)) {
                if (cur_res.run_res.run_succ) {
                    auto perf_data = cur_res.run_res.perf_data;
                    if (perf_data.accuracy) {
                        if (perf_data.kernel_time
                                < best_kernel_perf.kernel_time) {
                            best_kernel_perf = perf_data;
                            find_new = true;
                        }
                        exec_reslut = "Success";
                        exec_time_str = to_string(perf_data.kernel_time);
                        flops_str = to_string(perf_data.kernel_flops);
                        need_save_kernel = find_new
                                || ((better_kernel_map.size()
                                        < better_kernel_num));
                        if (need_save_kernel) {
                            save_better_kernel_info(cur_res, perf_data);
                        }

                    } else {
                        exec_reslut = "Accuracy fail";
                    }
                } else {
                    exec_reslut = "Run perf test fail";
                }
            } else {
                exec_reslut = "Run fail. kernle is not exist";
            }

        } else {
            exec_reslut = "Build kernel fail";
        }

        string tune_cfg = "";
        string tune_para = "Tuning para: ";
        for (auto cur_tune_para_name : get_tune_para_name_list()) {
            tune_cfg.append(cur_res.tuning_params[cur_tune_para_name] + ", ");
            tune_para.append(cur_tune_para_name + ": "
                    + cur_res.tuning_params[cur_tune_para_name] + ", ");
        }
        tune_cfg = tune_cfg.substr(0, tune_cfg.size() - 2);
        tune_para = tune_para.substr(0, tune_para.size() - 2);
        tune_output_rec->write_log(tune_cfg + "," + exec_time_str + ","
                + flops_str + "," + exec_reslut + "\n");
        if (find_new) {
            tune_dbg_rec->write_log("Find better kernel. " + tune_para
                    + ". Perf: " + to_string(best_kernel_perf.kernel_time)
                    + ". Path: " + best_kernel_perf.kernel_path + ".\n");
        }
        if (!need_save_kernel) {
            kernel_gen->clean_generated_info(cur_res.build_res.kernel_path);
        }
    }
}
bool explorer::fast_tune(cmdline_input_info &cmd_info,
        vector<vector<string>> &all_op_cart_proc,
        mk_name_to_mk_attr_map &mk_attr_map,
        vector<vector<one_micro_kernel_tune_info>> &tune_mk_set) {
    if (tune_mk_set.size() > 1) {
        std::cout << "Not support fast tune in fused kernel" << std::endl;
        tune_result_rec->write_log("Not support fast tune in fused kernel\n");
        return false;
    }

    auto mk = tune_mk_set.at(0);
    if (mk.at(0).micro_kernel_desc.mk_type != MK_GEMM) {
        std::cout << "Only GEMM supports fast tune in fused kernel"
                  << std::endl;
        tune_result_rec->write_log(
                "Only GEMM supports fast tune in fused kernel\n");
        return false;
    }

    tune_strategy *cur_strategy = nullptr;
    tune_in_cache_result tune_in_cache(this);
    cur_strategy = &tune_in_cache;

    tune_result_rec->write_log(
            "======start tune in existed tune result, now is " + get_cur_time()
            + "======\n");
    auto tune_res
            = cur_strategy->run_tuning(cmd_info, all_op_cart_proc, mk_attr_map);
    if (tune_res.tune_succ) {
        tune_result_rec->write_log(
                "find optimal cfg in existed tune result. tune result is:\n"
                + tune_res.tune_cfg + " kernel run time: "
                + to_string(tune_res.perf_data.kernel_time) + "\n");
    } else {
        tune_result_rec->write_log(
                "Not find optimal cfg in existed tune result.\n");
    }
    tune_result_rec->write_log("======end tune in existed tune result, now is "
            + get_cur_time() + "======\n\n");
    tune_result_rec->flush();
    if (tune_res.tune_succ) {
        cout << "======================================================\n";
        cout << "-----Top optimal kernel found in existed tune result:-----\n";
        cout << tune_res.tune_cfg
             << " kernel run time: " << tune_res.perf_data.kernel_time << "\n";
        cout << "======================================================\n";
        return true;
    }

    tune_result_rec->write_log(
            "======start tune in recommend tune result, now is "
            + get_cur_time() + "======\n");
    tune_in_recommendation_cfg tune_in_recom(this);
    cur_strategy = &tune_in_recom;
    auto tune_res_in_recom
            = cur_strategy->run_tuning(cmd_info, all_op_cart_proc, mk_attr_map);
    if (tune_res_in_recom.tune_succ) {
        tune_result_rec->write_log(
                "find better configuration in recommendation tune "
                "configuration. tune result is:"
                + tune_res_in_recom.tune_cfg + ", kernel run time: "
                + to_string(tune_res_in_recom.perf_data.kernel_time) + "\n");
    } else {
        tune_result_rec->write_log(
                "Not find better configuration in reommendation tune "
                "configuration.\n");
    }
    tune_result_rec->write_log(
            "======end tune in recommned tune result, now is " + get_cur_time()
            + "======\n\n");

    tune_result_rec->flush();
    return false;
}
void explorer::tune_kernel(cmdline_input_info &cmd_info,
        vector<vector<one_micro_kernel_tune_info>> &tune_mk_set) {
    kernel_gen->set_tune_para_name_list(tunning_para_name_list);
    kernel_gen->set_tune_para_code_gen_info(code_gen_info);
    // Assume micro-kernel name is unique
    map<string, vector<vector<string>>>
            mk_tune_para_set; //{(G0,{{p1}, {p2}......})}
    vector<vector<string>> op_set; //{{G0,G1},{MHA}}
    mk_name_to_mk_attr_map mk_attr_map; //{(mk_name, tune info)}
    for (auto op : tune_mk_set) {
        vector<string> all_mk_in_one_op;
        for (auto mk : op) {
            all_mk_in_one_op.push_back(mk.micro_kernel_desc.micro_kernel_name);
            mk_attr_map[mk.micro_kernel_desc.micro_kernel_name] = mk;
            vector<vector<string>>
                    mk_para_set; //= {{mk.micro_kernel_desc.micro_kernel_name}};
            for (auto tune_para : mk.tune_attr) {
                vector<string> para = {};
                range_to_vector(tune_para, para);
                if (0 == para.size()) {
                    std::cout << "\n size of " << tune_para.attr_name
                              << " is 0\n"
                              << std::endl;
                }
                mk_para_set.push_back(para);
            }
            // todo: may need to record every parameter name.
            mk_tune_para_set[mk.micro_kernel_desc.micro_kernel_name]
                    = mk_para_set;
        }
        op_set.push_back(all_mk_in_one_op); //{{G0,G1},{MHA}}
    }

    vector<vector<string>> all_op_cart_proc;
    cartesian(op_set, all_op_cart_proc);

    // print_cart_proc(all_op_cart_proc);
    if (fast_tune(cmd_info, all_op_cart_proc, mk_attr_map, tune_mk_set)) {
        return;
    }

    tune_result_rec->write_log("======start tune in whole search space, now is "
            + get_cur_time() + "======\n");

    tune_kernel_one_by_one(
            cmd_info, all_op_cart_proc, mk_attr_map, mk_tune_para_set);

    tune_result_rec->write_log("======end tune in whole search space, now is "
            + get_cur_time() + "======\n");
}

void explorer::print_cart_proc(vector<vector<string>> cart_res) {
    for (auto elem : cart_res) {
        cout << endl;
        for (auto st : elem) {
            cout << st << ", ";
        }
    }
    cout << endl;
    cout << "out put cart size:" << cart_res.size() << endl;
}
void explorer::range_to_vector(tune_attr_cfg &tune_cfg, vector<string> &res) {
    range_element_attr &range_attr = tune_cfg.attr;
    std::string &attr_name = tune_cfg.attr_name;
    for (auto i = range_attr.start_value; i <= range_attr.end_value;
            i += range_attr.stride) {
        // if ("local_kslicing" == attr_name)
        {
            if ((i & (i - 1)) != 0) { continue; }
        }
        res.push_back(to_string(i));
    }
}
template <typename T>
void explorer::cartesian(vector<vector<T>> &v, vector<vector<T>> &res) {
    auto product = [](long long a, vector<T> &b) { return a * b.size(); };
    const long long N = accumulate(v.begin(), v.end(), 1LL, product);
    vector<T> u(v.size());
    for (long long n = 0; n < N; ++n) {
        lldiv_t q {n, 0};
        for (long long i = v.size() - 1; 0 <= i; --i) {
            q = div(q.quot, v[i].size());
            u[i] = v[i][q.rem];
        }
        res.push_back(u);
    }
}

void explorer::save_better_kernel_info(
        build_kernel_perf_test_result &one_build_result,
        runner_record &perf_data) {
    if (better_kernel_map.size() >= better_kernel_num) {
        auto max_record = max_element(better_kernel_map.begin(),
                better_kernel_map.end(),
                [](const auto &l, const auto &r) { return l.first < r.first; });

        if (perf_data.kernel_time >= max_record->first) {
            std::cout << "No need to save kernel: " << perf_data.kernel_path
                      << ", kernel_time: " << perf_data.kernel_time
                      << std::endl;
            return;
        }

        std::cout << "Removed kernel info: kernel_path: "
                  << max_record->second.build_res.kernel_path
                  << ", kernel_time: " << max_record->first
                  << ". New kernel_time: " << perf_data.kernel_time
                  << std::endl;
        kernel_gen->clean_generated_info(
                max_record->second.build_res.kernel_path);
        better_kernel_map.erase(max_record->first);
    }
    better_kernel_map[perf_data.kernel_time] = one_build_result;
}

void explorer::find_optimal_kernel(cmdline_input_info &cmd_info) {
    vector<double> top_kernel;
    for (auto it = better_kernel_map.begin(); (it != better_kernel_map.end())
            && (top_kernel.size() < max_validate_kernel_num);
            ++it) {
        runner_record perf_data;
        auto run_res = kernel_runner->run_kernel_perf_test(
                it->second.build_res, perf_data);
        if ((run_res) && (perf_data.accuracy)) {
            top_kernel.push_back(it->first);
        }
    }

    cout << "\n\n\n";
    cout << "======================================================\n";
    tune_result_rec->write_log("-----Top kernel information:-----\n");
    cout << "-----Top kernel information:-----\n";
    for (auto cur_time : top_kernel) {
        cout << "-----kernnel time:" << cur_time << endl;
        cout << "-----kernnel path:"
             << better_kernel_map[cur_time].build_res.kernel_path << endl;

        string tune_cfg = "";
        for (auto para_name : tunning_para_name_list) {
            tune_cfg += para_name + ": "
                    + better_kernel_map[cur_time].tuning_params.at(para_name)
                    + ", ";
        }
        cout << "-----configuration: "
             << tune_cfg.substr(0, (tune_cfg.length() - 2)) << std::endl;
        cout << "\n\n";
        tune_result_rec->write_log(tune_cfg + " exec_result: " + "Success"
                + ", exec_time: " + to_string(cur_time) + ", kernel path: "
                + better_kernel_map[cur_time].build_res.test_bin_full_path
                + "\n");
    }

    if (top_kernel.size() == 0) {
        cout << "No optimal kernel is found!\n";
        tune_result_rec->write_log("No optimal kernel is found!\n");
    }

    cout << "======================================================\n";
}

bool explorer::check_is_valid_cfg(
        cmdline_input_info &cmd_info, smap_t &input_tune_para_set) {

    uint32_t sg_m = stoi(input_tune_para_set.at("sg_m"));
    uint32_t sg_n = stoi(input_tune_para_set.at("sg_n"));
    // sgm x sgn <2048
    if (long(sg_m) * sg_n > 2048) {
        // std::cout<<"sg_m:"<<sg_m<<",sg_n:"<<sg_n<<",line:"<<__LINE__<<std::endl;
        return false;
    }

    uint32_t local_kslicing = 1;
    if (input_tune_para_set.count("local_kslicing") != 0) {
        local_kslicing = stoi(input_tune_para_set.at("local_kslicing"));
    }
    uint32_t cur_value = local_kslicing;
    if ((cur_value & (cur_value - 1)) != 0) {
        // std::cout<<"local_kslicing:"<<local_kslicing<<",line:"<<__LINE__<<std::endl;
        return false;
    }
    uint32_t wg_m = stoi(input_tune_para_set.at("wg_m"));
    uint32_t wg_n = stoi(input_tune_para_set.at("wg_n"));

    // (wgm/sgm) *(wgn/sgn) * local_kslicing_list <=32
    if (((wg_m / sg_m) * (wg_n / sg_n) * local_kslicing) > 32) {
        // if (((wg_m / sg_m) * (wg_n / sg_n) * local_kslicing) != 32) {
        // std::cout<<"wg_m:"<<wg_m<<",wg_n:"<<wg_n<<",line:"<<__LINE__<<std::endl;
        // std::cout<<"sg_m:"<<sg_m<<",sg_n:"<<sg_n<<",line:"<<__LINE__<<std::endl;
        // std::cout<<"local_kslicing:"<<local_kslicing<<",line:"<<__LINE__<<std::endl;
        return false;
    }

    uint32_t sg_k = stoi(input_tune_para_set.at("sg_k"));
    std::map<string, uint32_t> data_type_size = {{"bf16", 2}, {"fp16", 2},
            {"float", 4}, {"tf32", 4}, {"int8", 1}, {"int4x2", 1}};
    std::string data_type_b
            = cmd_info.usr_cfg_kernel_attr.kernel_attr.at("data_type_b");
    // if (((sg_k * data_type_size.at(data_type_b))) % 32 != 0) {
    //     // std::cout<<"sg_k:"<<sg_k<<",data_type_a:"<<data_type_a<<",line:"<<__LINE__<<std::endl;
    //     return false;
    // }

    uint32_t global_kslicing = 1;
    if (input_tune_para_set.count("global_kslicing") != 0) {
        global_kslicing = stoi(input_tune_para_set.at("global_kslicing"));
    }
    cur_value = global_kslicing;
    if ((cur_value & (cur_value - 1)) != 0) { return false; }

    return true;

    // for (auto para :tuning_params) {
    //     auto cur_value = stoi(para);
    //     if ((cur_value & (cur_value - 1)) != 0) {
    //         return false;
    //     }
    // }
}

void explorer::set_tune_para_name_list(std::vector<string> &tuning_names) {
    tunning_para_name_list.assign(tuning_names.begin(), tuning_names.end());
}

void explorer::set_tune_para_code_gen_info(
        std::map<string, code_gen_info_type> &code_gen) {
    code_gen_info = code_gen;
}
} // namespace tuner_ns
