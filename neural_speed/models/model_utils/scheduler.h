//  Copyright (c) 2023 Intel Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#ifndef SCHEDULER_H
#define SCHEDULER_H

#include "models/model_utils/pool.h"
#include "models/model_utils/model_utils.h"

// iteration-level worker
class Iter_level_worker {
 public:
  explicit Iter_level_worker(const gpt_params& params);
  virtual ~Iter_level_worker();
  virtual bool step(std::vector<sequence>* seqs, const int& n_input) = 0;
  virtual bool beam_search_step(std::vector<sequence>* seqs, const int& n_input) = 0;
  virtual bool greedy_search_step(std::vector<sequence>* seqs, const int& n_input) = 0;
  virtual bool top_k_top_p_sample_step(std::vector<sequence>* seqs, const int& n_input) = 0;

  inline void set_threads(const int& n_threads) { threads = n_threads; }
  inline std::vector<int> get_request_done_ids() const { return request_done_ids; }
  inline void empty_request_done_ids() { request_done_ids.clear(); }

 protected:
  virtual bool prepare_inputs(std::vector<sequence>* seqs, const int& n_input, model_input* inputs) = 0;
  virtual bool update_seqs(std::vector<sequence>* seqs, const int& n_input) = 0;

  model_context* m_ctx = nullptr;
  int threads;
  beam_search_flow* bsf = nullptr;
  std::vector<model_token> next_tokens;
  std::vector<std::vector<model_token>> last_n_tokens;
  std::vector<int> request_done_ids;
  std::unordered_map<int, int> reqidx_to_vecid;
};

// continuous batching generation worker
class Cont_batch_gen_worker : public Iter_level_worker {
 public:
  explicit Cont_batch_gen_worker(const gpt_params& params);
  Cont_batch_gen_worker(const gpt_params& params, const int& n_threads);
  ~Cont_batch_gen_worker() = default;

  bool step(std::vector<sequence>* seqs, const int& n_input) override;
  bool beam_search_step(std::vector<sequence>*, const int& n_input) override;
  bool greedy_search_step(std::vector<sequence>* seqs, const int& n_input) override;
  bool top_k_top_p_sample_step(std::vector<sequence>* seqs, const int& n_input) override;

 protected:
  bool prepare_inputs(std::vector<sequence>*, const int& n_input, model_input* inputs) override;
  bool update_seqs(std::vector<sequence>* seqs, const int& n_input) override;
};

// iteration-level scheduler
class Iter_level_scheduler {
 public:
  explicit Iter_level_scheduler(const gpt_params& params);
  Iter_level_scheduler(const gpt_params& params, const std::string& policy, const int& log_level);
  virtual ~Iter_level_scheduler() = default;

  // TODO (YZT) kv cache ptr as input params
  virtual bool add_request(sequence seq) = 0;
  virtual bool step() = 0;
  virtual bool done() = 0;
  inline bool has_finished_seq() { return (finished_pool.size() > 0); }
  std::vector<sequence> pop_completed_requests();

 protected:
  virtual bool prepare_seqs() = 0;
  virtual bool update_pools() = 0;

  const serve_policy policy;
  const gpt_params params;
  serve_pool waiting_pool;
  serve_pool running_pool;
  serve_pool finished_pool;
  int log_level;  // 0: log info and error, 1 or other: log error
};

// continuous batching generation scheduler
class Cont_batch_gen_scheduler : public Iter_level_scheduler {
 public:
  explicit Cont_batch_gen_scheduler(const gpt_params& params);
  Cont_batch_gen_scheduler(const gpt_params& params, const std::string& policy, const int& log_level);
  ~Cont_batch_gen_scheduler() = default;

  bool add_request(sequence seq) override;
  bool step() override;
  bool done() override;

 protected:
  bool prepare_seqs() override;
  bool update_pools() override;
  int query_free_req_idx();

  const int max_requests;
  Cont_batch_gen_worker wr;
  std::vector<sequence> executed_seqs;
  std::vector<bool> free_req_idx;
  int waiting_free_req_idx_seqs_num;
  int cur_running_num = -1;
  // reserve at least one position for next prompt hidden states prefilling
  // when running_pool is full (size == max_requests)
  bool steps_decoding_for_next_prefill = false;
};

#endif  // SCHEDULER_H
