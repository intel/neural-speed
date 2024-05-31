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

#include "models/model_utils/scheduler.h"

// Iter_level_worker
Iter_level_worker::Iter_level_worker(const gpt_params& params) : m_ctx(model_init_from_gpt_params(params)) {
  if (m_ctx == nullptr) {
    fprintf(stderr, "%s: error: unable to load model.\n", __func__);
    exit(0);
  }
  if (m_ctx->beam_search && bsf == nullptr) {
    bsf = new beam_search_flow(m_ctx, m_ctx->max_request_num, params.n_threads);
    fprintf(stdout, "%s: use beam search generation in model server.\n", __func__);
  } else if (m_ctx->generation_conf.do_sample == false) {
    fprintf(stdout, "%s: use greedy search generation in model server.\n", __func__);
  } else {
    fprintf(stdout, "%s: use top_k_top_p sampling generation in model server.\n", __func__);
  }
  // for repetition penalizing sampling and long context
  if (!m_ctx->beam_search) {
    last_n_tokens.resize(m_ctx->max_request_num);
    for (int i = 0; i < m_ctx->max_request_num; ++i) {
      last_n_tokens[i].resize(m_ctx->n_ctx, 0);
    }
  }
  threads = params.n_threads;
}

Iter_level_worker::~Iter_level_worker() {
  if (m_ctx != nullptr) {
    model_free(m_ctx);
  }
  if (bsf != nullptr) {
    delete bsf;
  }
}

// Cont_batch_gen_worker
Cont_batch_gen_worker::Cont_batch_gen_worker(const gpt_params& params) : Iter_level_worker(params) {
  m_ctx->cont_batching = true;
}

bool Cont_batch_gen_worker::prepare_inputs(std::vector<sequence>* seqs, const int& n_input, model_input* inputs) {
  for (int i = 0; i < n_input; ++i) {
    if (seqs->at(i).status != seq_status::PREFILL && seqs->at(i).status != seq_status::DECODING) {
      fprintf(stderr, "%s: error: request %d status is unright (%d).\n", __func__, seqs->at(i).request_idx,
              static_cast<int>(seqs->at(i).status));
      return false;
    } else if (seqs->at(i).status == seq_status::PREFILL) {
      if (seqs->at(i).n_prompt_tokens + seqs->at(i).gen_conf.max_new_tokens > m_ctx->n_ctx) {
        fprintf(stderr, "%s: error: prompt + max_new_tokens is too long (%d tokens, max %d) for model server.\n",
                __func__, seqs->at(i).n_prompt_tokens + seqs->at(i).gen_conf.max_new_tokens, m_ctx->n_ctx);
        return false;
      }
      inputs[i].tokens = seqs->at(i).prompt_ids.data();
      inputs[i].n_tokens = seqs->at(i).n_prompt_tokens;
      inputs[i].n_prompt_tokens = seqs->at(i).n_prompt_tokens;
      inputs[i].n_past = 0;
      inputs[i].n_total = 0;
      inputs[i].request_idx = seqs->at(i).request_idx;
      // do not support padding for now
      inputs[i].n_padding = 0;
      inputs[i].gen_conf = seqs->at(i).gen_conf;
    } else if (seqs->at(i).status == seq_status::DECODING) {
      inputs[i].tokens = (bsf != nullptr) ? nullptr : &(seqs->at(i).generated_ids.back());
      inputs[i].n_tokens = 1;
      inputs[i].n_past = seqs->at(i).n_past;
      inputs[i].n_total = seqs->at(i).n_total;
      inputs[i].request_idx = seqs->at(i).request_idx;
      // do not support padding for now
      inputs[i].n_padding = 0;
    } else {
      continue;
    }
    // update last_n_tokens
    if (!m_ctx->beam_search &&
        (seqs->at(i).status == seq_status::PREFILL || seqs->at(i).status == seq_status::DECODING)) {
      int req_idx = inputs[i].request_idx;
      last_n_tokens[req_idx].erase(last_n_tokens[req_idx].begin(), last_n_tokens[req_idx].begin() + inputs[i].n_tokens);
      last_n_tokens[req_idx].insert(last_n_tokens[req_idx].end(), inputs[i].tokens,
                                    inputs[i].tokens + inputs[i].n_tokens);
    }
  }
  return true;
}

bool Cont_batch_gen_worker::beam_search_step(std::vector<sequence>* seqs, const int& n_input) {
  std::vector<model_input> step_inputs(n_input);
  if (!prepare_inputs(seqs, n_input, step_inputs.data())) {
    return false;
  }
  // step beam search decoding
  if (!bsf->step(step_inputs)) {
    return false;
  }
  return true;
}

bool Cont_batch_gen_worker::greedy_search_step(std::vector<sequence>* seqs, const int& n_input) {
  // prepare inputs
  std::vector<model_input> step_inputs(n_input);
  if (!prepare_inputs(seqs, n_input, step_inputs.data())) {
    return false;
  }
  m_ctx->batch_size = n_input;
  m_ctx->request_running_bs = n_input;
  // model eval
  if (model_eval(m_ctx, step_inputs.data(), step_inputs.size(), threads) > 0) {
    return false;
  }
  // greedy search
  next_tokens = model_post_greedy_search(m_ctx->logits.data(), m_ctx);
  return true;
}

bool Cont_batch_gen_worker::top_k_top_p_sample_step(std::vector<sequence>* seqs, const int& n_input) {
  // prepare inputs
  std::vector<model_input> step_inputs(n_input);
  if (!prepare_inputs(seqs, n_input, step_inputs.data())) {
    return false;
  }
  m_ctx->batch_size = n_input;
  m_ctx->request_running_bs = n_input;
  // model eval
  if (model_eval(m_ctx, step_inputs.data(), step_inputs.size(), threads) > 0) {
    return false;
  }
  // top_k_top_p sampling
  std::vector<int> last_n_tokens_indices(n_input, 0);
  for (int ni = 0; ni < n_input; ++ni) {
    last_n_tokens_indices[ni] = seqs->at(ni).request_idx;
  }
  next_tokens = model_post_sample_top_k_top_p_repeat(m_ctx->logits.data(), m_ctx, last_n_tokens, last_n_tokens_indices);
  return true;
}

bool Cont_batch_gen_worker::step(std::vector<sequence>* seqs, const int& n_input) {
  reqidx_to_vecid.clear();
  for (int ni = 0; ni < n_input; ++ni) {
    reqidx_to_vecid.emplace(seqs->at(ni).request_idx, ni);
  }
  // beam search
  if (m_ctx->beam_search) {
    if (bsf == nullptr || !beam_search_step(seqs, n_input)) {
      return false;
    }
    // greedy search
  } else if (m_ctx->generation_conf.do_sample == false) {
    if (!greedy_search_step(seqs, n_input)) {
      return false;
    }
    // top_k_top_p sampling
  } else {
    if (!top_k_top_p_sample_step(seqs, n_input)) {
      return false;
    }
  }

  return update_seqs(seqs, n_input);
}

bool Cont_batch_gen_worker::update_seqs(std::vector<sequence>* seqs, const int& n_input) {
  empty_request_done_ids();
  for (int ni = 0; ni < n_input; ++ni) {
    if (seqs->at(ni).status == seq_status::PREFILL) {
      seqs->at(ni).status = seq_status::DECODING;
      seqs->at(ni).n_past = seqs->at(ni).n_prompt_tokens;
      seqs->at(ni).n_total = seqs->at(ni).n_prompt_tokens;
      seqs->at(ni).n_tokens = 1;
    } else if (seqs->at(ni).status == seq_status::DECODING) {
      seqs->at(ni).n_tokens = 1;
      seqs->at(ni).n_past += seqs->at(ni).n_tokens;
      seqs->at(ni).n_total += seqs->at(ni).n_tokens;
    } else {
      fprintf(stderr, "%s: error: wrong sequence status %d.\n", __func__, static_cast<int>(seqs->at(ni).status));
      return false;
    }
    if (!m_ctx->beam_search) {
      if (next_tokens.size() != n_input) {
        fprintf(stderr, "%s: error: wrong next_tokens size %d, which should be %d.\n", __func__,
                static_cast<int>(next_tokens.size()), n_input);
        return false;
      }
      seqs->at(ni).generated_ids.emplace_back(next_tokens[ni]);
    }
  }
  if (m_ctx->beam_search) {
    if (bsf == nullptr) return false;
    request_done_ids = bsf->request_done_ids();
    std::vector<std::vector<model_token>> req_done_res = bsf->request_done_reponse();
    if (request_done_ids.size() != req_done_res.size()) {
      fprintf(stderr,
              "%s: error: beam search give mis-matched size between finished request ids and generated "
              "tokens.\n",
              __func__);
      return false;
    }
    for (int r = 0; r < request_done_ids.size(); ++r) {
      const int idx = request_done_ids[r];
      if (reqidx_to_vecid.count(idx) == 0) {
        fprintf(stderr, "%s: error: done request idx: %d not in executed_seqs.\n", __func__, idx);
        return false;
      }
      const int vecid = reqidx_to_vecid[idx];
      seqs->at(vecid).generated_ids = std::move(req_done_res[r]);
      seqs->at(vecid).status = seq_status::FINISHED;
      seqs->at(vecid).end_time = model_time_us();
    }
    return true;
  } else {
    for (int ni = 0; ni < n_input; ++ni) {
      if (seqs->at(ni).status == seq_status::DECODING && !seqs->at(ni).generated_ids.empty() &&
          (seqs->at(ni).generated_ids.back() == m_ctx->vocab.eos_token_id ||
           seqs->at(ni).generated_ids.size() >= seqs->at(ni).gen_conf.max_new_tokens)) {
        seqs->at(ni).status = seq_status::FINISHED;
        seqs->at(ni).end_time = model_time_us();
        request_done_ids.emplace_back(seqs->at(ni).request_idx);
        last_n_tokens[seqs->at(ni).request_idx].resize(m_ctx->n_ctx, 0);
      }
    }
    return true;
  }
}

// Iter_level_scheduler
Iter_level_scheduler::Iter_level_scheduler(const gpt_params& params, const std::string& policy, const int& log_level)
    : params(params),
      policy(parse_serve_policy(policy)),
      waiting_pool(pool_property::WAITING),
      running_pool(pool_property::RUNNING),
      finished_pool(pool_property::FINISHED),
      log_level(log_level) {}

Iter_level_scheduler::Iter_level_scheduler(const gpt_params& params) : Iter_level_scheduler(params, "fcfs", 1) {}

std::vector<sequence> Iter_level_scheduler::pop_completed_requests() {
  std::vector<sequence> ret_seqs;
  const int length = finished_pool.size();
  if (length == 0) {
    return ret_seqs;
  }
  ret_seqs.resize(length);
  for (int l = 0; l < length; ++l) {
    if (!finished_pool.pop(&ret_seqs[l])) {
      fprintf(stderr, "%s: error: pop finished_pool %dth seq failed.\n", __func__, l);
      return std::vector<sequence>();
    }
    if (log_level == 0) {
      fprintf(stdout,
              "%s: info: tokens generation time of sequence (query_id %" PRIu64 ", request_idx: %d) is %8.2fms.\n",
              __func__, ret_seqs[l].query_id, ret_seqs[l].request_idx,
              (ret_seqs[l].end_time - ret_seqs[l].receive_time) / 1000.0);
    }
  }
  return ret_seqs;
}

// Cont_batch_gen_scheduler
Cont_batch_gen_scheduler::Cont_batch_gen_scheduler(const gpt_params& params)
    : Iter_level_scheduler(params),
      max_requests(params.max_request_num),
      wr(params),
      free_req_idx(max_requests, true),
      waiting_free_req_idx_seqs_num(0) {}

Cont_batch_gen_scheduler::Cont_batch_gen_scheduler(const gpt_params& params, const std::string& policy,
                                                   const int& log_level)
    : Iter_level_scheduler(params, policy, log_level),
      max_requests(params.max_request_num),
      wr(params),
      free_req_idx(max_requests, true),
      waiting_free_req_idx_seqs_num(0) {}

int Cont_batch_gen_scheduler::query_free_req_idx() {
  auto iter = std::find_if(free_req_idx.begin(), free_req_idx.end(), [](const bool flag) { return flag; });
  if (iter == free_req_idx.end()) {
    return -1;
  } else {
    int idx = std::distance(free_req_idx.begin(), iter);
    free_req_idx[idx] = false;
    return idx;
  }
}

bool Cont_batch_gen_scheduler::add_request(sequence seq) {
  seq.receive_time = model_time_us();
  if (seq.status != seq_status::UNKNOWN) {
    fprintf(stderr, "%s: error: seq status is not UNKNOWN, can not decide to add into which pool.\n", __func__);
    return false;
  }
  // add into waiting_pool by default
  seq.status = seq_status::WAITING;
  seq.request_idx = waiting_free_req_idx_seqs_num > 0 ? -1 : query_free_req_idx();
  if (log_level == 0) {
    fprintf(stdout, "%s: info: added seq query_id: %" PRIu64 ", request_idx: %d \n", __func__, seq.query_id,
            seq.request_idx);
  }
  if (seq.request_idx == -1) waiting_free_req_idx_seqs_num++;
  return waiting_pool.add(seq);
}

bool Cont_batch_gen_scheduler::prepare_seqs() {
  executed_seqs.clear();
  cur_running_num = running_pool.size();
  if (cur_running_num > max_requests) {
    fprintf(stderr, "%s: error: cur_running_num is larger than max_request_num.\n", __func__);
    return false;
  }
  const int n_perfill_seqs = std::min(max_requests - cur_running_num, waiting_pool.size());
  executed_seqs.resize(n_perfill_seqs + cur_running_num);
  if (log_level == 0) {
    fprintf(stdout, "%s: info: prefilling seqs num is %d, decoding seqs num is %d.\n", __func__, n_perfill_seqs,
            cur_running_num);
  }
  if (waiting_pool.size() > 0) {
    // pop prompts
    if (cur_running_num < max_requests) {
      for (int np = 0; np < n_perfill_seqs; ++np) {
        if (waiting_pool.pop(&executed_seqs[cur_running_num + np])) {
          executed_seqs[cur_running_num + np].status = seq_status::PREFILL;
          executed_seqs[cur_running_num + np].generated_ids.reserve(
              executed_seqs[cur_running_num + np].gen_conf.max_new_tokens);
          if (executed_seqs[cur_running_num + np].request_idx == -1) {
            const int fidx = query_free_req_idx();
            if (fidx == -1) {
              fprintf(stderr, "%s: error: no free position to put the request.\n", __func__);
              return false;
            }
            executed_seqs[cur_running_num + np].request_idx = fidx;
            if (log_level == 0) {
              fprintf(stdout, "%s: info: updated seq query_id: %" PRIu64 ", request_idx: %d \n", __func__,
                      executed_seqs[cur_running_num + np].query_id, executed_seqs[cur_running_num + np].request_idx);
            }
            waiting_free_req_idx_seqs_num--;
          }
        } else {
          fprintf(stderr, "%s: error: pop waiting seq failed.\n", __func__);
          return false;
        }
      }
    } else {
      // steps generation
      steps_decoding_for_next_prefill = true;
    }
  }
  // step generation
  for (int dn = 0; dn < cur_running_num; ++dn) {
    if (!running_pool.pop(&executed_seqs[dn]) || executed_seqs[dn].status != seq_status::DECODING) {
      fprintf(stderr, "%s: error: pop running_pool %dth seq failed.\n", __func__, dn);
      return false;
    }
  }
  cur_running_num = executed_seqs.size();
  return true;
}

bool Cont_batch_gen_scheduler::step() {
  int64_t s_t0 = model_time_us();
  if (done()) {
    fprintf(stderr,
            "%s: warning: scheduler has no more requests, please add extra requests or just stop "
            "calling it.\n",
            __func__);
    return true;
  }
  if (!prepare_seqs()) {
    return false;
  }
  // one step
  if (!steps_decoding_for_next_prefill) {
    if (log_level == 0) {
      fprintf(stdout, "%s: info: running_pool size < max request num, will execute one step.\n", __func__);
    }
    if (!wr.step(&executed_seqs, executed_seqs.size())) {
      return false;
    }
  } else {
    // steps for next prompt prefilling
    if (log_level == 0) {
      fprintf(stdout, "%s: info: running_pool size = max request num, will execute several steps.\n", __func__);
    }
    wr.empty_request_done_ids();
    while (wr.get_request_done_ids().empty()) {
      if (!wr.step(&executed_seqs, executed_seqs.size())) {
        return false;
      }
    }
    steps_decoding_for_next_prefill = false;
  }
  bool success = update_pools();
  if (log_level == 0) {
    fprintf(stdout, "%s: info: scheduler step time usage is %8.2fms \n", __func__, (model_time_us() - s_t0) / 1000.0f);
  }
  return success;
}

bool Cont_batch_gen_scheduler::update_pools() {
  for (int ns = 0; ns < executed_seqs.size(); ++ns) {
    if (executed_seqs[ns].status == seq_status::DECODING) {
      running_pool.add(executed_seqs[ns]);
    } else if (executed_seqs[ns].status == seq_status::FINISHED) {
      finished_pool.add(executed_seqs[ns]);
      free_req_idx[executed_seqs[ns].request_idx] = true;
      if (log_level == 0) {
        fprintf(stdout, "%s: info: seq query_id: %" PRIu64 ", request_idx: %d finished.\n", __func__,
                executed_seqs[ns].query_id, executed_seqs[ns].request_idx);
      }
    } else {
      fprintf(stderr,
              "%s: error: wrong seq status: %d of seq query_id: %" PRIu64
              ", request_idx: %d, should be in DECODING OR FINISHED.\n",
              __func__, static_cast<int>(executed_seqs[ns].status), executed_seqs[ns].query_id,
              executed_seqs[ns].request_idx);
      return false;
    }
  }
  return true;
}

bool Cont_batch_gen_scheduler::done() {
  if (waiting_pool.empty() && running_pool.empty()) {
    return true;
  } else {
    return false;
  }
}
