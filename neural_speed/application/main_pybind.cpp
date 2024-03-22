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
// Defines sigaction on msys:
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <stdlib.h>

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <random>
#include <string>
#include <thread>  // NOLINT
#include <unordered_map>
#include <utility>
#include <vector>

#include "common.h"
#include "core/layers/bestla_common.hpp"
#include "core/layers/bestla_gemm.h"
#include "bestla/bestla_parallel.h"
#include "models/model_utils/model_types.h"
#include "models/model_utils/model_config.h"
#include "models/model_utils/model_utils.h"
#include "models/model_utils/quant_utils.h"
#include "models/model_utils/scheduler.h"

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <signal.h>
#include <windows.h>
#endif

namespace py = pybind11;

namespace {
struct Query {
  uint64_t id;
  std::vector<model_vocab::id> token_ids;
  Query() {}
  Query(uint64_t id, const pybind11::array_t<model_vocab::id, py::array::c_style | py::array::forcecast>& token_ids)
      : id(id), token_ids(token_ids.data(), token_ids.data() + token_ids.size()) {
    assert(token_ids.ndim() == 1 || (token_ids.ndim() == 2 && token_ids.shape(0) == 1));
  }

  std::string to_string() const {
    const std::string repr_ids(py::str(py::array_t<int, py::array::c_style>(token_ids.size(), token_ids.data())));
    return std::to_string(id) + ": " + repr_ids;
  }
};
// Response happens to be the same structure as Query, while the tokens IDs is for prompt in a Query but is for
// generated tokens in a Response.
using Response = Query;
using ResponseCallback = std::function<void(std::vector<Response>, int)>;
}  // namespace

static std::set<model_archs> cont_batching_model_archs = {MODEL_GPTJ, MODEL_LLAMA};
void init_gpt_params(gpt_params* params, const std::string& model_path, int max_new_tokens = -1, int n_batch = 512,
                     int ctx_size = 512, int seed = -1, int threads = 8, float repetition_penalty = 1.1f,
                     int num_beams = 1, bool do_sample = false, int top_k = 40, float top_p = 0.95,
                     float temperature = 0.8, int min_new_tokens = 0, float length_penalty = 1.0f,
                     bool early_stopping = false, int n_keep = 0, int n_discard = -1, bool shift_roped_k = false,
                     int batch_size = 1, model_vocab::id pad_token = -1, const std::string& memory_dtype = "auto",
                     bool continuous_batching = true, const int& max_request_num = MODEL_MAX_REQUEST_NUM,
                     const float& scratch_size_ratio = 1.0f) {
  MODEL_ASSERT(params != nullptr);
#ifdef MODEL_NAME
  params->model_name = MODEL_NAME;
#endif
  params->model_arch = model_name_to_arch::init().find(params->model_name);
  params->model = model_path;
  params->n_predict = max_new_tokens;
  params->n_batch = n_batch;
  params->n_ctx = ctx_size;
  params->seed = seed;
  params->n_threads = threads;
  params->repeat_penalty = repetition_penalty;
  params->beam_size = num_beams;
  params->do_sample = do_sample;
  params->batch_size = batch_size;
  params->beam_search = (num_beams > 1 && !do_sample);
  params->top_k = top_k;
  params->top_p = top_p;
  params->temp = temperature;
  params->n_keep = n_keep;
  params->n_discard = n_discard;
  params->shift_roped_k = shift_roped_k;
  if (memory_dtype == "f32")
    params->memory_type = KV_MEM_TYPE_F32;
  else if (memory_dtype == "f16")
    params->memory_type = KV_MEM_TYPE_F16;
  else if (memory_dtype == "auto")
    params->memory_type = KV_MEM_TYPE_AUTO;
  else
    fprintf(stderr, "Unexpected memory dtype %s!", memory_dtype.c_str());

  // TODO(Yi & YZT): MHA IN MULTI-BATCH For More Model Archs
  params->cont_batching = continuous_batching;
  if (params->shift_roped_k) params->cont_batching = false;
  if (cont_batching_model_archs.count(params->model_arch) == 0) params->cont_batching = false;
  if (batch_size > 1 && !continuous_batching) {
    params->memory_type = KV_MEM_TYPE_F16;
  }
  params->max_request_num = std::max(batch_size, max_request_num);
  params->min_new_tokens = min_new_tokens;
  params->length_penalty = length_penalty;
  params->do_early_stopping = early_stopping;
  params->scratch_size_ratio = scratch_size_ratio;

  // TODO(Yi): MHA FOR LONG TOKENS
  int32_t tokens_length = 6144;
  if (params->n_ctx > tokens_length) {
    params->memory_type = KV_MEM_TYPE_F16;
  }

  printf(
      "beam_size: %d, do_sample: %d, top_k: %d, top_p: %.3f, continuous_batching: %d, max_request_num: %d, "
      "early_stopping: %d, scratch_size_ratio: %.3f\n",
      params->beam_size, params->do_sample, params->top_k, params->top_p, params->cont_batching,
      params->max_request_num, params->do_early_stopping, params->scratch_size_ratio);
}

class ModelServer {
 public:
  ModelServer(const ResponseCallback& response, const std::string& model_path, bool return_prompt, int max_new_tokens,
              int n_batch, int ctx_size, int seed, int threads, float repetition_penalty, int num_beams, bool do_sample,
              int top_k, float top_p, float temperature, int min_new_tokens, float length_penalty, bool early_stopping,
              int n_keep, int n_discard, bool shift_roped_k, int batch_size, model_vocab::id pad_token,
              const std::string& memory_dtype, bool continuous_batching, const int& max_request_num,
              const float& scratch_size_ratio, const std::string& policy, bool print_log,
              const std::function<void()>& init_cb)
      : response(response),
        waiting(),
        running(true),
        params(),
        policy(policy),
        scheduler_empty(false),
        working_size(0),
        return_prompt(return_prompt),
        worker([=]() {
          if (!continuous_batching) fprintf(stderr, "Warning: ModelServer only supports continuous_batching.\n");
          {
            py::gil_scoped_acquire acquirer;
            init_cb();
          }
          this->InitServerParams(model_path, max_new_tokens, n_batch, ctx_size, seed, threads, repetition_penalty,
                                 num_beams, do_sample, top_k, top_p, temperature, min_new_tokens, length_penalty,
                                 early_stopping, n_keep, n_discard, shift_roped_k, batch_size, pad_token, memory_dtype,
                                 true, max_request_num, scratch_size_ratio);
          Cont_batch_gen_scheduler scheduler(this->params, policy, print_log ? 0 : 1);
          std::vector<sequence> added_seqs;
          while (running) {
            {                                               // add waiting tasks queue to running queue
              std::lock_guard<std::mutex> lock(queue_mtx);  // need lock as issueQuery may also access waiting

              // TODO(Yi): should have some limitations
              added_seqs.resize(waiting.size());
              working_size += waiting.size();
              std::transform(waiting.cbegin(), waiting.cend(), added_seqs.begin(),
                             [&](const Query& q) { return this->Query2Sequence(q); });
              waiting.clear();
            }
            if (!added_seqs.empty()) {
              for (int i = 0; i < added_seqs.size(); ++i) {
                scheduler.add_request(added_seqs[i]);
              }
              added_seqs.clear();
            }
            if (!scheduler.done()) {
              if (!scheduler.step()) {
                fprintf(stderr, "Server has running errors, exiting...\n");
                running = false;
              }
              scheduler_empty = false;
            } else {
              if (!scheduler_empty) {
                fprintf(stdout, "Server has no requests now, waiting new query...\n");
                scheduler_empty = true;
              }
              _mm_pause();  //  spin-wait loop
            }
            if (scheduler.has_finished_seq()) {
              py::gil_scoped_acquire acquirer;
              std::vector<sequence> finished_seqs = scheduler.pop_completed_requests();
              std::vector<Query> finished(finished_seqs.size());
              working_size -= finished_seqs.size();
              std::transform(finished_seqs.cbegin(), finished_seqs.cend(), finished.begin(),
                             [&](const sequence& seq) { return this->Sequence2Query(seq); });
              for (int i = 0; i < finished.size(); ++i) {
                py::print("ID", finished[i].id, "finished in CPP server!");
              }
              this->response(finished, working_size);
            }
          }
          {
            py::gil_scoped_acquire acquirer;
            py::print("Worker stopped!");
          }
        }) {
    py::print("CPP server launched! The serve policy is", policy);
  };

  int issueQuery(std::vector<Query>& qs) {
    if (!running) throw std::runtime_error("Server stopped!");
    std::lock_guard<std::mutex> lock(queue_mtx);
    std::copy(qs.cbegin(), qs.cend(), std::back_inserter(waiting));
    return waiting.size();
  }

  bool Empty() {
    std::lock_guard<std::mutex> lock(queue_mtx);
    return (waiting.empty() && scheduler_empty);
  }

  sequence Query2Sequence(const Query& query) {
    sequence ret_seq;
    ret_seq.request_idx = -1;  // let scheduler decides it
    ret_seq.prompt_ids = query.token_ids;
    ret_seq.n_prompt_tokens = query.token_ids.size();
    ret_seq.n_tokens = query.token_ids.size();
    ret_seq.n_past = 0;
    ret_seq.n_total = 0;
    ret_seq.gen_conf.max_new_tokens = params.n_predict;
    ret_seq.gen_conf.min_new_tokens = params.min_new_tokens;
    ret_seq.gen_conf.length_penalty = params.length_penalty;
    ret_seq.gen_conf.do_early_stopping = params.do_early_stopping;
    ret_seq.query_id = query.id;
    return ret_seq;
  }

  Query Sequence2Query(const sequence& seq) {
    Query ret_query;
    ret_query.id = seq.query_id;
    int ret_size = return_prompt ? seq.prompt_ids.size() + seq.generated_ids.size() : seq.generated_ids.size();
    ret_query.token_ids.resize(ret_size);
    if (return_prompt) {
      std::copy(seq.prompt_ids.cbegin(), seq.prompt_ids.cend(), ret_query.token_ids.begin());
      std::copy(seq.generated_ids.cbegin(), seq.generated_ids.cend(),
                ret_query.token_ids.begin() + seq.prompt_ids.size());
    } else {
      std::copy(seq.generated_ids.cbegin(), seq.generated_ids.cend(), ret_query.token_ids.begin());
    }
    return ret_query;
  }

  void InitServerParams(const std::string& model_path, int max_new_tokens, int n_batch, int ctx_size, int seed,
                        int threads, float repetition_penalty, int num_beams, bool do_sample, int top_k, float top_p,
                        float temperature, int min_new_tokens, float length_penalty, bool early_stopping, int n_keep,
                        int n_discard, bool shift_roped_k, int batch_size, model_vocab::id pad_token,
                        const std::string& memory_dtype, bool continuous_batching, const int& max_request_num,
                        const float& scratch_size_ratio) {
    init_gpt_params(&params, model_path, max_new_tokens, n_batch, ctx_size, seed, threads, repetition_penalty,
                    num_beams, do_sample, top_k, top_p, temperature, min_new_tokens, length_penalty, early_stopping,
                    n_keep, n_discard, shift_roped_k, batch_size, pad_token, memory_dtype, continuous_batching,
                    max_request_num, scratch_size_ratio);
    if (cont_batching_model_archs.count(params.model_arch) == 0) {
      fprintf(stderr, "\nERROR: ModelServer only supports gpt-j, llama!\n");
      running = false;
    }
  }

  ~ModelServer() {
    // "synchronized" function
    // stop spinning after calling ResponseCallback for the last query
    py::print("Stopping CPP server...");
    running = false;
    {
      py::gil_scoped_release releaser;
      worker.join();
    }
    py::print("CPP server stopped!");
  }

 private:
  // response function from outside for collecting generation results and checking server working status
  const ResponseCallback response;
  // waiting pool for new queries added into server
  std::vector<Query> waiting;
  // lock for waiting pool
  std::mutex queue_mtx;
  // status for telling server if it still need to continue running or not
  // true: checking waiting pool and performing one step (or waiting new query)
  // false: stop server
  bool running;
  gpt_params params;
  // server policy (only FCFS (first come, first serve) now)
  std::string policy;
  // if server scheduler has no queries to run or not
  bool scheduler_empty;
  // current number of queries the server need to deal with
  uint64_t working_size;
  // add prompt token ids before generated tokens in results if set it true
  bool return_prompt;
  // server working thread
  std::thread worker;
};

std::shared_ptr<quant_layer_base> get_model_quant_layer(const std::string model_name) {
  return ql_registry::create_ql(model_name);
}

#define STATIC_INPUT_HEAD_IDX 0
class Model {
 public:
  Model() { model_init_backend(); }
  ~Model() {
    if (ctx) model_free(ctx);
  }
  void init_model(const std::string& model_path, int max_new_tokens, int n_batch, int ctx_size, int seed, int threads,
                  float repetition_penalty, int num_beams, bool do_sample, int top_k, float top_p, float temperature,
                  int min_new_tokens, float length_penalty, bool early_stopping, int n_keep, int n_discard,
                  bool shift_roped_k, int batch_size, model_vocab::id pad_token, const std::string& memory_dtype,
                  bool continuous_batching, const int& max_request_num, const float& scratch_size_ratio);
  void reinit();
  std::vector<std::vector<model_token>> generate(const std::vector<std::vector<model_token>>& input_ids);
  // deprecated API
  std::vector<std::vector<model_token>> generate_tokens(const std::vector<std::vector<model_token>>& input_ids);
  const std::vector<float>& evaluate_(const std::vector<std::vector<model_token>>& input_ids);
  py::array_t<float> evaluate(const std::vector<std::vector<model_token>>& input_ids, bool logits_all = false) {
    if (logits_all) ctx->logits_all = true;
    if (!check_input_and_count_padding(input_ids)) return py::array_t<float>();
    const auto& logits = evaluate_(input_ids);
    for (auto& input_id : curr_input_ids) input_id.clear();  // clear curr_input_ids after eval
    return py::array_t<float, py::array::c_style>(logits.size(), logits.data())
        .reshape({py::ssize_t(-1), static_cast<py::ssize_t>(ctx->model.hparams.n_vocab)});
  }
  bool is_token_end() { return token_eos; }
  model_token get_eos_id() { return ctx->vocab.eos_token_id; }
  static int quant_model(const std::string& model_path, const std::string& out_path, const std::string& weight_dtype,
                         const std::string& alg, int group_size, const std::string& scale_dtype,
                         const std::string& compute_dtype, bool use_ggml, int threads);
  void reset_token_end() {
    token_eos = false;
    curr_input_ids.clear();
    curr_input_ids.resize(params.max_request_num);
    generate_count = 0;
  }
  void print_time() { model_print_timings(ctx); }
  void reset_time() { model_reset_timings(ctx); }
  static size_t np_bestla_qpack(py::array_t<int8_t> src_w, py::array_t<float> src_scales, py::array_t<int8_t> src_zeros,
                                py::array_t<int32_t> g_idx, py::array_t<int8_t> dst, const std::string& weight_dtype,
                                const std::string& alg, int group_size, const std::string& scale_dtype,
                                const std::string& compute_dtype, int threads) {
    int8_t* w_ptr = src_w.mutable_data();
    float* scales_ptr = src_scales.mutable_data();
    int8_t* zeros_ptr = nullptr;
    if (src_zeros.size() != 0) {
      zeros_ptr = src_zeros.mutable_data();
    }
    int32_t* g_idx_ptr = nullptr;
    if (g_idx.size() != 0) {
      g_idx_ptr = g_idx.mutable_data();
    }
    int8_t* dst_ptr = dst.mutable_data();

    quant_params_internal q_params;
    q_params.bits = parse_bits(weight_dtype);
    q_params.scale_dtype = parse_scale_dtype(scale_dtype);
    q_params.compute_dtype = parse_compute_type(compute_dtype, /*ggml_arg=*/0);
    q_params.alg = parse_alg(alg);
    q_params.group_size = group_size;
    return bestla_qpack(w_ptr, scales_ptr, zeros_ptr, dst_ptr, q_params, threads, src_w.shape(1), src_w.shape(0),
                        g_idx_ptr);
  }

  static size_t np_bestla_quantize(py::array_t<float> src_w, py::array_t<int8_t> dst, const std::string& weight_dtype,
                                   const std::string& alg, int group_size, const std::string& scale_dtype,
                                   const std::string& compute_dtype, int threads) {
    quant_params_internal q_params;
    q_params.bits = parse_bits(weight_dtype);
    q_params.scale_dtype = parse_scale_dtype(scale_dtype);
    q_params.compute_dtype = parse_compute_type(compute_dtype, /*ggml_arg=*/0);
    q_params.alg = parse_alg(alg);
    q_params.group_size = group_size;
    return bestla_quantize(src_w.mutable_data(), dst.mutable_data(), q_params, threads, src_w.shape(0), src_w.shape(1));
  }

 private:
  model_context* ctx = nullptr;
  gpt_params params;
  std::vector<std::vector<model_token>> curr_input_ids;
  int n_past = 0;
  int n_total = 0;
  int n_vocab = 0;
  int n_ctx = 0;
  std::vector<std::vector<model_token>> last_n_tokens;
  bool token_eos = false;
  int64_t generate_count = 0;
  std::vector<uint32_t> padding_count;
  uint32_t n_prompt_tokens = 0;
  std::vector<float> times;

  std::vector<std::vector<model_token>> beam_generate(const std::vector<std::vector<model_token>>& input_ids);
  std::vector<model_token> post_process(const float* logits);
  std::vector<model_token> post_greedy_search(const float* logits);
  std::vector<std::vector<model_token>> post_beam_search(model_context* lctx, const int& n_predict,
                                                         const std::vector<model_input>& inputs, const int& n_threads);
  std::vector<model_token> post_sample_top_k_top_p_repeat(const float* logits);
  bool check_input_and_count_padding(const std::vector<std::vector<model_token>>& input_ids);
};

void Model::init_model(const std::string& model_path, int max_new_tokens, int n_batch, int ctx_size, int seed,
                       int threads, float repetition_penalty, int num_beams, bool do_sample, int top_k, float top_p,
                       float temperature, int min_new_tokens, float length_penalty, bool early_stopping, int n_keep,
                       int n_discard, bool shift_roped_k, int batch_size, model_vocab::id pad_token,
                       const std::string& memory_dtype, bool continuous_batching, const int& max_request_num,
                       const float& scratch_size_ratio) {
  init_gpt_params(&params, model_path, max_new_tokens, n_batch, ctx_size, seed, threads, repetition_penalty, num_beams,
                  do_sample, top_k, top_p, temperature, min_new_tokens, length_penalty, early_stopping, n_keep,
                  n_discard, shift_roped_k, batch_size, pad_token, memory_dtype, continuous_batching, max_request_num,
                  scratch_size_ratio);

  n_past = 0;
  n_total = 0;
  token_eos = false;
  curr_input_ids.clear();
  curr_input_ids.resize(params.max_request_num);
  ctx = model_init_from_gpt_params(params);
  n_vocab = model_n_vocab(ctx);
  n_ctx = model_n_ctx(ctx);
  last_n_tokens.resize(params.max_request_num);
  for (int i = 0; i < params.max_request_num; ++i) {
    last_n_tokens[i].resize(n_ctx, 0);
  }
  if (pad_token != -1) ctx->vocab.pad_token_id = pad_token;
}

void Model::reinit() {
  n_past = 0;
  n_total = 0;
  last_n_tokens.clear();
  last_n_tokens.resize(params.max_request_num);
  for (int i = 0; i < params.max_request_num; ++i) {
    last_n_tokens[i].resize(n_ctx, 0);
  }
  token_eos = false;
  curr_input_ids.clear();
  curr_input_ids.resize(params.max_request_num);
  ctx->n_sample = 0;
  ctx->t_sample_us = 0;
  generate_count = 0;
  padding_count.clear();
  n_prompt_tokens = 0;
}

bool Model::check_input_and_count_padding(const std::vector<std::vector<model_token>>& input_ids) {
  if (input_ids.empty()) {  // next token generation (internal)
    if (curr_input_ids.empty()) {
      fprintf(stderr, "%s: error: no input\n", __func__);
      return false;
    }
    return true;
  } else if (input_ids.size() == 1) {
    padding_count = {0};
    ctx->batch_size = 1;
    n_prompt_tokens = input_ids[STATIC_INPUT_HEAD_IDX].size();
    return true;
  } else {  // multi-batch inputs (first token)
    ctx->batch_size = input_ids.size();
    MODEL_ASSERT(input_ids.size() <= ctx->max_request_num);
    static std::set<model_archs> batched_model_archs = {MODEL_GPTJ, MODEL_GPTNEOX, MODEL_CHATGLM, MODEL_LLAMA};
    if (batched_model_archs.count(params.model_arch) == 0) {
      fprintf(stderr, "\nERROR: Only gpt-j, gpt-neox, chatglm, llama support multi-batch generation!\n");
      return false;
    }
    if (ctx->vocab.pad_token_id == -1) {
      fprintf(stderr, "\nERROR: please set pad_token for static multi-batch generation (tokenizer.pad_token_id)!\n");
      return false;
    }
    if (!padding_count.empty()) padding_count.clear();
    if (ctx->cont_batching) {
      padding_count.assign(input_ids.size(), 0);
      return true;
    }
    for (int bs = 0; bs < input_ids.size(); ++bs) {
      model_vocab::id pad_token_id = ctx->vocab.pad_token_id;
      auto iter = std::find_if(input_ids[bs].begin(), input_ids[bs].end(),
                               [&pad_token_id](model_token t) { return (t != pad_token_id); });
      if (iter == input_ids[bs].end()) fprintf(stderr, "\nERROR: there are all pad tokens in batch %d!\n", bs);
      padding_count.push_back(std::distance(input_ids[bs].begin(), iter));
    }
    // should be same in static batching inference
    n_prompt_tokens = input_ids[STATIC_INPUT_HEAD_IDX].size();
    return true;
  }
}

std::vector<std::vector<model_token>> Model::beam_generate(const std::vector<std::vector<model_token>>& input_ids) {
  std::vector<model_input> inputs;
  for (int bs = 0; bs < input_ids.size(); ++bs) {
    inputs.push_back(model_input{
        /*.tokens              =*/input_ids[bs].data(),
        /*.n_tokens           =*/(uint32_t)input_ids[bs].size(),
        /*.n_prompt_tokens    =*/0,
        /*.n_past             =*/0,
        /*.n_total            =*/0,
        /*.request_idx        =*/bs,
        /*.beam_idx           =*/0,
        /*.padding_side       =*/0,
        /*n_padding           =*/padding_count[bs],
    });
  }
  return post_beam_search(ctx, params.n_predict, inputs, params.n_threads);
}

const std::vector<float>& Model::evaluate_(const std::vector<std::vector<model_token>>& input_ids) {
  static const std::vector<float> empty_ret{};

  static const std::vector<model_token> empty_id{};
  std::vector<model_input> inputs;
  for (int bs = 0; bs < ctx->batch_size; ++bs) {
    const auto& input_id_cb = input_ids.empty() ? empty_id : input_ids[bs];
    if (input_id_cb.empty()) {  // use internal input id
      if (curr_input_ids[bs].empty()) {
        fprintf(stderr, "%s: error: no input\n", __func__);
        return empty_ret;
      }
    } else if (!curr_input_ids[bs].empty()) {
      fprintf(stderr, "%s: error: prompt confliction\n", __func__);
      return empty_ret;
    } else if (input_id_cb.size() > n_ctx - params.n_keep) {  // long input_id_cb and empty curr_input_ids[bs]
      fprintf(stderr, "\n%s: Warning: prompt is too long (%zu tokens, max %d), will be truncated\n", __func__,
              input_id_cb.size(), n_ctx - params.n_keep);
      curr_input_ids[bs].resize(n_ctx - params.n_keep);
      std::copy(input_id_cb.end() - n_ctx - params.n_keep * 2, input_id_cb.end(),
                curr_input_ids[bs].begin() + params.n_keep);
      std::copy(input_id_cb.begin(), input_id_cb.begin() + params.n_keep, curr_input_ids[bs].begin());
    } else {  // good input_id_cb and empty curr_input_ids[bs]
      curr_input_ids[bs] = input_id_cb;
    }

    // push elements in curr_input_ids[bs] to the last_n_tokens[bs] queue
    last_n_tokens[bs].erase(last_n_tokens[bs].begin(), last_n_tokens[bs].begin() + curr_input_ids[bs].size());
    last_n_tokens[bs].insert(last_n_tokens[bs].end(), curr_input_ids[bs].begin(), curr_input_ids[bs].end());

    // infinite text generation via context swapping
    if (n_past + curr_input_ids[bs].size() > n_ctx) {
      // always keep the first token
      n_past = std::max(1, params.n_keep);

      int n_discard = params.n_discard;
      if (!params.shift_roped_k) {  // shift_roped_k can use ring-buffer and thus does not need re-computing
        if (n_discard == -1) n_discard = (n_ctx - curr_input_ids[bs].size() - params.n_keep) / 2;
        // drop n_discard tokens
        curr_input_ids[bs].insert(curr_input_ids[bs].begin(), last_n_tokens[bs].begin() + params.n_keep + n_discard,
                                  last_n_tokens[bs].end() - curr_input_ids[bs].size());
      } else {
        NE_ASSERT(("n_discard cannot be used with shift_roped_k!", n_discard == -1 || n_discard == 1));
      }
    }

    inputs.push_back({
        /*.tokens              =*/curr_input_ids[bs].data(),
        /*.n_tokens           =*/(uint32_t)curr_input_ids[bs].size(),
        /*.n_prompt_tokens    =*/n_prompt_tokens,
        /*.n_past             =*/(uint32_t)n_past,
        /*.n_total            =*/(uint32_t)n_total,
        /*.request_idx        =*/bs,
        /*.beam_idx           =*/0,
        /*.padding_side       =*/0,
        /*n_padding           =*/padding_count[bs],
    });
  }
  model_eval(ctx, inputs.data(), inputs.size(), params.n_threads);
  // static batching inference should have same input length and context window length
  n_past += curr_input_ids[STATIC_INPUT_HEAD_IDX].size();
  n_total += curr_input_ids[STATIC_INPUT_HEAD_IDX].size();

  return ctx->logits;
}

std::vector<std::vector<model_token>> Model::generate(const std::vector<std::vector<model_token>>& input_ids) {
  if (!check_input_and_count_padding(input_ids)) return {};
  if (ctx->beam_search) return beam_generate(input_ids);

  const auto& logits = evaluate_(input_ids);
  if (logits.empty()) return {};

  std::vector<model_token> next_token_ids = post_process(logits.data());
  MODEL_ASSERT(next_token_ids.size() == ctx->batch_size);
  std::vector<std::vector<model_token>> ret_next_tokens;
  for (int bs = 0; bs < next_token_ids.size(); ++bs) {
    // padding eos seq for continuous batched kv cache
    // TODO(Zhentao): batch reduction after for-loop attention implementation
    if (curr_input_ids[bs].back() == ctx->vocab.eos_token_id || curr_input_ids[bs].back() == ctx->vocab.pad_token_id) {
      curr_input_ids[bs] = {ctx->vocab.pad_token_id};
      ret_next_tokens.push_back({ctx->vocab.pad_token_id});
    } else {
      curr_input_ids[bs] = {next_token_ids[bs]};
      ret_next_tokens.push_back({next_token_ids[bs]});
    }
  }
  generate_count++;
  return ret_next_tokens;
}

// deprecated API
std::vector<std::vector<model_token>> Model::generate_tokens(const std::vector<std::vector<model_token>>& input_ids) {
  int n_remain = params.n_predict;
  std::vector<model_token> output_ids;
  std::vector<std::vector<model_token>> rets;

  if (ctx->beam_search) {
    MODEL_ASSERT(input_ids.size() == ctx->batch_size);
    if (ctx->batch_size > 1 && ctx->vocab.pad_token_id == -1) {
      fprintf(stderr, "\nERROR: please set pad_token for beam search multi-batch generation!\n");
      return rets;
    }
    std::vector<model_input> inputs;
    for (int bs = 0; bs < input_ids.size(); ++bs) {
      uint32_t count = 0;
      model_vocab::id pad_token_id = ctx->vocab.pad_token_id;
      auto iter = std::find_if(input_ids[bs].begin(), input_ids[bs].end(),
                               [&pad_token_id](model_token t) { return (t != pad_token_id); });
      if (iter == input_ids[bs].end()) fprintf(stderr, "\nERROR: there are all pad tokens in batch %d!\n", bs);
      count = std::distance(input_ids[bs].begin(), iter);
      inputs.push_back(model_input{
          /*.tokens              =*/input_ids[bs].data(),
          /*.n_tokens           =*/(uint32_t)input_ids[bs].size(),
          /*.n_prompt_tokens    =*/0,
          /*.n_past             =*/0,
          /*.n_total            =*/0,
          /*.request_idx        =*/bs,
          /*.beam_idx           =*/0,
          /*.padding_side       =*/0,
          /*n_padding           =*/count,
      });
    }
    return post_beam_search(ctx, n_remain, inputs, params.n_threads);
  }
  if (input_ids.size() > 1) {
    fprintf(stderr, "\nERROR: Only beam search supports multi-batch generation!\n");
    return rets;
  }

  if (curr_input_ids[STATIC_INPUT_HEAD_IDX].empty()) {
    if (input_ids[STATIC_INPUT_HEAD_IDX].size() > n_ctx - params.n_keep) {
      fprintf(stderr, "\n%s: Warning: prompt is too long (%zu tokens, max %d), will be truncated\n", __func__,
              input_ids[STATIC_INPUT_HEAD_IDX].size(), n_ctx - params.n_keep);
      curr_input_ids[STATIC_INPUT_HEAD_IDX].resize(n_ctx - params.n_keep);
      std::copy(input_ids[STATIC_INPUT_HEAD_IDX].end() - n_ctx - params.n_keep * 2,
                input_ids[STATIC_INPUT_HEAD_IDX].end(), curr_input_ids[STATIC_INPUT_HEAD_IDX].begin() + params.n_keep);
      std::copy(input_ids[STATIC_INPUT_HEAD_IDX].begin(), input_ids[STATIC_INPUT_HEAD_IDX].begin() + params.n_keep,
                curr_input_ids[STATIC_INPUT_HEAD_IDX].begin());
    } else {
      curr_input_ids[STATIC_INPUT_HEAD_IDX] = input_ids[STATIC_INPUT_HEAD_IDX];
    }
  }

  while (output_ids.size() < n_remain) {
    for (auto item : curr_input_ids[STATIC_INPUT_HEAD_IDX]) {
      last_n_tokens[STATIC_INPUT_HEAD_IDX].erase(last_n_tokens[STATIC_INPUT_HEAD_IDX].begin());
      last_n_tokens[STATIC_INPUT_HEAD_IDX].push_back(item);
    }
    // infinite text generation via context swapping
    if (n_past + curr_input_ids[STATIC_INPUT_HEAD_IDX].size() > n_ctx) {
      // always keep the first token
      n_past = std::max(1, params.n_keep);

      int n_discard = params.n_discard;
      if (!params.shift_roped_k) {  // shift_roped_k can use ring-buffer and thus does not need re-computing
        if (n_discard == -1) n_discard = (n_ctx - curr_input_ids[STATIC_INPUT_HEAD_IDX].size() - params.n_keep) / 2;
        // drop n_discard tokens
        curr_input_ids[STATIC_INPUT_HEAD_IDX].insert(
            curr_input_ids[STATIC_INPUT_HEAD_IDX].begin(),
            last_n_tokens[STATIC_INPUT_HEAD_IDX].begin() + params.n_keep + n_discard,
            last_n_tokens[STATIC_INPUT_HEAD_IDX].end() - curr_input_ids[STATIC_INPUT_HEAD_IDX].size());
      } else {
        NE_ASSERT(("n_discard cannot be used with shift_roped_k!", n_discard == -1 || n_discard == 1));
      }
    }
    std::vector<model_input> inputs = {model_input{
        /*.tokens              =*/curr_input_ids[STATIC_INPUT_HEAD_IDX].data(),
        /*.n_tokens           =*/(uint32_t)curr_input_ids[STATIC_INPUT_HEAD_IDX].size(),
        /*.n_prompt_tokens    =*/0,
        /*.n_past             =*/(uint32_t)n_past,
        /*.n_total            =*/(uint32_t)n_total,
        /*.request_idx        =*/0,
        /*.beam_idx           =*/0,
        /*.padding_side       =*/0,
        /*n_padding           =*/0,
    }};
    model_eval(ctx, inputs.data(), inputs.size(), params.n_threads);
    n_past += curr_input_ids[STATIC_INPUT_HEAD_IDX].size();
    n_total += curr_input_ids[STATIC_INPUT_HEAD_IDX].size();

    float* logits = model_get_logits(ctx);
    std::vector<model_token> next_token_id = post_process(logits);
    curr_input_ids[STATIC_INPUT_HEAD_IDX] = {next_token_id[STATIC_INPUT_HEAD_IDX]};
    output_ids.push_back(next_token_id[STATIC_INPUT_HEAD_IDX]);
    generate_count++;
    if (next_token_id[STATIC_INPUT_HEAD_IDX] == ctx->vocab.eos_token_id) {
      token_eos = true;
      break;
    }
    if (params.n_predict > 0 && generate_count >= params.n_predict) {
      token_eos = true;
      break;
    }
  }
  rets.push_back(output_ids);
  return rets;
}

std::vector<model_token> Model::post_greedy_search(const float* logits) {
  std::vector<model_token> ids(ctx->batch_size);
  static int n_vocab_segment = 1024;
  int num_segments = (n_vocab + n_vocab_segment - 1) / n_vocab_segment;
  std::vector<model_token> candidate_tokens(ctx->batch_size * num_segments);
  std::vector<float> candidate_logits(ctx->batch_size * num_segments);
#pragma omp parallel for collapse(2)
  for (int bs = 0; bs < ctx->batch_size; ++bs) {
    for (int vocab = 0; vocab < n_vocab; vocab += n_vocab_segment) {
      auto max_e =
          std::max_element(logits + bs * n_vocab + vocab, vocab + n_vocab_segment > n_vocab
                                                              ? logits + bs * n_vocab + n_vocab
                                                              : logits + bs * n_vocab + vocab + n_vocab_segment);
      candidate_tokens[bs * num_segments + vocab / n_vocab_segment] = max_e - (logits + bs * n_vocab);
      candidate_logits[bs * num_segments + vocab / n_vocab_segment] = *max_e;
    }
  }
  for (int bs = 0; bs < ctx->batch_size; ++bs) {
    ids[bs] = candidate_tokens[std::distance(candidate_logits.begin(),
                                             std::max_element(candidate_logits.begin() + bs * num_segments,
                                                              candidate_logits.begin() + (bs + 1) * num_segments))];
  }
  return ids;
}

std::vector<std::vector<model_token>> Model::post_beam_search(model_context* lctx, const int& n_predict,
                                                              const std::vector<model_input>& inputs,
                                                              const int& n_threads) {
  // TODO(Zhentao): to implement
  static std::set<model_archs> supported_archs = {MODEL_GPTJ, MODEL_GPTNEOX, MODEL_LLAMA};
  if (supported_archs.count(params.model_arch) != 0) {
    return beam_search(lctx, n_predict, inputs, n_threads);
  } else {
    fprintf(stderr, "\nERROR: this model does not support beam search generation!\n");
    return std::vector<std::vector<model_token>>();
  }
}

std::vector<model_token> Model::post_sample_top_k_top_p_repeat(const float* logits) {
  int alpha_frequency = 0;
  int alpha_presence = 0;
  int repeat_last_n = 64;
  int top_k = params.top_k;
  float tfs_z = 1.00f;
  float typical_p = 1.00f;
  float top_p = params.top_p;
  float temp = params.temp;
  std::vector<model_token> ids(ctx->batch_size);
  // #pragma omp parallel for  // omp will affect sampling positions in batch infer
  // TODO(Zhentao): (make sample functions support batch processing)
  for (int bs = 0; bs < ctx->batch_size; ++bs) {
    std::vector<model_token_data> candidates;
    candidates.reserve(n_vocab);
    for (model_token token_id = 0; token_id < n_vocab; token_id++) {
      candidates.emplace_back(model_token_data{token_id, logits[bs * n_vocab + token_id], 0.0f});
    }
    model_token_data_array candidates_p = {candidates.data(), candidates.size(), false};

    // Apply penalties
    float nl_logit = logits[bs * n_vocab + model_token_nl()];
    auto last_n_repeat = std::min(std::min(static_cast<int>(last_n_tokens[bs].size()), repeat_last_n), n_ctx);
    model_sample_repetition_penalty(ctx, &candidates_p,
                                    last_n_tokens[bs].data() + last_n_tokens[bs].size() - last_n_repeat, last_n_repeat,
                                    params.repeat_penalty);
    model_sample_frequency_and_presence_penalties(ctx, &candidates_p,
                                                  last_n_tokens[bs].data() + last_n_tokens[bs].size() - last_n_repeat,
                                                  last_n_repeat, alpha_frequency, alpha_presence);
    // int id = model_sample_token_greedy(ctx, &candidates_p);
    // Temperature sampling
    model_sample_top_k(ctx, &candidates_p, top_k, 1);
    model_sample_tail_free(ctx, &candidates_p, tfs_z, 1);
    model_sample_typical(ctx, &candidates_p, typical_p, 1);
    model_sample_top_p(ctx, &candidates_p, top_p, 1);
    model_sample_temperature(ctx, &candidates_p, temp);
    ids[bs] = model_sample_token(ctx, &candidates_p);
  }
  return ids;
}

std::vector<model_token> Model::post_process(const float* logits) {
  assert(("Beam search does not support streaming.", params.beam_size == 1));
  if (params.do_sample == false) {
    return post_greedy_search(logits);
  } else {
    return post_sample_top_k_top_p_repeat(logits);
  }
}

int Model::quant_model(const std::string& model_path, const std::string& out_path, const std::string& weight_dtype,
                       const std::string& alg, int group_size, const std::string& scale_dtype,
                       const std::string& compute_dtype, bool use_ggml, int threads) {
  quant_params q_params;
#ifdef MODEL_NAME
  q_params.model_name = MODEL_NAME;
#endif
  model_archs mt = model_name_to_arch::init().find(q_params.model_name);
  if (mt == MODEL_UNKNOWN) {
    fprintf(stderr, "error, please set model_name \n");
    exit(0);
  }
  q_params.model_arch = mt;
  q_params.model_file = model_path;
  q_params.out_file = out_path;
  q_params.weight_dtype = weight_dtype;
  q_params.alg = alg;
  q_params.group_size = group_size;
  q_params.scale_dtype = scale_dtype;
  q_params.compute_dtype = compute_dtype;
  q_params.use_ggml = use_ggml;
  q_params.nthread = threads;

  ne_ftype ftype = quant_params_to_ftype(q_params);
  printf("ne_ftype: %d\n", ftype);

  auto quant_layer = get_model_quant_layer(q_params.model_name);
  if (model_quantize(q_params, quant_layer)) {
    fprintf(stderr, "%s: failed to quantize model from '%s'\n", __func__, q_params.model_file.c_str());
    return 1;
  }
  return 0;
}

#if MODEL_NAME_ID == 1

PYBIND11_MODULE(gptj_cpp, m)
#elif MODEL_NAME_ID == 2

PYBIND11_MODULE(falcon_cpp, m)

#elif MODEL_NAME_ID == 3

PYBIND11_MODULE(gptneox_cpp, m)

#elif MODEL_NAME_ID == 4

PYBIND11_MODULE(dolly_cpp, m)

#elif MODEL_NAME_ID == 5

PYBIND11_MODULE(llama_cpp, m)

#elif MODEL_NAME_ID == 6

PYBIND11_MODULE(mpt_cpp, m)

#elif MODEL_NAME_ID == 7

PYBIND11_MODULE(starcoder_cpp, m)

#elif MODEL_NAME_ID == 8

PYBIND11_MODULE(opt_cpp, m)

#elif MODEL_NAME_ID == 9

PYBIND11_MODULE(bloom_cpp, m)

#elif MODEL_NAME_ID == 10

PYBIND11_MODULE(chatglm2_cpp, m)

#elif MODEL_NAME_ID == 11

PYBIND11_MODULE(chatglm_cpp, m)

#elif MODEL_NAME_ID == 12

PYBIND11_MODULE(baichuan_cpp, m)

#elif MODEL_NAME_ID == 13

PYBIND11_MODULE(polyglot_cpp, m)

#elif MODEL_NAME_ID == 14

PYBIND11_MODULE(mistral_cpp, m)

#elif MODEL_NAME_ID == 15

PYBIND11_MODULE(qwen_cpp, m)

#elif MODEL_NAME_ID == 16

PYBIND11_MODULE(phi_cpp, m)

#elif MODEL_NAME_ID == 17

PYBIND11_MODULE(stablelm_cpp, m)

#elif MODEL_NAME_ID == 18

PYBIND11_MODULE(whisper_cpp, m)

#elif MODEL_NAME_ID == 19

PYBIND11_MODULE(mixtral_cpp, m)

#elif MODEL_NAME_ID == 20

PYBIND11_MODULE(chatglm3_cpp, m)
#elif MODEL_NAME_ID == 21

PYBIND11_MODULE(gemma_cpp, m)

#endif
{
  m.doc() = "cpp model python binding";
  py::class_<Model>(m, "Model", py::module_local())
      .def(py::init())
      .def("init_model", &Model::init_model, "initial model with model path and parameters", py::arg("model_path"),
           py::arg("max_new_tokens") = -1, py::arg("n_batch") = 512, py::arg("ctx_size") = 1024, py::arg("seed") = -1,
           py::arg("threads") = 8, py::arg("repetition_penalty") = 1.1f, py::arg("num_beams") = 1,
           py::arg("do_sample") = false, py::arg("top_k") = 40, py::arg("top_p") = 0.95, py::arg("temperature") = 0.8,
           py::arg("min_new_tokens") = 0, py::arg("length_penalty") = 1.0, py::arg("early_stopping") = false,
           py::arg("n_keep") = 0, py::arg("n_discard") = -1, py::arg("shift_roped_k") = false,
           py::arg("batch_size") = 1, py::arg("pad_token") = -1, py::arg("memory_dtype") = "auto",
           py::arg("continuous_batching") = true, py::arg("max_request_num") = MODEL_MAX_REQUEST_NUM,
           py::arg("scratch_size_ratio") = 1.0f)
      .def("generate", &Model::generate, "Generate token with input ids", py::arg("input_ids"))
      .def("evaluate", &Model::evaluate, "Evaluate token with input ids and output logits",
           py::arg("input_ids") = std::vector<std::vector<model_token>>{}, py::arg("logits_all") = false)
      // deprecated API
      .def("generate_tokens", &Model::generate_tokens, "Generate tokens with input ids", py::arg("input_ids"))
      .def_static("quant_model", &Model::quant_model, "Quantize model", py::arg("model_path"), py::arg("out_path"),
                  py::arg("weight_dtype") = "int4", py::arg("alg") = "sym", py::arg("group_size") = 32,
                  py::arg("scale_dtype") = "fp32", py::arg("compute_dtype") = "int8", py::arg("use_ggml") = false,
                  py::arg("threads") = 8)
      .def("is_token_end", &Model::is_token_end)
      .def("reset_token_end", &Model::reset_token_end)
      .def_static("np_bestla_qpack", &Model::np_bestla_qpack, "QPack tensor to bestla format", py::arg("src_w"),
                  py::arg("src_scales"), py::arg("src_zeros"), py::arg("g_idx"), py::arg("dst"),
                  py::arg("weight_dtype") = "int4", py::arg("alg") = "sym", py::arg("group_size") = 32,
                  py::arg("scale_dtype") = "fp32", py::arg("compute_dtype") = "int8", py::arg("threads") = 8)
      .def_static("np_bestla_quantize", &Model::np_bestla_quantize, "Quantize tensor to bestla format",
                  py::arg("src_w"), py::arg("dst"), py::arg("weight_dtype") = "int4", py::arg("alg") = "sym",
                  py::arg("group_size") = 32, py::arg("scale_dtype") = "fp32", py::arg("compute_dtype") = "int8",
                  py::arg("threads") = 8)
      .def("print_time", &Model::print_time)
      .def("reset_time", &Model::reset_time)
      .def("get_eos_id", &Model::get_eos_id)
      .def("reinit", &Model::reinit);
  py::class_<Query>(m, "Query")
      .def(py::init<uint64_t, py::array_t<model_vocab::id>>())
      .def("__repr__", &Query::to_string)
      .def_readwrite("id", &Query::id)
      .def_readwrite("token_ids", &Query::token_ids);
  py::class_<ModelServer>(m, "ModelServer", py::module_local())
      .def(py::init<const ResponseCallback&, const std::string&, bool, int, int, int, int, int, float, int, bool, int,
                    float, float, int, float, bool, int, int, bool, int, model_vocab::id, const std::string&, bool,
                    const int&, const float&, const std::string&, bool, const std::function<void()>&>(),
           py::arg("response"), py::arg("model_path"), py::arg("return_prompt") = false, py::arg("max_new_tokens") = -1,
           py::arg("n_batch") = 512, py::arg("ctx_size") = 512, py::arg("seed") = -1, py::arg("threads") = 8,
           py::arg("repetition_penalty") = 1.1f, py::arg("num_beams") = 1, py::arg("do_sample") = false,
           py::arg("top_k") = 40, py::arg("top_p") = 0.95, py::arg("temperature") = 0.8, py::arg("min_new_tokens") = 0,
           py::arg("length_penalty") = 1.0, py::arg("early_stopping") = false, py::arg("n_keep") = 0,
           py::arg("n_discard") = -1, py::arg("shift_roped_k") = false, py::arg("batch_size") = 1,
           py::arg("pad_token") = -1, py::arg("memory_dtype") = "auto", py::arg("continuous_batching") = true,
           py::arg("max_request_num") = MODEL_MAX_REQUEST_NUM, py::arg("scratch_size_ratio") = 1.0f,
           py::arg("policy") = "fcfs", py::arg("print_log") = false,
           py::arg("init_cb") = std::function<void()>{[]() {}})
      .def("issueQuery", &ModelServer::issueQuery, "desc placeholder", py::arg("qs"))
      .def("Empty", &ModelServer::Empty, "No more queries to execute");
}
