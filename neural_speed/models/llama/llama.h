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

#ifndef LLAMA_H
#define LLAMA_H

#include "models/model_utils/model_files.h"
#include "models/model_utils/model_types.h"

enum llama_model {
  LLAMA_UNKNOWN,
  LLAMA_7B,
  LLAMA_13B,
  LLAMA_30B,
  LLAMA_65B,
};

static const model_scratch llama_mem_req(int n_layers, float scratch_size_ratio = 1.0f) {
  switch (n_layers) {
    case 32:
      return {
          static_cast<unsigned long long>(scratch_size_ratio * 4096) * MB,
          static_cast<unsigned long long>(scratch_size_ratio * 2048) * MB,
          static_cast<unsigned long long>(scratch_size_ratio * 4096) * MB,
      };
    case 40:
      return {
          static_cast<unsigned long long>(scratch_size_ratio * 4096) * MB,
          static_cast<unsigned long long>(scratch_size_ratio * 2048) * MB,
          static_cast<unsigned long long>(scratch_size_ratio * 4096) * MB,
      };
    case 48:
      return {
          static_cast<unsigned long long>(scratch_size_ratio * 4096) * MB,
          static_cast<unsigned long long>(scratch_size_ratio * 2048) * MB,
          static_cast<unsigned long long>(scratch_size_ratio * 4096) * MB,
      };
    case 60:
      return {
          static_cast<unsigned long long>(scratch_size_ratio * 4096) * MB,
          static_cast<unsigned long long>(scratch_size_ratio * 2048) * MB,
          static_cast<unsigned long long>(scratch_size_ratio * 4096) * MB,
      };
    case 80:
      return {
          static_cast<unsigned long long>(scratch_size_ratio * 3072) * MB,
          static_cast<unsigned long long>(scratch_size_ratio * 2048) * MB,
          static_cast<unsigned long long>(scratch_size_ratio * 3072 * 3) * MB,
      };
    default:
      MODEL_ASSERT(false);
  }
}

class Llama : public IModel {
 private:
  model_archs arch = MODEL_LLAMA;
  std::unique_ptr<model_model_loader> ml;
  uint32_t n_layer, n_embd, n_ff, n_vocab, n_head, n_head_kv, n_expert, n_expert_used;
  int n_gpu_layer;
  bool use_mmap, use_mlock, vocab_only;
  model_scratch scratch;

 public:
  void init(const char* path_model, model_context* ctx, int n_gpu_layers, bool use_mmap_, bool use_mlock_,
            bool vocab_only_) override;
  void load(model_context* ctx, model_progress_callback progress_callback, void* progress_callback_user_data) override;
};

#endif  // LLAMA_H
