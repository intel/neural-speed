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

#ifndef GEMMA_H
#define GEMMA_H

#include "models/model_utils/model_files.h"
#include "models/model_utils/model_types.h"

enum gemma_model {
  GEMMA_2B,
  GEMMA_7B,
};

static const model_scratch gemma_mem_req(int n_layers, float enlarge_scale = 1.0f) {
  switch (n_layers) {
    case 18:
      return {
          static_cast<unsigned long long>(enlarge_scale * 1024) * MB,
          static_cast<unsigned long long>(enlarge_scale * 1024) * MB,
          static_cast<unsigned long long>(enlarge_scale * 1608) * MB,
      };
    case 28:
      return {
          static_cast<unsigned long long>(enlarge_scale * 1024) * MB,
          static_cast<unsigned long long>(enlarge_scale * 1024) * MB,
          static_cast<unsigned long long>(enlarge_scale * 1608) * MB,
      };
    default:
      MODEL_ASSERT(false);
  }
}

class Gemma : public IModel {
 private:
  model_archs arch = MODEL_GEMMA;
  std::unique_ptr<model_model_loader> ml;
  uint32_t n_layer, n_embd, n_ff, n_vocab, n_head, n_head_kv, n_expert, n_expert_used, n_embd_head_k;
  int n_gpu_layer;
  bool use_mmap, use_mlock, vocab_only;
  model_scratch scratch;

 public:
  void init(const char* path_model, model_context* ctx, int n_gpu_layers, bool use_mmap_, bool use_mlock_,
            bool vocab_only_) override;
  void load(model_context* ctx, model_progress_callback progress_callback, void* progress_callback_user_data) override;
};

#endif  // GEMMA_H
