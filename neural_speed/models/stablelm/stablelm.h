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

#ifndef STABLELM_H
#define STABLELM_H

#include "models/model_utils/model_files.h"
#include "models/model_utils/model_types.h"

enum stablelm_model {
  STABLELM_UNKNOWN,
  STABLELM_1_6B,
  STABLELM_3B,
};

static const model_scratch stablelm_mem_req(int n_layers) {
  switch (n_layers) {
    case 24:
      return {512ull * MB, 512ull * MB, 1026ull * MB};  // StableLM2-1.6B & StableLM2-Zephyr-1.6B
    case 32:
      return {1024ull * MB, 1024ull * MB, 1026ull * MB};  // StableLM-3B
    default:
      MODEL_ASSERT(false);
  }
}

class stablelm : public IModel {
 private:
  model_archs name = MODEL_STABLELM;
  std::unique_ptr<model_model_loader> ml;
  uint32_t n_layer, n_embd, n_ff, n_vocab;
  int n_ctx, n_gpu_layer;
  bool use_mmap, use_mlock, vocab_only;
  model_scratch scratch;

 public:
  void init(const char* path_model, model_context* ctx, int n_gpu_layers, bool use_mmap_, bool use_mlock_,
            bool vocab_only_) override;
  void load(model_context* ctx, model_progress_callback progress_callback, void* progress_callback_user_data) override;
};

#endif  // STABLELM_H
