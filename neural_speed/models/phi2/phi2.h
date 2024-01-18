#ifndef PHI2_H
#define PHI2_H

#include "models/model_utils/model_files.h"
#include "models/model_utils/model_types.h"

enum new_model {
  MDOEL_UNKNOWN,
  PHI2,
};

static const model_scratch phi2_mem_req(int n_layers) {
  switch (n_layers) {
    case 32:
      return {1024ull * MB, 1024ull * MB, 1026ull * MB};
    default:
      MODEL_ASSERT(false);
  }
}

class phi2 : public IModel {
 private:
  model_archs name = MODEL_PHI2;
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

#endif  // PHI2_H
