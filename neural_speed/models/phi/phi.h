#ifndef PHI_H
#define PHI_H

#include "models/model_utils/model_files.h"
#include "models/model_utils/model_types.h"

enum new_model {
  MDOEL_UNKNOWN,
  PHI,
};

static const model_scratch phi_mem_req(int n_layers) {
  switch (n_layers) {
    case 24:
      return {512ull * MB, 512ull * MB, 1026ull * MB};
    case 32:
      return {1024ull * MB, 1024ull * MB, 1026ull * MB};
    default:
      MODEL_ASSERT(false);
  }
}

class phi : public IModel {
 private:
  model_archs name = MODEL_PHI;
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

#endif  // PHI_H
