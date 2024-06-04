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
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstring>
#include <exception>
#include <fstream>
#include <iterator>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/data_types.h"
#include "core/ne.h"
#include "core/ne_layers.h"
#include "core/ne_bestla.h"
#include "core/layers/mha_dense.h"
#include "models/model_utils/model_config.h"
#include "models/model_utils/model_utils.h"
#include "models/model_utils/util.h"

// evaluate the transformer
//
//   - lctx:      model context
//   - tokens:    new batch of tokens to process
//   - n_past:    the offset to which the kv is cached to
//   - n_total:   the number of tokens evaluated so far (including evicted tokens if there is any)
//   - n_threads: number of threads to use
//

static bool grok_model_eval_internal(model_context* ctx, const model_input* inputs, const int n_input,
                                     const int n_threads) {
  const int64_t t_start_us = ne_time_us();
  model_context& lctx = *ctx;

  // static batching for now
  const int N = inputs->n_tokens;
  const int n_past = inputs->n_past;
  const int n_total = inputs->n_total;
  const bool shift_roped_k = lctx.shift_roped_k;
  const bool is_ring_full = shift_roped_k && n_total > n_past;
  NE_ASSERT(("Shift-RoPE-K to be implemented for the neox-mode RoPE!", !is_ring_full));
  const int batch_size = lctx.batch_size;
  MODEL_ASSERT(batch_size == n_input);
  const int kv_n_ctx_block = lctx.kv_n_ctx_block;

  const auto& model = lctx.model;
  const auto& hparams = model.hparams;

  const auto& kv_self = model.kv_self;

  MODEL_ASSERT(!!kv_self.ctx);

  const int n_embd = hparams.n_embd;
  const int n_layer = hparams.n_layer;
  const int n_ctx = lctx.n_ctx;
  const int n_keep = lctx.n_keep;
  const int n_head = hparams.n_head;
  const int n_head_kv = hparams.n_head_kv;
  const int n_vocab = hparams.n_vocab;
  const int n_rot = hparams.n_rot;
  const int head_dim = n_embd / n_head;
  const int n_gqa_embd = head_dim * n_head_kv;
  int n_expert = hparams.n_experts;
  int n_expert_used = hparams.n_experts_used;

  auto& mem_per_token = lctx.mem_per_token;
  auto& buf_compute = lctx.buf_compute;

  struct ne_init_params params = {
      /*.mem_size   =*/buf_compute.size,
      /*.mem_buffer =*/buf_compute.addr,
      /*.no_alloc   =*/false,
  };

  struct ne_context* ctx0 = ne_init(params);

  // for big progptneoxs, if BLAS is enabled, it is better to use only one thread
  // otherwise, the threads are spin-lock waiting for the BLAS calls and are degrading the performance
  ne_cgraph gf = {};
  gf.n_threads = N >= 32 && ne_cpu_has_blas() ? 1 : n_threads;

  const bool run_mha_reordered = kv_self.k->type == NE_TYPE_BTLA;
  kv_cache_info_t kv_cache_info = {};
  if (run_mha_reordered) {
    NE_ASSERT(("kv cache should be the same dtype", kv_self.v->type == NE_TYPE_BTLA));
    attn_shape_t attn_shape = {
        /* .batch_size = */ 1,
        /* .head_num = */ n_head,
        /* .heads_kv = */ n_head_kv,
        /* .head_size = */ head_dim,
        /* .sl_q = */ N,  // Note: make sure that bestla reordered attn supports next token inference
        /* .sl_kv = */ n_past + N,
    };

    NE_ASSERT(("bestla managed kv-cache not supported; use `--memory-f16 / --memory-f32` instead",
               bestla_reordered_attn_fp32_support(&attn_shape)));
    kv_shape_t kv_shape{
        /* .heads_kv = */ static_cast<uint32_t>(n_head_kv),
        /* .head_size = */ static_cast<uint32_t>(head_dim),
        /* .sl_kv_max = */ static_cast<uint32_t>(n_ctx),
    };
    bestla_reordered_attn_fp32_batch_kv_info(&kv_shape, &kv_cache_info);
  }
  struct ne_tensor* embd = d_ne_new_tensor_1d(ctx0, NE_TYPE_I32, N * batch_size);
  ne_set_name(embd, "embd");
  for (int i = 0; i < batch_size; ++i) {
    memcpy(static_cast<model_token*>(embd->data) + i * N, (inputs + i)->tokens, N * ne_element_size(embd));
  }

  struct ne_tensor* inpL = ne_get_rows(ctx0, model.others[0], embd);
  inpL = ne_scale(ctx0, inpL, ne_new_f32(ctx0, 78.38367176906169f));

  for (int il = 0; il < n_layer; ++il) {
    struct ne_tensor* cur;
    lctx.use_buf(ctx0, 0);
    // RMS
    cur = ne_rms_norm(ctx0, inpL, hparams.norm_eps);
    // cur = cur*attention_norm(broadcasted)
    cur = ne_mul(ctx0, cur, model.layers[il].norm[0]);

    // compute QKV
    struct ne_tensor* Qcur;
    struct ne_tensor* Kcur;
    struct ne_tensor* Vcur;

    Qcur = ne_mul_mat(ctx0, model.layers[il].attn[0], cur);
    Qcur = ne_reshape_3d(ctx0, Qcur, head_dim, n_head, N);

    Kcur = ne_mul_mat(ctx0, model.layers[il].attn[1], cur);
    Kcur = ne_reshape_3d(ctx0, Kcur, head_dim, n_head_kv, N);

    Vcur = ne_mul_mat(ctx0, model.layers[il].attn[2], cur);
    Vcur = ne_reshape_3d(ctx0, Vcur, head_dim, n_head_kv, N);

    // using mode = 2 for GPT-NeoX mode
    Qcur = ne_rope_inplace(ctx0, Qcur, n_past, 128, 2, 0, hparams.freq_base, hparams.freq_scale);
    ne_set_name(Qcur, "Qcur");
    Kcur = ne_rope_inplace(ctx0, Kcur, n_past, 128, 2, 0, hparams.freq_base, hparams.freq_scale);
    ne_set_name(Kcur, "kcur");
    const float attn_scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    // store key and value to memory
    if (!run_mha_reordered) {
      {
        std::vector<ne_tensor*> Kcur_bs(batch_size);
        std::vector<ne_tensor*> Vcur_bs(batch_size);
        std::vector<ne_tensor*> k_bs(batch_size);
        std::vector<ne_tensor*> v_bs(batch_size);
        for (int i = 0; i < batch_size; ++i) {
          // batch K
          Kcur_bs[i] = ne_permute(ctx0,
                                  ne_view_4d(ctx0, Kcur, head_dim, n_head_kv, N, 1, ne_element_size(Kcur) * head_dim,
                                             ne_element_size(Kcur) * n_gqa_embd, ne_element_size(Kcur) * n_gqa_embd * N,
                                             i * ne_element_size(Kcur) * n_gqa_embd * N),
                                  0, 2, 1, 3);
          k_bs[i] = ne_view_4d(
              ctx0, kv_self.k, head_dim, N, n_head_kv, 1, ne_element_size(kv_self.k) * head_dim,
              ne_element_size(kv_self.k) * head_dim * n_ctx, ne_element_size(kv_self.k) * n_gqa_embd * n_ctx,
              ((il * n_ctx) * ne_element_size(kv_self.k) * n_gqa_embd * kv_n_ctx_block +
               i * n_ctx * n_gqa_embd * ne_element_size(kv_self.k) + head_dim * n_past * ne_element_size(kv_self.k)));

          // batch V
          Vcur_bs[i] =
              ne_permute(ctx0,
                         ne_reshape_4d(ctx0,
                                       ne_view_2d(ctx0, Vcur, n_gqa_embd, N, ne_element_size(Vcur) * n_gqa_embd,
                                                  i * ne_element_size(Vcur) * n_gqa_embd * N),
                                       head_dim, n_head_kv, N, 1),
                         1, 2, 0, 3);
          v_bs[i] =
              ne_view_4d(ctx0, kv_self.v, N, head_dim, n_head_kv, 1, n_ctx * ne_element_size(kv_self.v),
                         n_ctx * ne_element_size(kv_self.v) * head_dim, n_ctx * ne_element_size(kv_self.v) * n_gqa_embd,
                         ((il * n_ctx) * ne_element_size(kv_self.v) * n_gqa_embd * kv_n_ctx_block +
                          i * n_ctx * n_gqa_embd * ne_element_size(kv_self.v) + n_past * ne_element_size(kv_self.v)));
          ne_build_forward_expand(&gf, ne_cpy(ctx0, Kcur_bs[i], k_bs[i]));
          ne_build_forward_expand(&gf, ne_cpy(ctx0, Vcur_bs[i], v_bs[i]));
        }
      }
      // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
      struct ne_tensor* Q = ne_permute(ctx0, ne_reshape_4d(ctx0, Qcur, head_dim, n_head, N, batch_size), 0, 2, 1, 3);

      // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
      struct ne_tensor* K = ne_view_4d(
          ctx0, kv_self.k, head_dim, n_past + N, n_head_kv, batch_size, ne_element_size(kv_self.k) * head_dim,
          ne_element_size(kv_self.k) * head_dim * n_ctx, ne_element_size(kv_self.k) * n_gqa_embd * n_ctx,
          il * n_ctx * ne_element_size(kv_self.k) * n_gqa_embd * kv_n_ctx_block);

      // K * Q
      // struct ne_tensor* KQ = ne_scale_inplace(ctx0,ne_mul_mat(ctx0, K, Q),ne_new_f32(ctx0,0.08838834764831845f));
      struct ne_tensor* KQ = ne_scale_inplace(ctx0, ne_mul_mat(ctx0, K, Q), ne_new_f32(ctx0, attn_scale / 30.0f));
      struct ne_tensor* KQ_scaled = ne_scale_inplace(ctx0, ne_tanh(ctx0, KQ), ne_new_f32(ctx0, 30.0f));

      // KQ_masked = mask_past(KQ_scaled)
      struct ne_tensor* KQ_masked = ne_diag_mask_inf_inplace(ctx0, KQ_scaled, n_past);

      // KQ = soft_max(KQ_masked)
      struct ne_tensor* KQ_soft_max = ne_soft_max_inplace(ctx0, KQ_masked);

      // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
      struct ne_tensor* V =
          ne_view_4d(ctx0, kv_self.v, n_past + N, head_dim, n_head_kv, batch_size, n_ctx * ne_element_size(kv_self.v),
                     n_ctx * ne_element_size(kv_self.v) * head_dim, n_ctx * ne_element_size(kv_self.v) * n_gqa_embd,
                     il * n_ctx * ne_element_size(kv_self.v) * n_gqa_embd * kv_n_ctx_block);

      // KQV = transpose(V) * KQ_soft_max
      struct ne_tensor* KQV = ne_mul_mat(ctx0, V, KQ_soft_max);

      // KQV_merged = KQV.permute(0, 2, 1, 3)
      struct ne_tensor* KQV_merged = ne_permute(ctx0, KQV, 0, 2, 1, 3);

      // cur = KQV_merged.contiguous().view(n_embd, N)
      cur = ne_cpy(ctx0, KQV_merged, d_ne_new_tensor_2d(ctx0, NE_TYPE_F32, n_embd, N * batch_size));
    } else {
      const auto seq_kv = n_past + N;
      const auto k_size = kv_cache_info.k_bytes;
      const auto v_size = kv_cache_info.v_bytes;

      // store key and value to memory
      {
        const auto k_cache = ne_view_3d(ctx0, kv_self.k,             // tensor
                                        head_dim, n_ctx, n_head_kv,  // ne
                                        0, 0,                        // nb (bestla managed)
                                        il * k_size);                // offset
        ne_build_forward_expand(&gf, ne_flash_attn_update_k(ctx0, k_cache, Kcur, n_past, false));
        const auto v_cache = ne_view_3d(ctx0, kv_self.v,             // tensor
                                        head_dim, n_ctx, n_head_kv,  // ne
                                        0, 0,                        // nb (bestla managed)
                                        il * v_size);                // offset
        ne_build_forward_expand(&gf, ne_flash_attn_update_v(ctx0, v_cache, Vcur, n_past, false));
      }

      struct ne_tensor* Q = ne_permute(ctx0, Qcur, 0, 2, 1, 3);
      ne_set_name(Q, "Q");

      struct ne_tensor* K =
          ne_view_3d(ctx0, kv_self.k,                                             // tensor
                     head_dim, seq_kv, n_head_kv,                                 // ne
                     kv_cache_info.stride_k_sl, kv_cache_info.stride_k_head_num,  // nb (bestla managed)
                     il * k_size);                                                // offset
      *reinterpret_cast<ATTN_FWD_LAYOUT*>(&K->nb[0]) = kv_cache_info.k_layout;    // us nb0 for layout
      ne_set_name(K, "K");
      struct ne_tensor* V =
          ne_view_3d(ctx0, kv_self.v,                                                    // tensor
                     seq_kv, head_dim, n_head_kv,                                        // ne
                     kv_cache_info.stride_v_head_size, kv_cache_info.stride_v_head_num,  // nb (bestla managed)
                     il * v_size);                                                       // offset
      *reinterpret_cast<ATTN_FWD_LAYOUT*>(&V->nb[0]) = kv_cache_info.v_layout;           // us nb0 for layout
      ne_set_name(V, "V");

      ne_attn_flags_t attn_flags = 0;
      attn_flags |= NE_ATTN_FLAG_IS_TANH30;
      if (n_past == 0) attn_flags |= NE_ATTN_FLAG_IS_CAUSAL;  // no causal mask on next-token cases
      struct ne_tensor* KQV_Out = ne_flash_attn(ctx0, Q, K, V, attn_scale, attn_flags);
      cur = ne_view_2d(ctx0, KQV_Out, n_embd, N, n_embd * ne_element_size(KQV_Out), 0);
    }
    // projection

    cur = ne_mul_mat(ctx0, model.layers[il].attn[3], cur);

    lctx.use_buf(ctx0, 1);
    cur = ne_rms_norm(ctx0, cur, hparams.norm_eps);

    cur = ne_mul(ctx0, cur, model.layers[il].norm[2]);
    cur = ne_add(ctx0, cur, inpL);
    inpL = cur;

    // this is independent of the self-attention result, so it could be done in parallel to the self-attention
    // note here we pass inpL instead of cur
    {
      cur = ne_rms_norm(ctx0, cur, hparams.norm_eps);

      cur = ne_mul(ctx0, cur, model.layers[il].norm[1]);
    }
    ne_tensor* logits = ne_mul_mat(ctx0, model.layers[il].ffn_gate_inp, cur);  // [n_tokens, num_experts]
    ne_tensor* probs = ne_soft_max_inplace(ctx0, logits);
    ne_tensor* selected_experts = ne_top_k(ctx0, probs, n_expert_used);
    ne_tensor* weights = ne_get_rows(ctx0, ne_reshape_3d(ctx0, probs, 1, n_expert, N), selected_experts);
    weights = ne_reshape_2d(ctx0, weights, n_expert_used, N);
    // ne_tensor* weights_sum = ne_sum_rows(ctx0, weights);
    // weights_sum = ne_repeat(ctx0, weights_sum, weights);
    // weights = ne_div(ctx0, weights, weights_sum);
    ne_tensor* moe_out = nullptr;
    for (int i = 0; i < n_expert_used; ++i) {
      ne_tensor* cur_expert;
      if (bestla_fusion_FFN_Gelu_Mul_f32f32_support(
              model.layers[il].ffn_gate_exp[0]->data, model.layers[il].ffn_down_exp[0]->data,
              model.layers[il].ffn_up_exp[0]->data, N, cur->ne[0], model.layers[il].ffn_gate_exp[0]->ne[1],
              model.layers[il].ffn_down_exp[0]->ne[1])) {
        cur_expert = ne_mul_id_ffn_gelu(ctx0, model.layers[il].ffn_down_exp, model.layers[il].ffn_gate_exp,
                                        model.layers[il].ffn_up_exp, n_expert, selected_experts, i, cur);
      } else {
        ne_tensor* cur_up = ne_mul_mat_id(ctx0, model.layers[il].ffn_up_exp, n_expert, selected_experts, i, cur);
        ne_set_name(cur_up, "ffn_moe_up");

        ne_tensor* cur_gate = ne_mul_mat_id(ctx0, model.layers[il].ffn_gate_exp, n_expert, selected_experts, i, cur);
        ne_set_name(cur_gate, "ffn_moe_gate");

        cur_gate = ne_gelu(ctx0, cur_gate);
        ne_set_name(cur_gate, "ffn_moe_gelu");

        cur_expert = ne_mul(ctx0, cur_up, cur_gate);  // [n_tokens, n_embd]
        ne_set_name(cur_expert, "ffn_moe_gate_par");

        cur_expert = ne_mul_mat_id(ctx0, model.layers[il].ffn_down_exp, n_expert, selected_experts, i,
                                   cur_expert);  // [n_tokens, n_embd]
        ne_set_name(cur_expert, "ffn_moe_down");

        cur_expert =
            ne_mul(ctx0, cur_expert,
                   ne_repeat(ctx0, ne_view_2d(ctx0, weights, 1, N, weights->nb[1], i * weights->nb[0]), cur_expert));
        ne_set_name(cur_expert, "ffn_moe_weighted_");
      }

      if (i == 0) {
        moe_out = cur_expert;
      } else {
        moe_out = ne_add(ctx0, moe_out, cur_expert);
        ne_set_name(moe_out, ("ffn_moe_out"));
      }
    }
    // input for next layer
    cur = moe_out;
    cur = ne_rms_norm(ctx0, cur, hparams.norm_eps);
    cur = ne_mul(ctx0, cur, model.layers[il].norm[3]);
    inpL = ne_add(ctx0, cur, inpL);
  }
  lctx.use_buf(ctx0, 0);
  // used at the end to optionally extract the embeddings
  struct ne_tensor* embeddings = nullptr;
  // norm
  {
    inpL = ne_rms_norm(ctx0, inpL, hparams.norm_eps);
    inpL = ne_mul(ctx0, inpL, model.others[1]);
  }
  lctx.use_buf(ctx0, -1);
  // hidden_states = self.ln_f(hidden_states)&lm_head
  { inpL = ne_mul_mat(ctx0, model.others[2], inpL); }
  inpL = ne_scale(ctx0, inpL, ne_new_f32(ctx0, 0.5773502691896257f));

  // logits -> probs
  // inpL = ne_soft_max_inplace(ctx0, inpL);

  // run the computation
  ne_build_forward_expand(&gf, inpL);
  ne_graph_compute(ctx0, &gf);

  if (ns_log_level() == 0 || ns_log_level() == 2) {
    ne_graph_profiling(&gf);
  }

  // update kv token count
  lctx.model.kv_self.n = n_past + N;

  // extract logits
  {
    auto& logits_out = lctx.logits;

    size_t bs_stride = n_vocab * N;
    if (lctx.logits_all) {
      logits_out.resize(n_vocab * N * batch_size);
      for (int i = 0; i < batch_size; ++i) {
        memcpy(logits_out.data() + i * bs_stride, reinterpret_cast<float*>(ne_get_data(inpL)) + (i * bs_stride),
               sizeof(float) * n_vocab * N);
      }
    } else {
      // return result for just the last token
      logits_out.resize(n_vocab * batch_size);
      for (int i = 0; i < batch_size; ++i) {
        memcpy(logits_out.data() + (i * n_vocab),
               reinterpret_cast<float*>(ne_get_data(inpL)) + (i * bs_stride) + (n_vocab * (N - 1)),
               sizeof(float) * n_vocab);
      }
    }
  }

  // extract embeddings
  if (!lctx.embedding.empty()) {
    auto& embedding_out = lctx.embedding;

    embedding_out.resize(n_embd);
    memcpy(embedding_out.data(), reinterpret_cast<float*>(ne_get_data(embeddings)) + (n_embd * (N - 1)),
           sizeof(float) * n_embd);
  }

  if (mem_per_token == 0) {
    mem_per_token = ne_used_mem(ctx0) / N;
  }

  ne_free(ctx0);

  // measure the performance only for the single-token evals
  int64_t time_interval = ne_time_us() - t_start_us;
  if (N == 1) {
    lctx.t_eval_us += time_interval;
    lctx.n_eval++;
  } else if (N > 1) {
    lctx.t_p_eval_us += time_interval;
    lctx.n_p_eval += N;
  }
  lctx.eval_times.push_back(time_interval);

  return true;
}

int model_eval(struct model_context* ctx, const model_input* inputs, const int n_input, int n_threads) {
  if (!grok_model_eval_internal(ctx, inputs, n_input, n_threads)) {
    fprintf(stderr, "%s: failed to eval\n", __func__);
    return 1;
  }

  // get a more accurate load time, upon first eval

  if (!ctx->has_evaluated_once) {
    ctx->t_load_us = ne_time_us() - ctx->t_start_us;
    ctx->has_evaluated_once = true;
  }

  return 0;
}
