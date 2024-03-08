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
#include "models/llama/llama.h"

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
#include "core/layers/mha_dense.h"
#include "core/ne.h"
#include "core/ne_bestla.h"
#include "core/ne_layers.h"
#include "models/model_utils/model_config.h"
#include "models/model_utils/model_files.h"
#include "models/model_utils/model_types.h"
#include "models/model_utils/model_utils.h"
#include "models/model_utils/util.h"
#include "models/models.h"

static const bool NE_ATTN_PREFER_FP32 =
    getenv("NE_ATTN_PREFER_FP32") != nullptr && std::string("1") == getenv("NE_ATTN_PREFER_FP32");

// evaluate the transformer
//
//   - lctx:      model context
//   - inputs:    model_input array
//   - n_input    num of model_input
//   - n_threads: number of threads to use
//
static bool llama_model_eval_internal(model_context* ctx, const model_input* inputs, const int n_input,
                                      const int n_threads) {
  const int64_t t_start_us = ne_time_us();
  model_context& lctx = *ctx;
  // single prompt
  const int N = inputs->n_tokens;
  const int n_past = inputs->n_past;
  const int n_total = inputs->n_total;

  const int batch_size = lctx.batch_size;
  MODEL_ASSERT(batch_size == n_input);
  // continuous batching (no padding)
  // input shape will be [1, l_sum]
  if (batch_size > 1)
    MODEL_ASSERT(
        ("llama arch only supports continuous batching inference when giving multi prompts.", lctx.cont_batching));
  const bool concat_multi_seqs = batch_size > 1 ? true : false;
  std::vector<int> n_tokens(batch_size);
  std::vector<int> n_pasts(batch_size);
  std::vector<int> n_totals(batch_size);
  const int beam_size = lctx.beam_search ? lctx.beam_size : 1;
  std::vector<int> block_ids(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    n_tokens[i] = inputs[i].n_tokens;
    n_pasts[i] = inputs[i].n_past;
    n_totals[i] = inputs[i].n_total;
    block_ids[i] = inputs[i].request_idx * beam_size + inputs[i].beam_idx;
    // enforce that the first token is BOS
    if (n_totals[i] == 0 && inputs[i].tokens[0] != lctx.vocab.bos_token_id) {
      fprintf(stderr, "%s: first token must be BOS (token id is %d) in %dth prompt\n", __func__,
              lctx.vocab.bos_token_id, i);
      return false;
    }
  }
  const int seq_len_sum = std::accumulate(n_tokens.begin(), n_tokens.end(), 0);
  const int infer_bs = 1;
  const int infer_seq_len = seq_len_sum;
  const int kv_n_ctx_block = lctx.kv_n_ctx_block;
  const std::vector<std::vector<int>> infer_groups = split_inputs_into_groups(inputs, n_input);

  const auto& model = lctx.model;
  const auto& hparams = model.hparams;

  const auto& kv_self = model.kv_self;

  MODEL_ASSERT(!!kv_self.ctx);

  int n_embd = hparams.n_embd;
  const int n_layer = hparams.n_layer;
  const int n_ctx = lctx.n_ctx;  // max number fo tokens to keep in the kv-cache
  const int n_keep = lctx.n_keep;
  const bool shift_roped_k = lctx.shift_roped_k;
  MODEL_ASSERT(("continuous batching mechanism doesn't support shift rope.\n", !(concat_multi_seqs && shift_roped_k)));
  // Whether kv-cache uses ring-buffer and is already full in the current run of _model_eval
  const bool is_ring_full = shift_roped_k && n_total > n_past;
  const int n_cached = shift_roped_k ? std::min(n_total + N, n_ctx) : (n_past + N);  // #tokens cached after kv-append
  int n_head = hparams.n_head;
  int head_size = n_embd / n_head;
  int n_head_kv = hparams.n_head_kv;
  int n_expert = hparams.n_experts;
  int n_expert_used = hparams.n_experts_used;

  bool enable_tp = false;
#ifdef NS_TP_MODEL
  parallel_context* p_ctx = init_parallel_context();
  int32_t world_size = get_tp_size(p_ctx);
  int32_t rank = get_tp_rank(p_ctx);
  enable_tp = world_size > 1 ? true : false;

  // after TP the Q K n_head will become 1/world_size
  if (enable_tp) {
    n_head /= world_size;
    n_head_kv /= world_size;
  }
#endif
  MODEL_ASSERT(("continuous batching mechanism doesn't support TP.\n", !(concat_multi_seqs && enable_tp)));

  const int n_vocab = hparams.n_vocab;
  const int n_rot = head_size;
  const int n_embd_gqa = head_size * n_head_kv;

  auto& mem_per_token = lctx.mem_per_token;
  auto& buf_compute = lctx.buf_compute;

  struct ne_init_params params = {
      /*.mem_size   =*/buf_compute.size,
      /*.mem_buffer =*/buf_compute.addr,
      /*.no_alloc   =*/false,
  };

  struct ne_context* ctx0 = ne_init(params);

  // for big prompts, if BLAS is enabled, it is better to use only one thread
  // otherwise, the threads are spin-lock waiting for the BLAS calls and are degrading the performance
  ne_cgraph gf = {};
  gf.n_threads = N >= 32 && ne_cpu_has_blas() ? 1 : n_threads;

  const bool run_mha_reordered = kv_self.k->type == NE_TYPE_BTLA;
  kv_cache_info_t kv_cache_info = {0, 0};
  if (run_mha_reordered) {
    NE_ASSERT(kv_self.v->type == NE_TYPE_BTLA);  // kv type should be the same
    attn_shape_t attn_shape = {
        /* .batch_size = */ batch_size,
        /* .head_num = */ n_head,
        /* .heads_kv = */ n_head_kv,
        /* .head_size = */ head_size,
        /* .sl_q = */ N,  // Note: make sure that bestla reordered attn supports next token inferencing
        /* .sl_kv = */ n_cached,
    };

    NE_ASSERT(("bestla managed kv-cache not supported; use `--memory-f16 / --memory-f32` instead",
               bestla_reordered_attn_fp32_support(&attn_shape)));
    kv_shape_t kv_shape{
        /* .heads_kv = */ static_cast<uint32_t>(n_head_kv),
        /* .head_size = */ static_cast<uint32_t>(head_size),
        /* .sl_kv_max = */ static_cast<uint32_t>(n_ctx),
    };
    bestla_reordered_attn_fp32_batch_kv_info(&kv_shape, &kv_cache_info);
  }

  struct ne_tensor* embd = ne_new_tensor_1d(ctx0, NE_TYPE_I32, seq_len_sum, NE_SIZE_CALC);
  ne_set_name(embd, "embd");
  int cpy_off = 0;
  for (int i = 0; i < batch_size; ++i) {
    memcpy(static_cast<model_token*>(embd->data) + cpy_off, inputs[i].tokens, n_tokens[i] * ne_element_size(embd));
    cpy_off += n_tokens[i];
  }

#ifdef NS_TP_MODEL
  if (enable_tp) {
    // need to broadcast the ids
    broadcast(p_ctx, reinterpret_cast<float*>(embd->data), N * ne_element_size(embd));
  }
#endif

  struct ne_tensor* inpL = ne_get_rows(ctx0, model.others[0], embd);
  for (int il = 0; il < n_layer; ++il) {
    struct ne_tensor* inpSA = inpL;

    struct ne_tensor* cur;

    lctx.use_buf(ctx0, 0);

    // norm
    {
      cur = ne_rms_norm(ctx0, inpL, hparams.norm_eps);

      // cur = cur*attention_norm(broadcasted)
      cur = ne_mul(ctx0, cur, model.layers[il].norm[0]);
    }
    ne_tensor *Qcur, *Kcur, *Vcur;
    if (bestla_fusion_QKV_f32f32_support(model.layers[il].attn[0]->data, model.layers[il].attn[1]->data,
                                         model.layers[il].attn[2]->data, seq_len_sum, model.layers[il].attn[0]->ne[1],
                                         model.layers[il].attn[0]->ne[0]) &&
        n_head == n_head_kv) {  // fused execution of QKV
      struct ne_tensor* QKVcur =
          ne_mul_qkv(ctx0, model.layers[il].attn[0], model.layers[il].attn[1], model.layers[il].attn[2], cur);
      const size_t qkv_size = head_size * n_head * seq_len_sum;
      const size_t qkv_bytes = qkv_size * ne_element_size(QKVcur);
      Qcur = ne_reshape_4d(ctx0, ne_view_1d(ctx0, QKVcur, qkv_size, 0 * qkv_bytes), head_size, n_head, infer_seq_len,
                           infer_bs);
      Kcur = ne_reshape_4d(ctx0, ne_view_1d(ctx0, QKVcur, qkv_size, 1 * qkv_bytes), head_size, n_head_kv, infer_seq_len,
                           infer_bs);
      Vcur = ne_view_1d(ctx0, QKVcur, qkv_size, 2 * qkv_bytes);
    } else {
      Qcur = ne_reshape_4d(ctx0, ne_mul_mat(ctx0, model.layers[il].attn[0], cur), head_size, n_head, infer_seq_len,
                           infer_bs);
      Kcur = ne_reshape_4d(ctx0, ne_mul_mat(ctx0, model.layers[il].attn[1], cur), head_size, n_head_kv, infer_seq_len,
                           infer_bs);
      Vcur = ne_mul_mat(ctx0, model.layers[il].attn[2], cur);
    }
    if (concat_multi_seqs) {
      size_t off_sl = 0;
      // per_request rope
      for (int gi = 0; gi < infer_groups.size(); ++gi) {
        const int qk_bs = infer_groups[gi].size();
        const int qk_sl = n_tokens[infer_groups[gi].front()];
        const int qk_n_past = n_pasts[infer_groups[gi].front()];
        struct ne_tensor* Qcur_req =
            ne_view_4d(ctx0, Qcur, head_size, n_head, qk_sl, qk_bs, ne_element_size(Qcur) * head_size,
                       ne_element_size(Qcur) * head_size * n_head, ne_element_size(Qcur) * head_size * n_head * qk_sl,
                       off_sl * n_head * ne_element_size(Qcur));
        ne_build_forward_expand(
            &gf, ne_rope_inplace(ctx0, Qcur_req, qk_n_past, n_rot, 0, 0, hparams.freq_base, hparams.freq_scale));
        struct ne_tensor* Kcur_req = ne_view_4d(
            ctx0, Kcur, head_size, n_head_kv, qk_sl, qk_bs, ne_element_size(Kcur) * head_size,
            ne_element_size(Kcur) * head_size * n_head_kv, ne_element_size(Kcur) * head_size * n_head_kv * qk_sl,
            off_sl * n_head_kv * ne_element_size(Kcur));
        ne_build_forward_expand(
            &gf, ne_rope_inplace(ctx0, Kcur_req, qk_n_past, n_rot, 0, 0, hparams.freq_base, hparams.freq_scale));
        off_sl += head_size * qk_bs * qk_sl;
      }
    } else {
      Qcur = ne_rope_inplace(ctx0, Qcur, std::max(n_cached - N, n_past), n_rot, 0, 0, hparams.freq_base,
                             hparams.freq_scale);
      Kcur = ne_rope_inplace(  // n_ctx exceeds but it will be shift-roped back with cached K
          ctx0, Kcur, (is_ring_full ? n_ctx : n_past), n_rot, 0, 0, hparams.freq_base, hparams.freq_scale);
      // Vcur = ne_transpose(ctx0, ne_reshape_2d(ctx0, Vcur, head_size * n_head_kv, N));
    }
    ne_set_name(Qcur, "Qcur");
    ne_set_name(Kcur, "Kcur");
    ne_set_name(Vcur, "Vcur");
    // self-attention
    const float attn_scale = 1.0f / sqrtf(static_cast<float>(head_size));
    struct ne_tensor* KQV_merged_contiguous =
        ne_new_tensor_2d(ctx0, NE_TYPE_F32, head_size * n_head, seq_len_sum, NE_SIZE_CALC);
    if (!run_mha_reordered) {
      // store key and value to memory
      // important:
      // 1. storing RoPE-ed version of K in the KV cache!
      // 2. for loop self-attention in multi seqs infer (num_request > 1)
      {
        struct ne_tensor* const k_cache =
            ne_view_1d(ctx0, kv_self.k, n_ctx * n_embd_gqa * kv_n_ctx_block,
                       il * n_ctx * ne_element_size(kv_self.k) * n_embd_gqa * kv_n_ctx_block);
        struct ne_tensor* const v_cache =
            ne_view_1d(ctx0, kv_self.v, n_ctx * n_embd_gqa * kv_n_ctx_block,
                       il * n_ctx * ne_element_size(kv_self.v) * n_embd_gqa * kv_n_ctx_block);
        // cache = [tokens, beams, requests, layers],
        // tokens = [head_dim, head_num, n_ctx] (may different orders)
        size_t off_N_i = 0;
        for (int i = 0; i < batch_size; ++i) {
          const int block_idx = block_ids[i];
          const int N_i = n_tokens[i];
          const int n_past_i = n_pasts[i];
          // batch K
          struct ne_tensor* Kcur_bs_i =
              ne_permute(ctx0,
                         ne_view_4d(ctx0, Kcur, head_size, n_head_kv, N_i, 1, ne_element_size(Kcur) * head_size,
                                    ne_element_size(Kcur) * n_embd_gqa, ne_element_size(Kcur) * n_embd_gqa * N_i,
                                    ne_element_size(Kcur) * off_N_i),
                         0, 2, 1, 3);
          struct ne_tensor* k_bs_i =
              ne_view_4d(ctx0, k_cache, head_size, N_i, n_head_kv, 1, ne_element_size(k_cache) * head_size,
                         ne_element_size(k_cache) * head_size * n_ctx, ne_element_size(k_cache) * n_embd_gqa * n_ctx,
                         block_idx * n_ctx * n_embd_gqa * ne_element_size(k_cache) +
                             head_size * n_past_i * ne_element_size(k_cache));
          // batch V
          struct ne_tensor* Vcur_bs_i =
              ne_permute(ctx0,
                         ne_reshape_4d(ctx0,
                                       ne_view_2d(ctx0, Vcur, n_embd_gqa, N_i, ne_element_size(Vcur) * n_embd_gqa,
                                                  ne_element_size(Vcur) * off_N_i),
                                       head_size, n_head_kv, N_i, 1),
                         1, 2, 0, 3);
          struct ne_tensor* v_bs_i = ne_view_4d(
              ctx0, v_cache, N_i, head_size, n_head_kv, 1, n_ctx * ne_element_size(v_cache),
              n_ctx * ne_element_size(v_cache) * head_size, n_ctx * ne_element_size(v_cache) * n_embd_gqa,
              block_idx * n_ctx * n_embd_gqa * ne_element_size(v_cache) + n_past_i * ne_element_size(v_cache));
          // concat
          ne_build_forward_expand(&gf, ne_cpy(ctx0, Kcur_bs_i, k_bs_i));
          ne_build_forward_expand(&gf, ne_cpy(ctx0, Vcur_bs_i, v_bs_i));
          off_N_i += head_size * n_head_kv * N_i;
        }
      }

      // for-loop attention
      size_t off_sl = 0;
      for (int gi = 0; gi < infer_groups.size(); ++gi) {
        const int attn_bs = infer_groups[gi].size();
        const int attn_sl = n_tokens[infer_groups[gi].front()];
        const int attn_block_id = block_ids[infer_groups[gi].front()];
        const int attn_n_past = n_pasts[infer_groups[gi].front()];
        const int attn_n_total = n_totals[infer_groups[gi].front()];
        struct ne_tensor* Q =
            ne_permute(ctx0,
                       ne_view_4d(ctx0, Qcur, head_size, n_head, attn_sl, attn_bs, ne_element_size(Qcur) * head_size,
                                  ne_element_size(Qcur) * head_size * n_head,
                                  ne_element_size(Qcur) * head_size * n_head * attn_sl, off_sl * ne_element_size(Qcur)),
                       0, 2, 1, 3);
        std::string suffix = std::to_string(gi);
        ne_set_name(Q, std::string("Q_" + suffix).c_str());
        const int n_cached_gi = shift_roped_k ? n_cached : attn_n_past + attn_sl;
        std::vector<int> attn_block_ids(infer_groups[gi].size());
        for (int j = 0; j < infer_groups[gi].size(); ++j) {
          attn_block_ids[j] = block_ids[infer_groups[gi][j]];
        }
        struct ne_tensor* K =
            model_kv_cache_seq_concat(&gf, &lctx, ctx0, head_size, n_cached_gi, n_head_kv, attn_bs, attn_block_ids, il);
        if (is_ring_full) {
          K = ne_permute(ctx0, K, 0, 2, 1, 3);
          struct ne_tensor* cossin_cache = nullptr;
          // Currently we only cache cossin for N == 1 in model-wide; It may be worthwhile to cache cossin for other N
          // in a single eval execution
          if (N == 1) cossin_cache = kv_self.cossin;
          K = ne_rope_shift_inplace(ctx0, K, -N, n_rot, 0, 0, n_keep, cossin_cache, hparams.freq_base,
                                    hparams.freq_scale);
          K = ne_permute(ctx0, K, 0, 2, 1, 3);
        }

        // split cached V into n_head heads
        struct ne_tensor* V = model_kv_cache_seq_concat(&gf, &lctx, ctx0, n_cached_gi, head_size, n_head_kv, attn_bs,
                                                        attn_block_ids, il, false);
        ne_set_name(K, std::string("K_" + suffix).c_str());
        ne_set_name(V, std::string("V_" + suffix).c_str());

        // K * Q
        struct ne_tensor* KQ = ne_mul_mat(ctx0, K, Q);
        ne_set_name(KQ, std::string("KQ_" + suffix).c_str());

        // KQ_scaled = KQ / sqrt(n_embd/n_head)
        struct ne_tensor* KQ_scale = ne_new_f32(ctx0, attn_scale);
        ne_set_name(KQ_scale, std::string("1/sqrt(n_embd/n_head)_" + suffix).c_str());

        // KQ_scaled shape [n_cached, N, n_head, 1]
        struct ne_tensor* KQ_scaled = ne_scale_inplace(ctx0, KQ, KQ_scale);
        ne_set_name(KQ_scaled, std::string("KQ_scaled_" + suffix).c_str());

        // KQ_masked = mask_past(KQ_scaled)
        if (N > 1 || !shift_roped_k || attn_n_total == 0) {  // TODO(Yi): shift roped-k with N > 1 next-token
          KQ_scaled = ne_diag_mask_inf_inplace(ctx0, KQ_scaled, attn_n_past);
          ne_set_name(KQ_scaled, std::string("KQ_masked_" + suffix).c_str());
        }

        // KQ = soft_max(KQ_masked)
        struct ne_tensor* KQ_soft_max = ne_soft_max_inplace(ctx0, KQ_scaled);
        ne_set_name(KQ_soft_max, std::string("KQ_soft_max_" + suffix).c_str());

        struct ne_tensor* KQV = ne_mul_mat(ctx0, V, KQ_soft_max);
        ne_set_name(KQV, std::string("KQV_" + suffix).c_str());

        // KQV_merged = KQV.permute(0, 2, 1, 3)
        struct ne_tensor* KQV_merged_gi = ne_permute(ctx0, KQV, 0, 2, 1, 3);
        ne_set_name(KQV_merged_gi, std::string("KQV_merged_" + suffix).c_str());

        ne_build_forward_expand(&gf,
                                ne_cpy(ctx0, KQV_merged_gi,
                                       ne_view_2d(ctx0, KQV_merged_contiguous, head_size * n_head, attn_sl * attn_bs,
                                                  head_size * n_head * ne_element_size(KQV_merged_contiguous),
                                                  ne_element_size(KQV_merged_contiguous) * off_sl)));
        off_sl += head_size * n_head * attn_sl * attn_bs;
      }
      ne_set_name(KQV_merged_contiguous, "KQV_merged_contiguous");
      // projection (no bias)
      cur = ne_mul_mat(ctx0, model.layers[il].attn[3], KQV_merged_contiguous);
    } else {
      const auto k_size = kv_cache_info.k_bytes;
      const auto v_size = kv_cache_info.v_bytes;
      // store key and value to memory
      {
        size_t off_sl = 0;
        for (int gi = 0; gi < infer_groups.size(); ++gi) {
          const int update_bs = infer_groups[gi].size();
          const int update_sl = n_tokens[infer_groups[gi].front()];
          const int update_block_id = block_ids[infer_groups[gi].front()];
          const int update_n_past = n_pasts[infer_groups[gi].front()];
          const auto k_cache_g = ne_view_4d(ctx0, kv_self.k,                         // tensor
                                            head_size, n_ctx, n_head_kv, update_bs,  // ne
                                            0, 0, k_size,                            // nb (bestla managed)
                                            il * kv_n_ctx_block * k_size + update_block_id * k_size);  // offset
          const auto k_cur_g =
              ne_view_4d(ctx0, Kcur, head_size, n_head_kv, update_sl, update_bs, ne_element_size(Kcur) * head_size,
                         ne_element_size(Kcur) * n_embd_gqa, ne_element_size(Kcur) * n_embd_gqa * update_sl,
                         ne_element_size(Kcur) * off_sl);
          ne_build_forward_expand(&gf, ne_flash_attn_update_k(ctx0, k_cache_g, k_cur_g, update_n_past, is_ring_full));
          struct ne_tensor* v_cache_g =
              ne_view_4d(ctx0, kv_self.v,                                           // tensor
                         head_size, n_ctx, n_head_kv, update_bs,                    // ne
                         0, 0, v_size,                                              // nb (bestla managed)
                         il * kv_n_ctx_block * v_size + update_block_id * v_size);  // offset);
          // bestla always view V as (D, n_head, seq, bs)
          const auto v_cur_g =
              ne_view_4d(ctx0, Vcur, head_size, n_head_kv, update_sl, update_bs, ne_element_size(Vcur) * head_size,
                         ne_element_size(Vcur) * n_embd_gqa, ne_element_size(Vcur) * n_embd_gqa * update_sl,
                         ne_element_size(Vcur) * off_sl);
          ne_build_forward_expand(&gf, ne_flash_attn_update_v(ctx0, v_cache_g, v_cur_g, update_n_past, is_ring_full));
          off_sl += n_embd_gqa * update_sl * update_bs;
        }
      }

      // for-loop attention
      size_t off_sl = 0;
      for (int gi = 0; gi < infer_groups.size(); ++gi) {
        const int attn_bs = infer_groups[gi].size();
        const int attn_sl = n_tokens[infer_groups[gi].front()];
        const int attn_block_id = block_ids[infer_groups[gi].front()];
        const int attn_n_past = n_pasts[infer_groups[gi].front()];
        const int attn_n_total = n_totals[infer_groups[gi].front()];
        struct ne_tensor* Q =
            ne_permute(ctx0,
                       ne_view_4d(ctx0, Qcur, head_size, n_head, attn_sl, attn_bs, ne_element_size(Qcur) * head_size,
                                  ne_element_size(Qcur) * head_size * n_head,
                                  ne_element_size(Qcur) * head_size * n_head * attn_sl, off_sl * ne_element_size(Qcur)),
                       0, 2, 1, 3);
        std::string suffix = std::to_string(gi);
        ne_set_name(Q, std::string("Q_" + suffix).c_str());
        const int n_cached_gi = shift_roped_k ? n_cached : attn_n_past + attn_sl;
        struct ne_tensor* K =
            ne_view_4d(ctx0, kv_self.k,                                                     // tensor
                       head_size, n_cached_gi, n_head_kv, attn_bs,                          // ne
                       kv_cache_info.stride_k_sl, kv_cache_info.stride_k_head_num, k_size,  // nb (bestla managed)
                       il * kv_n_ctx_block * k_size + attn_block_id * k_size);              // offset
        *reinterpret_cast<ATTN_FWD_LAYOUT*>(&K->nb[0]) = kv_cache_info.k_layout;            // use nb0 for layout
        if (is_ring_full) {
          struct ne_tensor* cossin_cache = nullptr;
          // Currently we only cache cossin for N == 1 in model-wide; It may be worthwhile to cache cossin for other N
          // in a single eval execution
          if (N == 1) cossin_cache = kv_self.cossin;
          K = ne_rope_shift_inplace(ctx0, K, -N, n_rot, 0, 0, n_keep, cossin_cache, hparams.freq_base,
                                    hparams.freq_scale);
        }
        struct ne_tensor* V = ne_view_4d(ctx0, kv_self.v,                             // tensor
                                         n_cached_gi, head_size, n_head_kv, attn_bs,  // ne
                                         kv_cache_info.stride_v_head_size, kv_cache_info.stride_v_head_num,
                                         v_size,                                                  // nb (bestla managed)
                                         il * kv_n_ctx_block * v_size + attn_block_id * v_size);  // use nb0 for layout
        *reinterpret_cast<ATTN_FWD_LAYOUT*>(&V->nb[0]) = kv_cache_info.v_layout;
        ne_set_name(K, std::string("K_" + suffix).c_str());
        ne_set_name(V, std::string("V_" + suffix).c_str());

        ne_attn_flags_t attn_flags = NE_ATTN_FLAG_NONE;
        if (NE_ATTN_PREFER_FP32) attn_flags |= NE_ATTN_FLAG_PREFER_FP32;
        if (n_total == 0 || !shift_roped_k) attn_flags |= NE_ATTN_FLAG_IS_CAUSAL;  // no causal mask on next-token cases
        struct ne_tensor* KQV_Out = ne_flash_attn(ctx0, Q, K, V, attn_scale, attn_flags);
        struct ne_tensor* KQV_merged_gi = ne_view_2d(ctx0, KQV_Out, head_size * n_head, attn_sl * attn_bs,
                                                     head_size * n_head * ne_element_size(KQV_Out), 0);
        ne_set_name(KQV_merged_gi, std::string("KQV_merged_" + suffix).c_str());
        ne_build_forward_expand(&gf,
                                ne_cpy(ctx0, KQV_merged_gi,
                                       ne_view_2d(ctx0, KQV_merged_contiguous, head_size * n_head, attn_sl * attn_bs,
                                                  head_size * n_head * ne_element_size(KQV_merged_contiguous),
                                                  ne_element_size(KQV_merged_contiguous) * off_sl)));
        off_sl += head_size * n_head * attn_sl * attn_bs;
      }
      ne_set_name(KQV_merged_contiguous, "KQV_merged_contiguous");
      // projection (no bias)
      cur = ne_mul_mat(ctx0, model.layers[il].attn[3], KQV_merged_contiguous);
    }
#ifdef NS_TP_MODEL
    if (enable_tp) {
      cur = ne_all_reduce(ctx0, cur);
    }
#endif

    lctx.use_buf(ctx0, 1);

    struct ne_tensor* inpFF = ne_add(ctx0, cur, inpSA);

    // feed-forward network
    {
      // norm
      {
        cur = ne_rms_norm(ctx0, inpFF, hparams.norm_eps);

        // cur = cur*ffn_norm(broadcasted)
        cur = ne_mul(ctx0, cur, model.layers[il].norm[1]);
      }
      if (n_expert == 0) {
        if (bestla_fusion_FFN_SiLu_f32f32_support(model.layers[il].ffn[0]->data, model.layers[il].ffn[1]->data,
                                                  model.layers[il].ffn[2]->data, seq_len_sum, cur->ne[0],
                                                  model.layers[il].ffn[0]->ne[1], model.layers[il].ffn[1]->ne[1])) {
          cur = ne_ffn_silu(ctx0, model.layers[il].ffn[0], model.layers[il].ffn[1], model.layers[il].ffn[2], cur);
        } else {
          struct ne_tensor* tmp = ne_mul_mat(ctx0, model.layers[il].ffn[2], cur);
          cur = ne_mul_mat(ctx0, model.layers[il].ffn[0], cur);
          cur = ne_silu(ctx0, cur);
          cur = ne_mul(ctx0, cur, tmp);
          cur = ne_mul_mat(ctx0, model.layers[il].ffn[1], cur);
        }
      } else {
        // for-loop MOE (deal with sequence one by one)
        struct ne_tensor* moe_out = ne_new_tensor_2d(ctx0, NE_TYPE_F32, head_size * n_head, seq_len_sum, NE_SIZE_CALC);
        size_t off_sl = 0;
        for (int bi = 0; bi < batch_size; ++bi) {
          const int moe_sl = n_tokens[bi];
          struct ne_tensor* cur_seq =
              ne_view_2d(ctx0, cur, head_size * n_head, moe_sl, head_size * n_head * ne_element_size(cur),
                         ne_element_size(cur) * off_sl);
          std::string suffix = std::to_string(bi);
          ne_tensor* logits = ne_mul_mat(ctx0, model.layers[il].ffn_gate_inp, cur_seq);  // [n_tokens, num_experts]
          ne_tensor* probs = ne_soft_max_inplace(ctx0, logits);
          ne_tensor* selected_experts = ne_top_k(ctx0, probs, n_expert_used);
          ne_tensor* weights = ne_get_rows(ctx0, ne_reshape_3d(ctx0, probs, 1, n_expert, moe_sl), selected_experts);
          weights = ne_reshape_2d(ctx0, weights, n_expert_used, moe_sl);
          ne_tensor* weights_sum = ne_sum_rows(ctx0, weights);
          weights_sum = ne_repeat(ctx0, weights_sum, weights);
          weights = ne_div(ctx0, weights, weights_sum);
          ne_tensor* moe_out_i = nullptr;

          for (int i = 0; i < n_expert_used; ++i) {
            ne_tensor* cur_expert;
            if (moe_sl == 1 && bestla_fusion_FFN_SiLu_f32f32_support(
                                   model.layers[il].ffn_gate_exp[0]->data, model.layers[il].ffn_down_exp[0]->data,
                                   model.layers[il].ffn_up_exp[0]->data, moe_sl, cur_seq->ne[0],
                                   model.layers[il].ffn_gate_exp[0]->ne[1], model.layers[il].ffn_down_exp[0]->ne[1])) {
              cur_expert = ne_mul_id_ffn_silu(ctx0, model.layers[il].ffn_down_exp, model.layers[il].ffn_gate_exp,
                                              model.layers[il].ffn_up_exp, n_expert, selected_experts, i, cur_seq);
            } else {
              ne_tensor* cur_up =
                  ne_mul_mat_id(ctx0, model.layers[il].ffn_up_exp, n_expert, selected_experts, i, cur_seq);
              ne_set_name(cur_up, std::string("ffn_moe_up_" + suffix).c_str());

              ne_tensor* cur_gate =
                  ne_mul_mat_id(ctx0, model.layers[il].ffn_gate_exp, n_expert, selected_experts, i, cur_seq);
              ne_set_name(cur_gate, std::string("ffn_moe_gate_" + suffix).c_str());

              cur_gate = ne_silu(ctx0, cur_gate);
              ne_set_name(cur_gate, std::string("ffn_moe_silu_" + suffix).c_str());

              cur_expert = ne_mul(ctx0, cur_up, cur_gate);  // [n_tokens, n_embd]
              ne_set_name(cur_expert, std::string("ffn_moe_gate_par_" + suffix).c_str());

              cur_expert = ne_mul_mat_id(ctx0, model.layers[il].ffn_down_exp, n_expert, selected_experts, i,
                                         cur_expert);  // [n_tokens, n_embd]
              ne_set_name(cur_expert, std::string("ffn_moe_down_" + suffix).c_str());
            }

            cur_expert = ne_mul(
                ctx0, cur_expert,
                ne_repeat(ctx0, ne_view_2d(ctx0, weights, 1, moe_sl, weights->nb[1], i * weights->nb[0]), cur_expert));
            ne_set_name(cur_expert, std::string("ffn_moe_weighted_" + suffix).c_str());

            if (i == 0) {
              moe_out_i = cur_expert;
            } else {
              moe_out_i = ne_add(ctx0, moe_out_i, cur_expert);
              ne_set_name(moe_out_i, std::string("ffn_moe_out_" + suffix).c_str());
            }
          }
          ne_build_forward_expand(&gf, ne_cpy(ctx0, moe_out_i,
                                              ne_view_2d(ctx0, moe_out, head_size * n_head, moe_sl,
                                                         head_size * n_head * ne_element_size(moe_out),
                                                         ne_element_size(moe_out) * off_sl)));
          off_sl += head_size * n_head * moe_sl;
        }

        cur = moe_out;
      }
#ifdef NS_TP_MODEL
      // ffn2 and ffn0 use split row, ffn1 use split column
      if (enable_tp) {
        cur = ne_all_reduce(ctx0, cur);
      }
#endif
    }

    cur = ne_add(ctx0, cur, inpFF);

    // input for next layer
    inpL = cur;
  }

  lctx.use_buf(ctx0, 0);

  // used at the end to optionally extract the embeddings
  struct ne_tensor* embeddings = nullptr;
  // norm
  {
    inpL = ne_rms_norm(ctx0, inpL, hparams.norm_eps);

    // inpL = inpL*norm(broadcasted)
    inpL = ne_mul(ctx0, inpL, model.others[1]);

    embeddings = inpL;
  }

  // lm_head
  inpL = ne_mul_mat(ctx0, model.others[2], inpL);

  lctx.use_buf(ctx0, -1);

  // logits -> probs
  // inpL = ne_soft_max_inplace(ctx0, inpL);

  // run the computation
  ne_build_forward_expand(&gf, inpL);
  ne_graph_compute(ctx0, &gf);

  if (ns_log_level() == 0 || ns_log_level() == 2) {
    ne_graph_profiling(&gf);
  }

  // update kv token count
  lctx.model.kv_self.n = n_cached;

  // extract logits
  {
    auto& logits_out = lctx.logits;

    if (lctx.logits_all) {
      logits_out.resize(n_vocab * seq_len_sum);
      memcpy(logits_out.data(), reinterpret_cast<float*>(ne_get_data(inpL)), sizeof(float) * n_vocab * seq_len_sum);
    } else {
      // return result for just the last token
      logits_out.resize(n_vocab * batch_size);
#pragma omp parallel for
      for (int i = 0; i < batch_size; ++i) {
        size_t bs_off = std::accumulate(n_tokens.begin(), n_tokens.begin() + i, 0) * n_vocab;
        memcpy(logits_out.data() + (i * n_vocab),
               reinterpret_cast<float*>(ne_get_data(inpL)) + bs_off + (n_vocab * (n_tokens[i] - 1)),
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
  if (!llama_model_eval_internal(ctx, inputs, n_input, n_threads)) {
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
