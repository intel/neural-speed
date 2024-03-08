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
#include <iostream>
#include <iterator>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/data_types.h"
#include "core/layers/mha_dense.h"
#include "core/ne.h"
#include "core/ne_layers.h"
#include "core/ne_bestla.h"
#include "models/model_utils/model_config.h"
#include "models/model_utils/model_utils.h"
#include "models/model_utils/util.h"

#define MHA_FUSION 0  //  turn it off for naive beam_search kv cache reorder
#define MHA_FP16 (MHA_FUSION && 0)

static const bool NE_ATTN_PREFER_FP32 =
    getenv("NE_ATTN_PREFER_FP32") != nullptr && std::string("1") == getenv("NE_ATTN_PREFER_FP32");

// evaluate the transformer
//
//   - lctx:      model context
//   - inputs:    model_input array
//   - n_input    num of model_input
//   - n_threads: number of threads to use
//
static bool gptj_model_eval_internal(model_context* ctx, const model_input* inputs, const int n_input,
                                     const int n_threads) {
  const int64_t t_start_us = ne_time_us();
  model_context& lctx = *ctx;

  const int batch_size = lctx.batch_size;  // num of beams of all batches
  MODEL_ASSERT(batch_size == n_input);
  // if each sequence length l_1 = l_2 = l_i
  // input shape will be [B, l_i]
  const int N = inputs->n_tokens;
  const int n_past = inputs->n_past;
  const int n_total = inputs->n_total;
  // continuous batching (no padding)
  // if each sequence length l_i ! = l_k
  // input shape will be [1, l_sum]
  const bool concat_multi_seqs = (batch_size > 1 && lctx.cont_batching) ? true : false;
  std::vector<int> n_tokens(batch_size);
  std::vector<int> n_pasts(batch_size);
  std::vector<int> n_totals(batch_size);
  const int beam_size = lctx.beam_search ? lctx.beam_size : 1;
  std::vector<int> block_ids(batch_size);
  std::vector<int> n_padding;
  bool no_padding = true;
  for (int i = 0; i < batch_size; ++i) {
    n_tokens[i] = inputs[i].n_tokens;
    n_pasts[i] = inputs[i].n_past;
    n_totals[i] = inputs[i].n_total;
    block_ids[i] = inputs[i].request_idx * beam_size + inputs[i].beam_idx;
    if (!concat_multi_seqs) {
      n_padding.push_back(inputs[i].n_padding);
      if (no_padding && inputs[i].n_padding != 0) no_padding = false;
    }
  }
  const int seq_len_sum = std::accumulate(n_tokens.begin(), n_tokens.end(), 0);
  if (!concat_multi_seqs) MODEL_ASSERT(seq_len_sum == N * batch_size);
  const int infer_bs = concat_multi_seqs ? 1 : batch_size;
  const int infer_seq_len = concat_multi_seqs ? seq_len_sum : N;
  const std::vector<std::vector<int>> infer_groups = split_inputs_into_groups(inputs, n_input);
  const auto& model = lctx.model;
  const auto& hparams = model.hparams;

  const auto& kv_self = model.kv_self;

  MODEL_ASSERT(!!kv_self.ctx);

  const int n_embd = hparams.n_embd;
  const int n_layer = hparams.n_layer;
  const int n_ctx = lctx.n_ctx;  // max number fo tokens to keep in the kv-cache
  const int n_keep = lctx.n_keep;
  const bool shift_roped_k = lctx.shift_roped_k;
  MODEL_ASSERT(("continuous batching mechanism doesn't support shift rope.\n", !(concat_multi_seqs && shift_roped_k)));
  const bool is_ring_full = shift_roped_k && n_total > n_past;
  const int n_cached = shift_roped_k ? std::min(n_total + N, n_ctx) : (n_past + N);  // #tokens cached after kv-append
  int n_head = hparams.n_head;
  const int head_size = n_embd / n_head;
  const int n_vocab = hparams.n_vocab;
  const int n_rot = hparams.n_rot;

  bool enable_tp = false;
#ifdef NS_TP_MODEL
  parallel_context* p_ctx = init_parallel_context();
  int32_t world_size = get_tp_size(p_ctx);
  int32_t rank = get_tp_rank(p_ctx);
  enable_tp = world_size > 1 ? true : false;
  // IMPORTANT, when TP, the n_head will 1 / world_size
  if (enable_tp) {
    n_head /= world_size;
  }
#endif

  MODEL_ASSERT(("continuous batching mechanism doesn't support TP.\n", !(concat_multi_seqs && enable_tp)));
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
  gf.n_threads = n_threads;

  // no padding input for optimized MHA kernel
  const bool run_mha_reordered = (kv_self.k->type == NE_TYPE_BTLA);
  const bool run_mha_fp16 = !run_mha_reordered && MHA_FP16 && bestla_fusion_attn_fp16_support(nullptr);
  const bool run_mha_bf16_first =
      !run_mha_reordered && MHA_FUSION && !MHA_FP16 && bestla_fusion_attn_fp32_fp16_fp16_fp32_support(nullptr);
  kv_cache_info_t kv_cache_info = {0, 0};
  if (run_mha_reordered) {
    NE_ASSERT(kv_self.v->type == NE_TYPE_BTLA);  // kv type should be the same
    attn_shape_t attn_shape = {
        /* .batch_size = */ batch_size,
        /* .head_num = */ n_head,
        /* .heads_kv = */ n_head,  // GPT-J does not have MQA/GQA
        /* .head_size = */ head_size,
        /* .sl_q = */ N,  // Note: make sure that bestla reordered attn supports next token inference
        /* .sl_kv = */ n_cached,
    };
    NE_ASSERT(("bestla managed kv-cache not supported; use `--memory-f16 / --memory-f32` instead",
               bestla_reordered_attn_fp32_support(&attn_shape)));
    kv_shape_t kv_shape{
        /* .head_num = */ static_cast<uint32_t>(n_head),
        /* .head_size = */ static_cast<uint32_t>(head_size),
        /* .sl_kv_max = */ static_cast<uint32_t>(n_ctx),
    };
    bestla_reordered_attn_fp32_batch_kv_info(&kv_shape, &kv_cache_info);
  }

  struct ne_tensor* embd = d_ne_new_tensor_1d(ctx0, NE_TYPE_I32, seq_len_sum);
  ne_set_name(embd, "embd");
  int cpy_off = 0;
  for (int i = 0; i < batch_size; ++i) {
    memcpy(static_cast<model_token*>(embd->data) + cpy_off, inputs[i].tokens, n_tokens[i] * ne_element_size(embd));
    cpy_off += n_tokens[i];
  }

  struct ne_tensor* inpL = ne_get_rows(ctx0, model.others[0], embd);

  for (int il = 0; il < n_layer; ++il) {
    struct ne_tensor* cur;

    lctx.use_buf(ctx0, 0);

    // norm
    cur = ne_norm(ctx0, inpL, hparams.norm_eps);

    // cur = ln_1_g*cur + ln_1_b
    cur = ne_add(ctx0, ne_mul(ctx0, cur, model.layers[il].norm[0]), model.layers[il].norm[1]);

    struct ne_tensor* inpSA = cur;

    ne_tensor *Qcur, *Kcur, *Vcur;
    int kv_n_ctx_block = lctx.kv_n_ctx_block;
    if (bestla_fusion_QKV_f32f32_support(model.layers[il].attn[0]->data, model.layers[il].attn[1]->data,
                                         model.layers[il].attn[2]->data, seq_len_sum, head_size * n_head,
                                         head_size * n_head)) {  // fused execution of QKV
                                                                 // if (false) {
      struct ne_tensor* QKVcur =
          ne_mul_qkv(ctx0, model.layers[il].attn[0], model.layers[il].attn[1], model.layers[il].attn[2], cur);
      const size_t qkv_size = head_size * n_head * seq_len_sum;
      const size_t qkv_bytes = qkv_size * ne_element_size(QKVcur);
      Qcur = ne_reshape_4d(ctx0, ne_view_1d(ctx0, QKVcur, qkv_size, 0 * qkv_bytes), head_size, n_head, infer_seq_len,
                           infer_bs);
      Kcur = ne_reshape_4d(ctx0, ne_view_1d(ctx0, QKVcur, qkv_size, 1 * qkv_bytes), head_size, n_head, infer_seq_len,
                           infer_bs);
      Vcur = ne_view_1d(ctx0, QKVcur, qkv_size, 2 * qkv_bytes);
    } else {
      Qcur = ne_reshape_4d(ctx0, ne_mul_mat(ctx0, model.layers[il].attn[0], cur), head_size, n_head, infer_seq_len,
                           infer_bs);
      Kcur = ne_reshape_4d(ctx0, ne_mul_mat(ctx0, model.layers[il].attn[1], cur), head_size, n_head, infer_seq_len,
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
                       off_sl * ne_element_size(Qcur));
        ne_build_forward_expand(
            &gf, ne_rope_inplace(ctx0, Qcur_req, qk_n_past, n_rot, 0, 0, hparams.freq_base, hparams.freq_scale));
        struct ne_tensor* Kcur_req =
            ne_view_4d(ctx0, Kcur, head_size, n_head, qk_sl, qk_bs, ne_element_size(Kcur) * head_size,
                       ne_element_size(Kcur) * head_size * n_head, ne_element_size(Kcur) * head_size * n_head * qk_sl,
                       off_sl * ne_element_size(Kcur));
        ne_build_forward_expand(
            &gf, ne_rope_inplace(ctx0, Kcur_req, qk_n_past, n_rot, 0, 0, hparams.freq_base, hparams.freq_scale));
        off_sl += head_size * n_head * qk_bs * qk_sl;
      }
    } else {
      Qcur = ne_rope_inplace(ctx0, Qcur, std::max(n_cached - N, n_past), n_rot, 0, 0, hparams.freq_base,
                             hparams.freq_scale);
      Kcur = ne_rope_inplace(  // n_ctx exceeds but it will be shift-roped back with cached K
          ctx0, Kcur, (is_ring_full ? n_ctx : n_past), n_rot, 0, 0, hparams.freq_base, hparams.freq_scale);
    }
    ne_set_name(Qcur, "Qcur");
    ne_set_name(Kcur, "Kcur");
    ne_set_name(Vcur, "Vcur");
    // self-attention
    // store key and value to memory
    // important:
    // 1. storing RoPE-ed version of K in the KV cache!
    // 2. for loop self-attention in multi seqs infer (num_request > 1)
    if (!run_mha_reordered) {
      struct ne_tensor* const k_cache =
          ne_view_1d(ctx0, kv_self.k, n_ctx * head_size * n_head * kv_n_ctx_block,
                     il * n_ctx * ne_element_size(kv_self.k) * head_size * n_head * kv_n_ctx_block);
      struct ne_tensor* const v_cache =
          ne_view_1d(ctx0, kv_self.v, n_ctx * head_size * n_head * kv_n_ctx_block,
                     il * n_ctx * ne_element_size(kv_self.v) * head_size * n_head * kv_n_ctx_block);
      std::vector<ne_tensor*> Kcur_bs(batch_size);
      std::vector<ne_tensor*> Vcur_bs(batch_size);
      std::vector<ne_tensor*> k_bs(batch_size);
      std::vector<ne_tensor*> v_bs(batch_size);
      // cache = [tokens, beams, requests, layers],
      // tokens = [head_dim, head_num, n_ctx] (may different orders)
      size_t off_N_i = 0;
      for (int i = 0; i < batch_size; ++i) {
        const int block_idx = block_ids[i];
        const int N_i = n_tokens[i];
        const int n_past_i = n_pasts[i];
        if (run_mha_fp16) {
          // batch V
          Vcur_bs[i] = ne_view_4d(ctx0, Vcur, head_size, n_head, N_i, 1, ne_element_size(Vcur) * head_size,
                                  ne_element_size(Vcur) * head_size * n_head,
                                  ne_element_size(Vcur) * head_size * n_head * N_i, ne_element_size(Vcur) * off_N_i);
          v_bs[i] = ne_view_1d(ctx0, v_cache, head_size * n_head * N_i * 1,
                               ne_element_size(v_cache) * head_size * n_head * n_past_i +
                                   block_idx * n_ctx * head_size * n_head * ne_element_size(v_cache));
          // batch K
          Kcur_bs[i] = ne_permute(
              ctx0,
              ne_reshape_4d(ctx0,
                            ne_view_2d(ctx0, Kcur, head_size * n_head, N_i, ne_element_size(Kcur) * head_size * n_head,
                                       ne_element_size(Kcur) * off_N_i),
                            head_size, n_head, N_i, 1),
              1, 2, 0, 3);
          k_bs[i] = ne_view_4d(
              ctx0, k_cache, N_i, head_size, n_head, 1, n_ctx * ne_element_size(k_cache),
              n_ctx * ne_element_size(k_cache) * head_size, n_ctx * ne_element_size(k_cache) * head_size * n_head,
              block_idx * n_ctx * head_size * n_head * ne_element_size(k_cache) + n_past_i * ne_element_size(k_cache));
        } else {
          // batch K
          Kcur_bs[i] =
              ne_permute(ctx0,
                         ne_view_4d(ctx0, Kcur, head_size, n_head, N_i, 1, ne_element_size(Kcur) * head_size,
                                    ne_element_size(Kcur) * head_size * n_head,
                                    ne_element_size(Kcur) * head_size * n_head * N_i, ne_element_size(Kcur) * off_N_i),
                         0, 2, 1, 3);
          k_bs[i] = ne_view_4d(ctx0, k_cache, head_size, N_i, n_head, 1, ne_element_size(k_cache) * head_size,
                               ne_element_size(k_cache) * head_size * n_ctx,
                               ne_element_size(k_cache) * head_size * n_head * n_ctx,
                               block_idx * n_ctx * head_size * n_head * ne_element_size(k_cache) +
                                   head_size * n_past_i * ne_element_size(k_cache));

          // batch V
          Vcur_bs[i] = ne_permute(
              ctx0,
              ne_reshape_4d(ctx0,
                            ne_view_2d(ctx0, Vcur, head_size * n_head, N_i, ne_element_size(Vcur) * head_size * n_head,
                                       ne_element_size(Vcur) * off_N_i),
                            head_size, n_head, N_i, 1),
              1, 2, 0, 3);
          v_bs[i] = ne_view_4d(
              ctx0, v_cache, N_i, head_size, n_head, 1, n_ctx * ne_element_size(v_cache),
              n_ctx * ne_element_size(v_cache) * head_size, n_ctx * ne_element_size(v_cache) * head_size * n_head,
              block_idx * n_ctx * head_size * n_head * ne_element_size(v_cache) + n_past_i * ne_element_size(v_cache));
        }
        ne_build_forward_expand(&gf, ne_cpy(ctx0, Kcur_bs[i], k_bs[i]));
        ne_build_forward_expand(&gf, ne_cpy(ctx0, Vcur_bs[i], v_bs[i]));
        off_N_i += head_size * n_head * N_i;
      }
    } else {
      const auto k_size = kv_cache_info.k_bytes;
      const auto v_size = kv_cache_info.v_bytes;
      size_t off_sl = 0;
      for (int gi = 0; gi < infer_groups.size(); ++gi) {
        const int update_bs = infer_groups[gi].size();
        const int update_sl = n_tokens[infer_groups[gi].front()];
        const int update_block_id = block_ids[infer_groups[gi].front()];
        const int update_n_past = n_pasts[infer_groups[gi].front()];
        struct ne_tensor* k_cache_g = ne_view_4d(ctx0, kv_self.k,                      // tensor
                                                 head_size, n_ctx, n_head, update_bs,  // ne
                                                 0, 0, k_size,                         // nb (bestla managed)
                                                 il * kv_n_ctx_block * k_size + update_block_id * k_size);  // offset);
        struct ne_tensor* k_cur_g =
            ne_view_4d(ctx0, Kcur, head_size, n_head, update_sl, update_bs, ne_element_size(Kcur) * head_size,
                       ne_element_size(Kcur) * head_size * n_head,
                       ne_element_size(Kcur) * head_size * n_head * update_sl, ne_element_size(Kcur) * off_sl);
        ne_build_forward_expand(&gf, ne_flash_attn_update_k(ctx0, k_cache_g, k_cur_g, update_n_past, is_ring_full));
        struct ne_tensor* v_cache_g = ne_view_4d(ctx0, kv_self.v,                      // tensor
                                                 head_size, n_ctx, n_head, update_bs,  // ne
                                                 0, 0, v_size,                         // nb (bestla managed)
                                                 il * kv_n_ctx_block * v_size + update_block_id * v_size);  // offset);
        // bestla always view V as (D, n_head, seq, bs)
        struct ne_tensor* v_cur_g =
            ne_view_4d(ctx0, Vcur, head_size, n_head, update_sl, update_bs, ne_element_size(Vcur) * head_size,
                       ne_element_size(Vcur) * head_size * n_head,
                       ne_element_size(Vcur) * head_size * n_head * update_sl, ne_element_size(Vcur) * off_sl);
        ne_build_forward_expand(&gf, ne_flash_attn_update_v(ctx0, v_cache_g, v_cur_g, update_n_past, is_ring_full));
        off_sl += head_size * n_head * update_sl * update_bs;
      }
    }

    // for-loop self-attention
    struct ne_tensor* KQV_merged_contiguous =
        ne_new_tensor_2d(ctx0, NE_TYPE_F32, head_size * n_head, seq_len_sum, NE_SIZE_CALC);
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
      struct ne_tensor *K, *V;
      const int n_cached_gi = shift_roped_k ? n_cached : attn_n_past + attn_sl;
      if (run_mha_reordered) {
        const auto k_size = kv_cache_info.k_bytes;
        K = ne_view_4d(ctx0, kv_self.k,                                                     // tensor
                       head_size, n_cached_gi, n_head, attn_bs,                             // ne
                       kv_cache_info.stride_k_sl, kv_cache_info.stride_k_head_num, k_size,  // nb (bestla managed)
                       il * kv_n_ctx_block * k_size + attn_block_id * k_size);              // offset
        *reinterpret_cast<ATTN_FWD_LAYOUT*>(&K->nb[0]) = kv_cache_info.k_layout;
        if (is_ring_full) {
          struct ne_tensor* cossin_cache = nullptr;
          // Currently we only cache cossin for N == 1 in model-wide; It may be worthwhile to cache cossin for other N
          // in a single eval execution
          if (N == 1) cossin_cache = kv_self.cossin;
          K = ne_rope_shift_inplace(ctx0, K, -N, n_rot, 0, 0, n_keep, cossin_cache, hparams.freq_base,
                                    hparams.freq_scale);
        }
        const auto v_size = kv_cache_info.v_bytes;
        V = ne_view_4d(ctx0, kv_self.v,                          // tensor
                       n_cached_gi, head_size, n_head, attn_bs,  // ne
                       kv_cache_info.stride_v_head_size, kv_cache_info.stride_v_head_num,
                       v_size,                                                  // nb (bestla managed)
                       il * kv_n_ctx_block * v_size + attn_block_id * v_size);  // offset
        *reinterpret_cast<ATTN_FWD_LAYOUT*>(&V->nb[0]) = kv_cache_info.v_layout;
      } else if (run_mha_fp16) {
        size_t off_kv_layer =
            il * n_ctx * head_size * n_head * kv_n_ctx_block + attn_block_id * n_ctx * head_size * n_head;
        V = ne_permute(
            ctx0,
            ne_view_4d(ctx0, kv_self.v, head_size, n_head, n_cached_gi, attn_bs, ne_element_size(kv_self.v) * head_size,
                       ne_element_size(kv_self.v) * head_size * n_head,
                       ne_element_size(kv_self.v) * head_size * n_head * n_ctx,
                       ne_element_size(kv_self.v) * off_kv_layer),
            1, 2, 0, 3);

        // split cached V into n_head heads
        K = ne_view_4d(ctx0, kv_self.k, n_cached_gi, head_size, n_head, attn_bs, n_ctx * ne_element_size(kv_self.k),
                       n_ctx * ne_element_size(kv_self.k) * head_size,
                       n_ctx * ne_element_size(kv_self.k) * head_size * n_head,
                       ne_element_size(kv_self.k) * off_kv_layer);
        K = ne_permute(ctx0, K, 1, 0, 2, 3);  // head_size n_cached_gi n_head attn_bs
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
      } else {
        std::vector<int> attn_block_ids(infer_groups[gi].size());
        for (int j = 0; j < infer_groups[gi].size(); ++j) {
          attn_block_ids[j] = block_ids[infer_groups[gi][j]];
        }
        K = model_kv_cache_seq_concat(&gf, &lctx, ctx0, head_size, n_cached_gi, n_head, attn_bs, attn_block_ids, il);
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
        V = model_kv_cache_seq_concat(&gf, &lctx, ctx0, n_cached_gi, head_size, n_head, attn_bs, attn_block_ids, il,
                                      false);
      }
      ne_set_name(K, std::string("K_" + suffix).c_str());
      ne_set_name(V, std::string("V_" + suffix).c_str());

      struct ne_tensor* KQV_merged_gi;
      const float attn_scale = 1.0f / sqrtf(static_cast<float>(head_size));
      ne_attn_flags_t attn_flags = NE_ATTN_FLAG_NONE;
      if (NE_ATTN_PREFER_FP32) attn_flags |= NE_ATTN_FLAG_PREFER_FP32;
      if (attn_n_total == 0 || !shift_roped_k)
        attn_flags |= NE_ATTN_FLAG_IS_CAUSAL;  // no causal mask on next-token cases
      if (run_mha_reordered) {                 // reordered kv-cache bf16 mha must be used if run_mha_reordered
        struct ne_tensor* KQV_Out = ne_flash_attn(ctx0, Q, K, V, attn_scale, attn_flags);
        KQV_merged_gi = ne_view_2d(ctx0, KQV_Out, head_size * n_head, attn_sl * attn_bs,
                                   head_size * n_head * ne_element_size(KQV_Out), 0);
      } else if (run_mha_fp16) {  // non-reordered kv-cache fp16 mha
        struct ne_tensor* KQV_Out = ne_flash_attn(ctx0, Q, K, V, attn_scale, attn_flags);
        KQV_merged_gi = ne_view_2d(ctx0, KQV_Out, head_size * n_head, attn_sl * attn_bs,
                                   head_size * n_head * ne_element_size(KQV_Out), 0);
      } else if (attn_n_total == 0 && run_mha_bf16_first) {
        // non-reordered kv-cache bf16 mha (first token only)
        auto vnele = ne_nelements(Vcur);
        struct ne_tensor* Vtmp = ne_new_tensor_1d(ctx0, NE_TYPE_F16, vnele, NE_SIZE_CALC);
        Vtmp = ne_cpy(ctx0, ne_view_1d(ctx0, Vcur, vnele, 0), Vtmp);
        Vtmp = ne_view_4d(ctx0, Vtmp, head_size, n_head, attn_sl, attn_bs, ne_element_size(Vtmp) * head_size,
                          ne_element_size(Vtmp) * head_size * n_head,
                          attn_sl * ne_element_size(Vtmp) * head_size * n_head, 0);
        Vtmp = ne_permute(ctx0, Vtmp, 1, 2, 0, 3);
        struct ne_tensor* KQV_Out = ne_flash_attn(ctx0, Q, K, Vtmp, attn_scale, attn_flags);
        KQV_merged_gi = ne_view_2d(ctx0, KQV_Out, head_size * n_head, attn_sl * attn_bs,
                                   head_size * n_head * ne_element_size(KQV_Out), 0);
      } else {
        // K * Q
        struct ne_tensor* KQ = ne_mul_mat(ctx0, K, Q);
        ne_set_name(KQ, std::string("KQ_" + suffix).c_str());

        // KQ_scaled = KQ / sqrt(n_embd/n_head)
        struct ne_tensor* KQ_scale = ne_new_f32(ctx0, attn_scale);
        ne_set_name(KQ_scale, std::string("1/sqrt(n_embd/n_head)_" + suffix).c_str());

        // KQ_scaled shape [n_cached, N, n_head, 1]
        struct ne_tensor* KQ_scaled = ne_scale_inplace(ctx0, KQ, KQ_scale);
        ne_set_name(KQ_scaled, std::string("KQ_scaled_" + suffix).c_str());

        // KQ_scaled = mask_past(KQ_scaled)
        if (attn_n_total == 0 || !shift_roped_k || !no_padding) {
          std::vector<int> attn_n_padding(infer_groups[gi].size(), 0);
          for (int npa = 0; !n_padding.empty() && npa < infer_groups[gi].size(); ++npa) {
            attn_n_padding[npa] = n_padding[infer_groups[gi][npa]];
          }
          KQ_scaled = ne_diag_mask_inf_with_padding_inplace(ctx0, KQ_scaled, attn_n_past, attn_n_padding.data());
          ne_set_name(KQ_scaled, std::string("KQ_masked_" + suffix).c_str());
        }

        // KQ = soft_max(KQ_masked)
        struct ne_tensor* KQ_soft_max = ne_soft_max_inplace(ctx0, KQ_scaled);
        ne_set_name(KQ_soft_max, std::string("KQ_soft_max_" + suffix).c_str());

        struct ne_tensor* KQV = ne_mul_mat(ctx0, V, KQ_soft_max);
        ne_set_name(KQV, std::string("KQV_" + suffix).c_str());

        // KQV_merged = KQV.permute(0, 2, 1, 3)
        KQV_merged_gi = ne_permute(ctx0, KQV, 0, 2, 1, 3);
      }
      ne_set_name(KQV_merged_gi, std::string("KQV_merged_" + suffix).c_str());
      ne_build_forward_expand(&gf, ne_cpy(ctx0, KQV_merged_gi,
                                          ne_view_2d(ctx0, KQV_merged_contiguous, head_size * n_head, attn_sl * attn_bs,
                                                     head_size * n_head * ne_element_size(KQV_merged_contiguous),
                                                     ne_element_size(KQV_merged_contiguous) * off_sl)));
      off_sl += head_size * n_head * attn_sl * attn_bs;
    }
    ne_set_name(KQV_merged_contiguous, "KQV_merged_contiguous");

    // projection (no bias)
    struct ne_tensor* KQV_out = ne_mul_mat(ctx0, model.layers[il].attn[3], KQV_merged_contiguous);
    ne_set_name(KQV_out, "KQV_out");

#ifdef NS_TP_MODEL
    if (enable_tp) {
      KQV_out = ne_all_reduce(ctx0, KQV_out);
    }
#endif

    lctx.use_buf(ctx0, 1);
    struct ne_tensor* inpFF = KQV_out;

    // feed-forward network
    // disable ffn fusion because fp32 support not ready
    if (bestla_fusion_FFN_Add_GeLu_f32f32_support(model.layers[il].ffn[0]->data, model.layers[il].ffn[2]->data,
                                                  seq_len_sum, inpSA->ne[0], model.layers[il].ffn[0]->ne[1],
                                                  model.layers[il].ffn[2]->ne[1])) {
      cur = ne_ffn_add_gelu(ctx0, model.layers[il].ffn[0], model.layers[il].ffn[2], model.layers[il].ffn[1],
                            model.layers[il].ffn[3], inpSA);
    } else {
      struct ne_tensor* FFN_in = ne_mul_mat(ctx0, model.layers[il].ffn[0], inpSA);
      ne_set_name(FFN_in, "FFN_in");

      cur = ne_add(ctx0, ne_repeat(ctx0, model.layers[il].ffn[1], FFN_in), FFN_in);

      // GELU activation
      cur = ne_gelu(ctx0, cur);

      struct ne_tensor* FFN_out = ne_mul_mat(ctx0, model.layers[il].ffn[2], cur);
      ne_set_name(FFN_out, "FFN_out");
      // NOTICE: when TP, only master node add this bias
      cur = ne_add(ctx0, ne_repeat(ctx0, model.layers[il].ffn[3], FFN_out), FFN_out);
    }
#ifdef NS_TP_MODEL
    // if tp model then all reduce as the weight has been split
    if (enable_tp) {
      cur = ne_all_reduce(ctx0, cur);
    }
#endif
    cur = ne_add(ctx0, cur, inpFF);
    // if (il == 20) {
    //   cur = ne_dump_tensor(ctx0, cur);
    // }

    // input for next layer
    inpL = ne_add(ctx0, cur, inpL);
  }
  lctx.use_buf(ctx0, 0);

  // used at the end to optionally extract the embeddings
  struct ne_tensor* embeddings = nullptr;

  // norm
  {
    inpL = ne_norm(ctx0, inpL, hparams.norm_eps);

    // inpL = inpL*norm(broadcasted)
    inpL = ne_add(ctx0, ne_mul(ctx0, inpL, model.others[1]), model.others[2]);
  }

  // lm_head
  if (bestla_fusion_add_f32f32_support(model.others[3]->data, seq_len_sum, model.others[3]->ne[1],
                                       model.others[3]->ne[0])) {
    inpL = ne_mul_mat_with_bias(ctx0, model.others[3], model.others[4], inpL);
  } else {
    inpL = ne_mul_mat(ctx0, model.others[3], inpL);
    inpL = ne_add(ctx0, ne_repeat(ctx0, model.others[4], inpL), inpL);
  }

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
  if (!gptj_model_eval_internal(ctx, inputs, n_input, n_threads)) {
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
