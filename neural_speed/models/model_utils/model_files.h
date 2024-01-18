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
// Defines fileno on msys:

#ifndef MODEL_FILES_H
#define MODEL_FILES_H

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#include <cstddef>
#include <cstdint>
#include <cstdio>
#endif

#if UINTPTR_MAX == 0xFFFFFFFF
#define NE_MEM_ALIGN 4
#else
#define NE_MEM_ALIGN 16
#endif

#include "core/layers/bestla_common.hpp"
#include "core/ne_layers.h"
#include "models/model_utils/util.h"
#include "models/models.h"
#include "models/model_utils/gguf.h"
#include <iostream>

template <typename T>
static T checked_mul(T a, T b) {
  T ret = a * b;
  if (a != 0 && ret / a != b) {
    throw format("overflow multiplying %llu * %llu", (unsigned long long)a, (unsigned long long)b);
  }
  return ret;
}

static size_t checked_div(size_t a, size_t b) {
  if (b == 0 || a % b != 0) {
    throw format("error dividing %zu / %zu", a, b);
  }
  return a / b;
}

static std::string model_format_tensor_shape(const std::vector<uint32_t>& ne) {
  char buf[256];
  snprintf(buf, sizeof(buf), "%5u", ne.at(0));
  for (size_t i = 1; i < ne.size(); i++) {
    snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), " x %5u", ne.at(i));
  }
  return buf;
}

static size_t model_calc_tensor_size(const std::vector<uint32_t>& ne, enum ne_type type) {
  size_t size = ne_type_size(type);
  for (uint32_t dim : ne) {
    size = checked_mul<size_t>(size, dim);
  }
  return size / ne_blck_size(type);
}

struct model_load_tensor_shard {
  std::vector<uint32_t> ne;
  size_t size;
  enum ne_type type;
  size_t file_idx;
  size_t file_off;

  void calc_size() { size = model_calc_tensor_size(ne, type); }
};

enum model_split_type {
  SPLIT_NONE,
  SPLIT_BY_COLUMNS,
  SPLIT_BY_ROWS,
  TP_1D_ROW,
  TP_1D_COLUMN,
  TP_1D_ONLY_MASTER,
  TP_1D_QKV_ROW,
  TP_1D_QKV_COLUMN
};

struct model_load_tensor {
  std::vector<model_load_tensor_shard> shards;

#ifdef NS_TP_MODEL
  parallel_context* p_ctx = init_parallel_context();
  int32_t world_size = get_tp_size(p_ctx);
  int32_t rank = get_tp_rank(p_ctx);
  bool enable_tp = world_size > 1 ? true : false;

#endif
  std::string name;
  enum ne_type type = NE_TYPE_F32;
  model_split_type split_type = SPLIT_NONE;
  std::vector<uint32_t> ne;
  size_t size;
  struct ne_tensor* ne_tensor = nullptr;
  uint8_t* data;

  model_load_tensor(const std::string& name) : name(name) {}

  void calc_all() {
    calc_type();
    calc_split_type();
    calc_ne();
    if (type == NE_TYPE_BTLA) {
      size = shards[0].size;
    } else {
      calc_size();
    }
  }

  void calc_type() {
    const auto& first_shard = shards.at(0);
    for (const auto& shard : shards) {
      if (shard.type != first_shard.type) {
        throw format("inconsistent tensor shard type in '%s'", name.c_str());
      }
    }
    type = first_shard.type;
  }

  void calc_split_type() {
    if (shards.at(0).ne.size() == 1 ||  // 1D tensors are just duplicated in every file
        shards.size() == 1) {           // only one file?
      split_type = SPLIT_NONE;
    } else if (name.find("tok_embeddings.") == 0 || name.find(".attention.wo.weight") != std::string::npos ||
               name.find(".feed_forward.w2.weight") != std::string::npos) {
      split_type = SPLIT_BY_COLUMNS;
    } else {
      split_type = SPLIT_BY_ROWS;
    }

#ifdef NS_TP_MODEL
    if (enable_tp) {
      // TODO it's not good to check type here, mmaybe move to specific model files
      if (name.find(".attn.q_proj.weight") != std::string::npos ||
          name.find(".attn.k_proj.weight") != std::string::npos ||
          name.find(".attn.v_proj.weight") != std::string::npos ||
          name.find(".mlp.fc_in.weight") != std::string::npos ||
          // for baichuan
          name.find(".mlp.gate_proj.weight") != std::string::npos ||
          name.find(".mlp.up_proj.weight") != std::string::npos ||
          // for chatglm2
          name.find(".mlp.dense_h_to_4h.weight") != std::string::npos ||
          // for llama model
          name.find(".attention.wq.weight") != std::string::npos ||
          name.find(".attention.wk.weight") != std::string::npos ||
          name.find(".attention.wv.weight") != std::string::npos ||
          name.find(".feed_forward.w1.weight") != std::string::npos ||
          name.find(".feed_forward.w3.weight") != std::string::npos) {
        split_type = TP_1D_ROW;
      }
      if (name.find(".self_attn.W_pack.weight") != std::string::npos ||
          // for chatglm2
          name.find(".self_attention.query_key_value.weight") != std::string::npos) {
        split_type = TP_1D_QKV_ROW;
      }
      if (name.find(".self_attention.query_key_value.bias") != std::string::npos) {
        split_type = TP_1D_QKV_COLUMN;
      }
      if (name.find(".mlp.fc_in.bias") != std::string::npos || name.find(".mlp.fc_out.weight") != std::string::npos ||
          name.find(".attn.out_proj.weight") != std::string::npos ||
          name.find(".self_attention.dense.weight") != std::string::npos ||
          // for baichuan
          name.find(".self_attn.o_proj.weight") != std::string::npos ||
          name.find(".mlp.down_proj.weight") != std::string::npos ||
          // for chatglm2
          name.find(".mlp.dense_4h_to_h.weight") != std::string::npos ||
          // TODO check if this part should be column
          name.find(".attention.wo.weight") != std::string::npos ||
          name.find(".feed_forward.w2.weight") != std::string::npos) {
        split_type = TP_1D_COLUMN;
      }
      if (name.find(".mlp.fc_out.bias") != std::string::npos) {
        split_type = TP_1D_ONLY_MASTER;
      }
    }
#endif
  }

  void calc_ne() {
    const auto& first_shard = shards.at(0);
    for (const auto& shard : shards) {
      if (shard.ne != first_shard.ne) {
        throw format("inconsistent tensor shard shape in '%s': first was %s, other was %s", name.c_str(),
                     model_format_tensor_shape(first_shard.ne).c_str(), model_format_tensor_shape(shard.ne).c_str());
      }
    }
    ne = first_shard.ne;
    MODEL_ASSERT(shards.size() <= UINT32_MAX);
    uint32_t n_shards = (uint32_t)shards.size();
    switch (split_type) {
      case SPLIT_NONE:
        ne = first_shard.ne;
        break;
      case SPLIT_BY_COLUMNS:
        ne = {checked_mul<uint32_t>(first_shard.ne[0], n_shards), first_shard.ne[1]};
        break;
      case SPLIT_BY_ROWS:
        ne = {first_shard.ne[0], checked_mul<uint32_t>(first_shard.ne[1], n_shards)};
        break;
#ifdef NS_TP_MODEL
      case TP_1D_ROW:
      case TP_1D_QKV_ROW:
        MODEL_ASSERT(first_shard.ne.size() > 1);
        MODEL_ASSERT(first_shard.ne[1] % world_size == 0);
        ne = {first_shard.ne[0], first_shard.ne[1] / world_size};
        break;
      case TP_1D_COLUMN:
      case TP_1D_QKV_COLUMN:
        MODEL_ASSERT(first_shard.ne[0] % world_size == 0);
        if (first_shard.ne.size() == 1) {
          ne = {first_shard.ne[0] / world_size};
        } else {
          ne = {first_shard.ne[0] / world_size, first_shard.ne[1]};
        }
        break;
      case TP_1D_ONLY_MASTER:
        ne = first_shard.ne;
        break;
#endif
    }
  }

  void calc_size() { size = model_calc_tensor_size(ne, type); }
};

struct model_load_tensors_map {
  // tensors is kept in a separate vector to preserve file order
  std::vector<model_load_tensor> tensors;
  std::unordered_map<std::string, size_t> name_to_idx;
};

struct gguf_loader {
  FILE* gguf_file;

  gguf_loader(FILE* ne_file) : gguf_file(ne_file) {}

  const char* gguf_type_name(enum gguf_type type) { return GGUF_TYPE_NAME[type]; }

  int gguf_get_version(const struct gguf_context* ctx) { return ctx->header.version; }

  size_t gguf_get_alignment(const struct gguf_context* ctx) { return ctx->alignment; }

  size_t gguf_get_data_offset(const struct gguf_context* ctx) { return ctx->offset; }

  void* gguf_get_data(const struct gguf_context* ctx) { return ctx->data; }

  int gguf_get_n_kv(const struct gguf_context* ctx) { return ctx->header.n_kv; }

  int gguf_find_key(const struct gguf_context* ctx, const char* key) {
    // return -1 if key not found
    int keyfound = -1;

    const int n_kv = gguf_get_n_kv(ctx);

    for (int i = 0; i < n_kv; ++i) {
      if (strcmp(key, gguf_get_key(ctx, i)) == 0) {
        keyfound = i;
        break;
      }
    }

    return keyfound;
  }

  const char* gguf_get_key(const struct gguf_context* ctx, int key_id) {
    NE_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    return ctx->kv[key_id].key.data;
  }

  enum gguf_type gguf_get_kv_type(const struct gguf_context* ctx, int key_id) {
    NE_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    return ctx->kv[key_id].type;
  }

  enum gguf_type gguf_get_arr_type(const struct gguf_context* ctx, int key_id) {
    NE_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    NE_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_ARRAY);
    return ctx->kv[key_id].value.arr.type;
  }

  const void* gguf_get_arr_data(const struct gguf_context* ctx, int key_id) {
    NE_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    NE_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_ARRAY);
    return ctx->kv[key_id].value.arr.data;
  }

  const char* gguf_get_arr_str(const struct gguf_context* ctx, int key_id, int i) {
    NE_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    NE_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_ARRAY);
    struct gguf_kv* kv = &ctx->kv[key_id];
    struct gguf_str* str = &((struct gguf_str*)kv->value.arr.data)[i];
    return str->data;
  }

  int gguf_get_arr_n(const struct gguf_context* ctx, int key_id) {
    NE_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    NE_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_ARRAY);
    return ctx->kv[key_id].value.arr.n;
  }

  uint8_t gguf_get_val_u8(const struct gguf_context* ctx, int key_id) {
    NE_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    NE_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_UINT8);
    return ctx->kv[key_id].value.uint8;
  }

  int8_t gguf_get_val_i8(const struct gguf_context* ctx, int key_id) {
    NE_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    NE_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_INT8);
    return ctx->kv[key_id].value.int8;
  }

  uint16_t gguf_get_val_u16(const struct gguf_context* ctx, int key_id) {
    NE_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    NE_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_UINT16);
    return ctx->kv[key_id].value.uint16;
  }

  int16_t gguf_get_val_i16(const struct gguf_context* ctx, int key_id) {
    NE_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    NE_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_INT16);
    return ctx->kv[key_id].value.int16;
  }

  uint32_t gguf_get_val_u32(const struct gguf_context* ctx, int key_id) {
    NE_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    NE_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_UINT32);
    return ctx->kv[key_id].value.uint32;
  }

  int32_t gguf_get_val_i32(const struct gguf_context* ctx, int key_id) {
    NE_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    NE_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_INT32);
    return ctx->kv[key_id].value.int32;
  }

  float gguf_get_val_f32(const struct gguf_context* ctx, int key_id) {
    NE_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    NE_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_FLOAT32);
    return ctx->kv[key_id].value.float32;
  }

  uint64_t gguf_get_val_u64(const struct gguf_context* ctx, int key_id) {
    NE_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    NE_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_UINT64);
    return ctx->kv[key_id].value.uint64;
  }

  int64_t gguf_get_val_i64(const struct gguf_context* ctx, int key_id) {
    NE_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    NE_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_INT64);
    return ctx->kv[key_id].value.int64;
  }

  double gguf_get_val_f64(const struct gguf_context* ctx, int key_id) {
    NE_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    NE_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_FLOAT64);
    return ctx->kv[key_id].value.float64;
  }

  bool gguf_get_val_bool(const struct gguf_context* ctx, int key_id) {
    NE_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    NE_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_BOOL);
    return ctx->kv[key_id].value.bool_;
  }

  const char* gguf_get_val_str(const struct gguf_context* ctx, int key_id) {
    NE_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    NE_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_STRING);
    return ctx->kv[key_id].value.str.data;
  }

  const void* gguf_get_val_data(const struct gguf_context* ctx, int key_id) {
    NE_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    NE_ASSERT(ctx->kv[key_id].type != GGUF_TYPE_ARRAY);
    NE_ASSERT(ctx->kv[key_id].type != GGUF_TYPE_STRING);
    return &ctx->kv[key_id].value;
  }

  int gguf_get_n_tensors(const struct gguf_context* ctx) { return ctx->header.n_tensors; }

  int gguf_find_tensor(const struct gguf_context* ctx, const char* name) {
    // return -1 if tensor not found
    int tensorfound = -1;

    const int n_tensors = gguf_get_n_tensors(ctx);

    for (int i = 0; i < n_tensors; ++i) {
      if (strcmp(name, gguf_get_tensor_name(ctx, i)) == 0) {
        tensorfound = i;
        break;
      }
    }

    return tensorfound;
  }

  size_t gguf_get_tensor_offset(const struct gguf_context* ctx, int i) { return ctx->infos[i].offset; }

  char* gguf_get_tensor_name(const struct gguf_context* ctx, int i) { return ctx->infos[i].name.data; }

  // returns the index
  // remove static
  int gguf_get_or_add_key(struct gguf_context* ctx, const char* key) {
    const int idx = gguf_find_key(ctx, key);
    if (idx >= 0) {
      return idx;
    }

    const int n_kv = gguf_get_n_kv(ctx);

    ctx->kv = reinterpret_cast<struct gguf_kv*>(realloc(ctx->kv, (n_kv + 1) * sizeof(struct gguf_kv)));
    ctx->kv[n_kv].key.n = strlen(key);
    ctx->kv[n_kv].key.data = strdup(key);
    ctx->header.n_kv++;

    return n_kv;
  }

  // remove static
  std::string gguf_kv_to_str(struct gguf_context* ctx_gguf, int i) {
    const enum gguf_type type = gguf_get_kv_type(ctx_gguf, i);

    switch (type) {
      case GGUF_TYPE_STRING:
        return gguf_get_val_str(ctx_gguf, i);
      case GGUF_TYPE_ARRAY: {
        const enum gguf_type arr_type = gguf_get_arr_type(ctx_gguf, i);
        int arr_n = gguf_get_arr_n(ctx_gguf, i);
        const void* data = gguf_get_arr_data(ctx_gguf, i);
        std::stringstream ss;
        ss << "[";
        for (int j = 0; j < arr_n; j++) {
          if (arr_type == GGUF_TYPE_STRING) {
            std::string val = gguf_get_arr_str(ctx_gguf, i, j);
            // escape quotes
            replace_all(val, "\\", "\\\\");
            replace_all(val, "\"", "\\\"");
            ss << '"' << val << '"';
          } else if (arr_type == GGUF_TYPE_ARRAY) {
            ss << "???";
          } else {
            ss << gguf_data_to_str(arr_type, data, j);
          }
          if (j < arr_n - 1) {
            ss << ", ";
          }
        }
        ss << "]";
        return ss.str();
      }
      default:
        return gguf_data_to_str(type, gguf_get_val_data(ctx_gguf, i), 0);
    }
  }

  size_t file_offset(const struct gguf_context* ctx_gguf, const char* name) {
    const int idx = gguf_find_tensor(ctx_gguf, name);

    if (idx < 0) {
      throw std::runtime_error(format("%s: tensor '%s' not found in the file", __func__, name));
    }

    size_t data_offset = gguf_get_data_offset(ctx_gguf);
    size_t tensor_offset = gguf_get_tensor_offset(ctx_gguf, idx);
    return data_offset + tensor_offset;
  }

  void gguf_set_val_u8(struct gguf_context* ctx, const char* key, uint8_t val) {
    const int idx = gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type = GGUF_TYPE_UINT8;
    ctx->kv[idx].value.uint8 = val;
  }

  void gguf_set_val_i8(struct gguf_context* ctx, const char* key, int8_t val) {
    const int idx = gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type = GGUF_TYPE_INT8;
    ctx->kv[idx].value.int8 = val;
  }

  void gguf_set_val_u16(struct gguf_context* ctx, const char* key, uint16_t val) {
    const int idx = gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type = GGUF_TYPE_UINT16;
    ctx->kv[idx].value.uint16 = val;
  }

  void gguf_set_val_i16(struct gguf_context* ctx, const char* key, int16_t val) {
    const int idx = gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type = GGUF_TYPE_INT16;
    ctx->kv[idx].value.int16 = val;
  }

  void gguf_set_val_u32(struct gguf_context* ctx, const char* key, uint32_t val) {
    const int idx = gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type = GGUF_TYPE_UINT32;
    ctx->kv[idx].value.uint32 = val;
  }

  void gguf_set_val_i32(struct gguf_context* ctx, const char* key, int32_t val) {
    const int idx = gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type = GGUF_TYPE_INT32;
    ctx->kv[idx].value.int32 = val;
  }

  void gguf_set_val_f32(struct gguf_context* ctx, const char* key, float val) {
    const int idx = gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type = GGUF_TYPE_FLOAT32;
    ctx->kv[idx].value.float32 = val;
  }

  void gguf_set_val_u64(struct gguf_context* ctx, const char* key, uint64_t val) {
    const int idx = gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type = GGUF_TYPE_UINT64;
    ctx->kv[idx].value.uint64 = val;
  }

  void gguf_set_val_i64(struct gguf_context* ctx, const char* key, int64_t val) {
    const int idx = gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type = GGUF_TYPE_INT64;
    ctx->kv[idx].value.int64 = val;
  }

  void gguf_set_val_f64(struct gguf_context* ctx, const char* key, double val) {
    const int idx = gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type = GGUF_TYPE_FLOAT64;
    ctx->kv[idx].value.float64 = val;
  }

  void gguf_set_val_bool(struct gguf_context* ctx, const char* key, bool val) {
    const int idx = gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type = GGUF_TYPE_BOOL;
    ctx->kv[idx].value.bool_ = val;
  }

  void gguf_set_val_str(struct gguf_context* ctx, const char* key, const char* val) {
    const int idx = gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type = GGUF_TYPE_STRING;
    ctx->kv[idx].value.str.n = strlen(val);
    ctx->kv[idx].value.str.data = strdup(val);
  }

  void gguf_set_arr_data(struct gguf_context* ctx, const char* key, enum gguf_type type, const void* data, int n) {
    const int idx = gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type = GGUF_TYPE_ARRAY;
    ctx->kv[idx].value.arr.type = type;
    ctx->kv[idx].value.arr.n = n;
    ctx->kv[idx].value.arr.data = malloc(n * GGUF_TYPE_SIZE[type]);
    memcpy(ctx->kv[idx].value.arr.data, data, n * GGUF_TYPE_SIZE[type]);
  }

  void gguf_set_arr_str(struct gguf_context* ctx, const char* key, const char** data, int n) {
    const int idx = gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type = GGUF_TYPE_ARRAY;
    ctx->kv[idx].value.arr.type = GGUF_TYPE_STRING;
    ctx->kv[idx].value.arr.n = n;
    ctx->kv[idx].value.arr.data = malloc(n * sizeof(struct gguf_str));
    for (int i = 0; i < n; i++) {
      struct gguf_str* str = &((struct gguf_str*)ctx->kv[idx].value.arr.data)[i];
      str->n = strlen(data[i]);
      str->data = strdup(data[i]);
    }
  }

  void gguf_free(struct gguf_context* ctx) {
    if (ctx == NULL) {
      return;
    }

    if (ctx->kv) {
      // free string memory - not great..
      for (uint32_t i = 0; i < ctx->header.n_kv; ++i) {
        struct gguf_kv* kv = &ctx->kv[i];

        if (kv->key.data) {
          free(kv->key.data);
        }

        if (kv->type == GGUF_TYPE_STRING) {
          if (kv->value.str.data) {
            free(kv->value.str.data);
          }
        }

        if (kv->type == GGUF_TYPE_ARRAY) {
          if (kv->value.arr.data) {
            if (kv->value.arr.type == GGUF_TYPE_STRING) {
              for (uint32_t j = 0; j < kv->value.arr.n; ++j) {
                struct gguf_str* str = &((struct gguf_str*)kv->value.arr.data)[j];
                if (str->data) {
                  free(str->data);
                }
              }
            }
            free(kv->value.arr.data);
          }
        }
      }

      free(ctx->kv);
    }

    if (ctx->infos) {
      for (uint32_t i = 0; i < ctx->header.n_tensors; ++i) {
        struct gguf_tensor_info* info = &ctx->infos[i];

        if (info->name.data) {
          free(info->name.data);
        }
      }

      free(ctx->infos);
    }
  }

  struct gguf_context* gguf_init_from_file(model_load_tensors_map& tensors_map, size_t& gguf_data_offset) {
    if (!gguf_file) {
      return nullptr;
    }

    size_t offset = 0;
    char magic[4];

    gguf_fread_el(gguf_file, &magic, sizeof(magic), &offset);

    struct gguf_context* ctx = reinterpret_cast<struct gguf_context*>(GGML_ALIGNED_MALLOC(sizeof(struct gguf_context)));
    ctx->offset = 0;
    // read the header
    strncpy(ctx->header.magic, magic, 4);

    bool ok = true;
    ctx->kv = NULL;
    ctx->infos = NULL;
    ctx->data = NULL;

    ok = ok && gguf_fread_el(gguf_file, &ctx->header.version, sizeof(ctx->header.version), &offset);
    ok = ok && gguf_fread_el(gguf_file, &ctx->header.n_tensors, sizeof(ctx->header.n_tensors), &offset);
    ok = ok && gguf_fread_el(gguf_file, &ctx->header.n_kv, sizeof(ctx->header.n_kv), &offset);

    if (ctx->header.version == 1) {
      fprintf(stderr, "%s: GGUFv1 is no longer supported. please use a more up-to-date version\n", __func__);
      fclose(gguf_file);
      gguf_free(ctx);
      return nullptr;
    }

    if (!ok) {
      fprintf(stderr, "%s: failed to read header\n", __func__);
      fclose(gguf_file);
      gguf_free(ctx);
      return nullptr;
    }

    // read the kv pairs
    ctx->kv = reinterpret_cast<struct gguf_kv*>(malloc(ctx->header.n_kv * sizeof(struct gguf_kv)));

    for (uint64_t i = 0; i < ctx->header.n_kv; ++i) {
      struct gguf_kv* kv = &ctx->kv[i];

      ok = ok && gguf_fread_str(gguf_file, &kv->key, &offset);
      ok = ok && gguf_fread_el(gguf_file, &kv->type, sizeof(kv->type), &offset);

      switch (kv->type) {
        case GGUF_TYPE_UINT8:
          ok = ok && gguf_fread_el(gguf_file, &kv->value.uint8, sizeof(kv->value.uint8), &offset);
          break;
        case GGUF_TYPE_INT8:
          ok = ok && gguf_fread_el(gguf_file, &kv->value.int8, sizeof(kv->value.int8), &offset);
          break;
        case GGUF_TYPE_UINT16:
          ok = ok && gguf_fread_el(gguf_file, &kv->value.uint16, sizeof(kv->value.uint16), &offset);
          break;
        case GGUF_TYPE_INT16:
          ok = ok && gguf_fread_el(gguf_file, &kv->value.int16, sizeof(kv->value.int16), &offset);
          break;
        case GGUF_TYPE_UINT32:
          ok = ok && gguf_fread_el(gguf_file, &kv->value.uint32, sizeof(kv->value.uint32), &offset);
          break;
        case GGUF_TYPE_INT32:
          ok = ok && gguf_fread_el(gguf_file, &kv->value.int32, sizeof(kv->value.int32), &offset);
          break;
        case GGUF_TYPE_FLOAT32:
          ok = ok && gguf_fread_el(gguf_file, &kv->value.float32, sizeof(kv->value.float32), &offset);
          break;
        case GGUF_TYPE_UINT64:
          ok = ok && gguf_fread_el(gguf_file, &kv->value.uint64, sizeof(kv->value.uint64), &offset);
          break;
        case GGUF_TYPE_INT64:
          ok = ok && gguf_fread_el(gguf_file, &kv->value.int64, sizeof(kv->value.int64), &offset);
          break;
        case GGUF_TYPE_FLOAT64:
          ok = ok && gguf_fread_el(gguf_file, &kv->value.float64, sizeof(kv->value.float64), &offset);
          break;
        case GGUF_TYPE_BOOL:
          ok = ok && gguf_fread_el(gguf_file, &kv->value.bool_, sizeof(kv->value.bool_), &offset);
          break;
        case GGUF_TYPE_STRING:
          ok = ok && gguf_fread_str(gguf_file, &kv->value.str, &offset);
          break;
        case GGUF_TYPE_ARRAY: {
          ok = ok && gguf_fread_el(gguf_file, &kv->value.arr.type, sizeof(kv->value.arr.type), &offset);
          ok = ok && gguf_fread_el(gguf_file, &kv->value.arr.n, sizeof(kv->value.arr.n), &offset);

          switch (kv->value.arr.type) {
            case GGUF_TYPE_UINT8:
            case GGUF_TYPE_INT8:
            case GGUF_TYPE_UINT16:
            case GGUF_TYPE_INT16:
            case GGUF_TYPE_UINT32:
            case GGUF_TYPE_INT32:
            case GGUF_TYPE_FLOAT32:
            case GGUF_TYPE_UINT64:
            case GGUF_TYPE_INT64:
            case GGUF_TYPE_FLOAT64:
            case GGUF_TYPE_BOOL: {
              kv->value.arr.data = malloc(kv->value.arr.n * GGUF_TYPE_SIZE[kv->value.arr.type]);
              ok = ok && gguf_fread_el(gguf_file, kv->value.arr.data,
                                       kv->value.arr.n * GGUF_TYPE_SIZE[kv->value.arr.type], &offset);
            } break;
            case GGUF_TYPE_STRING: {
              kv->value.arr.data = malloc(kv->value.arr.n * sizeof(struct gguf_str));
              for (uint64_t j = 0; j < kv->value.arr.n; ++j) {
                ok = ok && gguf_fread_str(gguf_file, &((struct gguf_str*)kv->value.arr.data)[j], &offset);
              }
            } break;
            case GGUF_TYPE_ARRAY:
            case GGUF_TYPE_COUNT:
              printf("False && invalid type");
              break;  // NE_ASSERT(false && "invalid type"); break;
          }
        } break;
        case GGUF_TYPE_COUNT:
          printf("False && invalid type");  // NE_ASSERT(false && "invalid type");
      }

      if (!ok) {
        break;
      }
    }

    // read the tensor infos
    ctx->infos =
        reinterpret_cast<struct gguf_tensor_info*>(malloc(ctx->header.n_tensors * sizeof(struct gguf_tensor_info)));

    for (uint64_t i = 0; i < ctx->header.n_tensors; ++i) {
      struct gguf_tensor_info* info = &ctx->infos[i];

      for (int j = 0; j < GGML_MAX_DIMS; ++j) {
        info->ne[j] = 1;
      }

      ok = ok && gguf_fread_str(gguf_file, &info->name, &offset);
      ok = ok && gguf_fread_el(gguf_file, &info->n_dims, sizeof(info->n_dims), &offset);
      for (uint32_t j = 0; j < info->n_dims; ++j) {
        ok = ok && gguf_fread_el(gguf_file, &info->ne[j], sizeof(info->ne[j]), &offset);
      }
      ok = ok && gguf_fread_el(gguf_file, &info->type, sizeof(info->type), &offset);
      ok = ok && gguf_fread_el(gguf_file, &info->offset, sizeof(info->offset), &offset);

      if (!ok) {
        fprintf(stderr, "%s: failed to read tensor info\n", __func__);
        fclose(gguf_file);
        gguf_free(ctx);
        return nullptr;
      }

      model_load_tensor_shard shard;
      std::string name = gguf_get_tensor_name(ctx, i);
      uint32_t name_len = name.length();
      shard.type = (enum ne_type)info->type;

      uint32_t n_dims = info->n_dims;
      shard.ne.resize(n_dims);
      for (uint32_t j = 0; j < info->n_dims; ++j) {
        shard.ne[j] = info->ne[j];
      }

      if (n_dims < 1 || n_dims > 2) {
        throw format("model.cpp: tensor '%s' should not be %u-dimensional", name.c_str(), n_dims);
      }
      switch (shard.type) {
        case NE_TYPE_F32:
        case NE_TYPE_F16:
        case NE_TYPE_Q4_0:
        case NE_TYPE_Q4_1:
        case NE_TYPE_Q5_0:
        case NE_TYPE_Q5_1:
        case NE_TYPE_Q8_0:
        case NE_TYPE_Q6_K:
        case NE_TYPE_BTLA:
          break;
        default: {
          throw format("unrecognized tensor type %u\n", shard.type);
        }
      }

      shard.file_idx = 0;
      const size_t offs = file_offset(ctx, name.c_str());
      int length = info->ne[0] * info->ne[1] * info->ne[2] * info->ne[3] * 4;

      shard.file_off = offs;

      auto it = tensors_map.name_to_idx.find(name);
      size_t idx;
      if (it != tensors_map.name_to_idx.end()) {
        idx = it->second;
      } else {
        tensors_map.tensors.emplace_back(name);
        idx = tensors_map.tensors.size() - 1;
        tensors_map.name_to_idx.emplace(name, idx);
      }
      tensors_map.tensors.at(idx).shards.push_back(shard);
    }

    ctx->alignment = GGUF_DEFAULT_ALIGNMENT;

    int alignment_idx = gguf_find_key(ctx, "general.alignment");
    if (alignment_idx != -1) {
      ctx->alignment = gguf_get_val_u32(ctx, alignment_idx);
    }

    const size_t offset_pad = offset % ctx->alignment;

    if (offset_pad != 0) {
      offset += ctx->alignment - offset_pad;
      // fseek(file, offset, SEEK_SET);
    }

    ctx->offset = offset;
    gguf_data_offset = offset;

    return ctx;
  }

  void gguf_load_from_file(struct gguf_context* ctx_gguf, model_hparams& hparams, model_vocab& vocab) {
    int n_kv = 0;
    n_kv = gguf_get_n_kv(ctx_gguf);

    int n_tensors = 0;
    n_tensors = gguf_get_n_tensors(ctx_gguf);

    llama_fver fver;
    fver = (enum llama_fver)gguf_get_version(ctx_gguf);
    printf("%s: loaded meta data with %d key-value pairs and %d tensors (version %s)\n", __func__, n_kv, n_tensors,
           llama_file_version_name(fver));

    for (int i = 0; i < n_kv; i++) {
      const char* name = gguf_get_key(ctx_gguf, i);
      const enum gguf_type type = gguf_get_kv_type(ctx_gguf, i);
      const std::string type_name =
          type == GGUF_TYPE_ARRAY ? format("%s[%s,%d]", gguf_type_name(type),
                                           gguf_type_name(gguf_get_arr_type(ctx_gguf, i)), gguf_get_arr_n(ctx_gguf, i))
                                  : gguf_type_name(type);

      std::string value = gguf_kv_to_str(ctx_gguf, i);
      const size_t MAX_VALUE_LEN = 40;
      if (value.size() > MAX_VALUE_LEN) {
        value = format("%s...", value.substr(0, MAX_VALUE_LEN - 3).c_str());
      }
      replace_all(value, "\n", "\\n");

      printf("%s: - kv %3d: %42s %-16s = %s\n", __func__, i, name, type_name.c_str(), value.c_str());
    }

    // Get model name
    uint32_t general_architecture_idex = 0;
    std::string arch_name = gguf_kv_to_str(ctx_gguf, general_architecture_idex);
    llm_arch arch = llm_arch_from_string(arch_name);
    const auto kv = LLM_KV(arch);

    // Get general kv
    uint32_t magic = -1;
    uint32_t version = -1;
    GGUF_GET_KEY(ctx_gguf, magic, gguf_get_val_u32, GGUF_TYPE_UINT32, false, "magic");
    GGUF_GET_KEY(ctx_gguf, version, gguf_get_val_u32, GGUF_TYPE_UINT32, false, "version");

    // Get hparams kv
    GGUF_GET_KEY(ctx_gguf, hparams.n_vocab, gguf_get_val_u32, GGUF_TYPE_UINT32, false, "n_vocab");
    GGUF_GET_KEY(ctx_gguf, hparams.n_embd, gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_EMBEDDING_LENGTH));
    GGUF_GET_KEY(ctx_gguf, hparams.n_mult, gguf_get_val_u32, GGUF_TYPE_UINT32, false, "n_mult");
    GGUF_GET_KEY(ctx_gguf, hparams.n_head, gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_ATTENTION_HEAD_COUNT));
    GGUF_GET_KEY(ctx_gguf, hparams.n_head_kv, gguf_get_val_u32, GGUF_TYPE_UINT32, false,
                 kv(LLM_KV_ATTENTION_HEAD_COUNT_KV));
    GGUF_GET_KEY(ctx_gguf, hparams.n_layer, gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_BLOCK_COUNT));
    GGUF_GET_KEY(ctx_gguf, hparams.n_rot, gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_ROPE_DIMENSION_COUNT));

    GGUF_GET_KEY(ctx_gguf, hparams.rms_norm_eps, gguf_get_val_f32, GGUF_TYPE_FLOAT32, false,
                 kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS));
    GGUF_GET_KEY(ctx_gguf, hparams.freq_base, gguf_get_val_f32, GGUF_TYPE_FLOAT32, false, kv(LLM_KV_ROPE_FREQ_BASE));

    // Get NeuralSpeed ftype
    uint32_t ftype = 1;
    GGUF_GET_KEY(ctx_gguf, ftype, gguf_get_val_u32, GGUF_TYPE_UINT32, false, "ftype");
    hparams.ftype = (enum ne_ftype)ftype;

    // Get specific model paramters
    GGUF_GET_KEY(ctx_gguf, hparams.max_seq_len, gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_CONTEXT_LENGTH));
    GGUF_GET_KEY(ctx_gguf, hparams.alibi_bias_max, gguf_get_val_f32, GGUF_TYPE_FLOAT32, false,
                 kv(LLM_KV_ATTENTION_MAX_ALIBI_BIAS));
    GGUF_GET_KEY(ctx_gguf, hparams.clip_qkv, gguf_get_val_u32, GGUF_TYPE_UINT32, false, "clip_qkv");
    GGUF_GET_KEY(ctx_gguf, hparams.par_res, gguf_get_val_u32, GGUF_TYPE_UINT32, false, "par_res");

    GGUF_GET_KEY(ctx_gguf, hparams.word_embed_proj_dim, gguf_get_val_u32, GGUF_TYPE_UINT32, false,
                 "word_embed_proj_dim");
    GGUF_GET_KEY(ctx_gguf, hparams.do_layer_norm_before, gguf_get_val_u32, GGUF_TYPE_UINT32, false,
                 "do_layer_norm_before");

    GGUF_GET_KEY(ctx_gguf, hparams.multi_query_group_num, gguf_get_val_u32, GGUF_TYPE_UINT32, false,
                 "multi_query_group_num");
    GGUF_GET_KEY(ctx_gguf, hparams.ffn_hidden_size, gguf_get_val_u32, GGUF_TYPE_UINT32, false,
                 kv(LLM_KV_FEED_FORWARD_LENGTH));
    GGUF_GET_KEY(ctx_gguf, hparams.inner_hidden_size, gguf_get_val_u32, GGUF_TYPE_UINT32, false, "inner_hidden_size");

    // Get special vocab ids
    GGUF_GET_KEY(ctx_gguf, vocab.bos_token_id, gguf_get_val_u32, GGUF_TYPE_UINT32, false,
                 "tokenizer.ggml.bos_token_id");
    GGUF_GET_KEY(ctx_gguf, vocab.eos_token_id, gguf_get_val_u32, GGUF_TYPE_UINT32, false,
                 "tokenizer.ggml.eos_token_id");
    GGUF_GET_KEY(ctx_gguf, vocab.pad_token_id, gguf_get_val_u32, GGUF_TYPE_UINT32, false,
                 "tokenizer.ggml.pad_token_id");
    GGUF_GET_KEY(ctx_gguf, vocab.sep_token_id, gguf_get_val_u32, GGUF_TYPE_UINT32, false,
                 "tokenizer.ggml.sep_token_id");

    // load vocab
    std::string tokens = "tokenizer.ggml.tokens";
    const int token_idx = gguf_find_key(ctx_gguf, tokens.c_str());
    if (token_idx == -1) {
      throw std::runtime_error("cannot find tokenizer vocab in model file\n");
    }

    const float* scores = nullptr;
    std::string scores_name = "tokenizer.ggml.scores";
    const int score_idx = gguf_find_key(ctx_gguf, scores_name.c_str());
    if (score_idx != -1) {
      scores = (const float*)gguf_get_arr_data(ctx_gguf, score_idx);
    }

    uint32_t default_n_vocab = 32000;
    const uint32_t n_vocab = gguf_get_arr_n(ctx_gguf, token_idx);
    if ((hparams.n_vocab == default_n_vocab) && (hparams.n_vocab != n_vocab)) {
      hparams.n_vocab = n_vocab;
    }

    vocab.id_to_token.resize(hparams.n_vocab);
    for (uint32_t i = 0; i < n_vocab; i++) {
      std::string word = gguf_get_arr_str(ctx_gguf, token_idx, i);
      // NE_ASSERT(codepoints_from_utf8(word).size() > 0);

      vocab.token_to_id[word] = i;

      auto& tok_score = vocab.id_to_token[i];
      tok_score.tok = std::move(word);
      tok_score.score = scores ? scores[i] : 0.0f;
    }
  }
};

struct model_file_loader {
  model_file file;
  model_file_version file_version;
  model_hparams hparams;
  model_vocab vocab;

  size_t gguf_data_offset = 0;  // offset of the GGUF tensor data from the beginning of the file.
  enum model_format model_magic = UNKNOWN;

  model_file_loader(const char* fname, size_t file_idx, model_load_tensors_map& tensors_map) : file(fname, "rb") {
    fprintf(stderr, "model.cpp: loading model from %s\n", fname);
    model_magic = read_file_magic();
    if (model_magic == GGUF) {
      std::cout << "Loading the bin file with GGUF format..." << std::endl;
      fseek(file.fp, 0, SEEK_SET);
      model_magic = GGUF;

      gguf_loader gguf_loader(file.fp);

      struct gguf_context* ctx_gguf = NULL;
      ctx_gguf = gguf_loader.gguf_init_from_file(tensors_map, gguf_data_offset);
      if (!ctx_gguf) {
        throw std::runtime_error(format("%s: failed to load model\n", __func__));
      }

      gguf_loader.gguf_load_from_file(ctx_gguf, hparams, vocab);
    } else if (model_magic == NE) {
      std::cout << "Loading the bin file with NE format..." << std::endl;
      fseek(file.fp, 0, SEEK_SET);
      read_ne_magic();
      read_ne_hparams();
      read_ne_vocab();
      read_tensor_metadata(file_idx, tensors_map);
    } else {
      throw format("unknown file format model_maigc = %d", model_magic);
    }
  }

  void read_ne_magic() {
    uint32_t magic = file.read_u32();

    if (magic == MODEL_FILE_MAGIC_NE) {
      file_version = MODEL_FILE_VERSION_NE;
      return;
    }

    uint32_t version = file.read_u32();

    switch (magic) {
      case MODEL_FILE_MAGIC_GGMF:
        switch (version) {
          case 1:
            file_version = MODEL_FILE_VERSION_GGMF_V1;
            return;
        }
        break;
      case MODEL_FILE_MAGIC_GGJT:
        switch (version) {
          case 1:
            file_version = MODEL_FILE_VERSION_GGJT_V1;
            return;
          case 2:
            file_version = MODEL_FILE_VERSION_GGJT_V2;
            return;
          case 3:
            file_version = MODEL_FILE_VERSION_GGJT_V3;
            return;
        }
    }

    throw format("unknown (magic, version) combination: %08x, %08x; is this really a NE file?", magic, version);
  }

  enum model_format read_file_magic() {
    char gguf_magic[4];
    const size_t n = fread(&gguf_magic, 1, sizeof(gguf_magic), file.fp);
    bool ok = true;
    ok = ok & gguf_magic[0] == 'G';
    ok = ok & gguf_magic[1] == 'G';
    ok = ok & gguf_magic[2] == 'U';
    ok = ok & gguf_magic[3] == 'F';

    if (ok) {
      model_magic = GGUF;
    } else {
      model_magic = NE;
    }
    return model_magic;
  }

  void read_ne_hparams() {
    hparams.n_vocab = file.read_u32();
    hparams.n_embd = file.read_u32();
    hparams.n_mult = file.read_u32();
    hparams.n_head = file.read_u32();
    hparams.n_head_kv = file.read_u32();
    hparams.n_layer = file.read_u32();
    hparams.n_rot = file.read_u32();
    hparams.ftype = (enum ne_ftype)file.read_u32();
    hparams.max_seq_len = file.read_u32();
    file.read_raw(&hparams.alibi_bias_max, sizeof(float));
    file.read_raw(&hparams.clip_qkv, sizeof(float));
    hparams.par_res = file.read_u32();

    hparams.word_embed_proj_dim = file.read_u32();
    hparams.do_layer_norm_before = bool(file.read_u32());

    // For ChatGLM-2
    hparams.multi_query_group_num = file.read_u32();
    hparams.ffn_hidden_size = file.read_u32();

    // For ChatGLM-2
    hparams.inner_hidden_size = file.read_u32();

    file.read_raw(&hparams.rms_norm_eps, sizeof(float));
    file.read_raw(&hparams.freq_base, sizeof(float));
    file.read_raw(&hparams.freq_scale, sizeof(float));
  }

  void read_ne_vocab() {
    file.read_raw(&vocab.bos_token_id, sizeof(model_vocab::id));
    file.read_raw(&vocab.eos_token_id, sizeof(model_vocab::id));
    file.read_raw(&vocab.pad_token_id, sizeof(model_vocab::id));
    file.read_raw(&vocab.sep_token_id, sizeof(model_vocab::id));

    vocab.id_to_token.resize(hparams.n_vocab);
    for (uint32_t i = 0; i < hparams.n_vocab; i++) {
      uint32_t len = file.read_u32();
      std::string word = file.read_string(len);

      float score = 0.0f;
      if (file_version >= MODEL_FILE_VERSION_GGMF_V1) {
        file.read_raw(&score, sizeof(score));
      }

      vocab.token_to_id[word] = i;

      auto& tok_score = vocab.id_to_token[i];
      tok_score.tok = std::move(word);
      tok_score.score = score;
    }
  }
  void read_tensor_metadata(size_t file_idx, model_load_tensors_map& tensors_map) {
    while (file.tell() < file.size) {
      model_load_tensor_shard shard;
      uint32_t n_dims = file.read_u32();
      uint32_t name_len = file.read_u32();
      shard.type = (enum ne_type)file.read_u32();
      shard.ne.resize(n_dims);
      file.read_raw(shard.ne.data(), sizeof(shard.ne[0]) * n_dims);
      std::string name = file.read_string(name_len);
      if (n_dims < 1 || n_dims > 2) {
        throw format("model.cpp: tensor '%s' should not be %u-dimensional", name.c_str(), n_dims);
      }
      switch (shard.type) {
        case NE_TYPE_F32:
        case NE_TYPE_F16:
        case NE_TYPE_Q4_0:
        case NE_TYPE_Q4_1:
        case NE_TYPE_Q5_0:
        case NE_TYPE_Q5_1:
        case NE_TYPE_Q8_0:
        case NE_TYPE_BTLA:
          break;
        default: {
          throw format("unrecognized tensor type %u\n", shard.type);
        }
      }

      if (file_version >= MODEL_FILE_VERSION_GGJT_V1) {
        // skip to the next multiple of 32 bytes
        file.seek(-static_cast<ptrdiff_t>(file.tell()) & 31, SEEK_CUR);
      }
      shard.file_idx = file_idx;
      shard.file_off = file.tell();
      if (shard.type == NE_TYPE_BTLA) {
        size_t size = 0;
        file.read_raw(&size, sizeof(size_t));
        shard.size = size;
        file.seek(shard.size - sizeof(size_t), SEEK_CUR);
      } else {
        shard.calc_size();
        file.seek(shard.size, SEEK_CUR);
      }

      auto it = tensors_map.name_to_idx.find(name);
      size_t idx;
      if (it != tensors_map.name_to_idx.end()) {
        idx = it->second;
      } else {
        tensors_map.tensors.emplace_back(name);
        idx = tensors_map.tensors.size() - 1;
        tensors_map.name_to_idx.emplace(name, idx);
      }
      tensors_map.tensors.at(idx).shards.push_back(shard);
    }
  }
};

struct model_file_saver {
  model_file file;
  model_file_loader* any_file_loader;
  model_file_saver(const char* fname, model_file_loader* any_file_loader, enum ne_ftype new_ftype)
      : file(fname, "wb"), any_file_loader(any_file_loader) {
    fprintf(stderr, "model.cpp: saving model to %s\n", fname);
    write_magic();
    write_hparams(new_ftype);
    write_vocab();
  }
  void write_magic() {
    file.write_u32(MODEL_FILE_MAGIC);    // magic
    file.write_u32(MODEL_FILE_VERSION);  // version
  }
  void write_hparams(enum ne_ftype new_ftype) {
    const model_hparams& hparams = any_file_loader->hparams;
    file.write_u32(hparams.n_vocab);
    file.write_u32(hparams.n_embd);
    file.write_u32(hparams.n_mult);
    file.write_u32(hparams.n_head);
    file.write_u32(hparams.n_head_kv);
    file.write_u32(hparams.n_layer);
    file.write_u32(hparams.n_rot);
    file.write_u32(hparams.ftype);
    file.write_u32(hparams.max_seq_len);
    file.write_raw(&hparams.alibi_bias_max, sizeof(float));
    file.write_raw(&hparams.clip_qkv, sizeof(float));
    file.write_u32(hparams.par_res);
    file.write_u32(hparams.word_embed_proj_dim);
    file.write_u32(static_cast<int>(hparams.do_layer_norm_before));

    file.write_u32(hparams.multi_query_group_num);
    file.write_u32(hparams.ffn_hidden_size);
    file.write_u32(hparams.inner_hidden_size);

    file.write_raw(&hparams.rms_norm_eps, sizeof(float));
    file.write_raw(&hparams.freq_base, sizeof(float));
    file.write_raw(&hparams.freq_scale, sizeof(float));
  }
  void write_vocab() {
    if (any_file_loader->file_version == MODEL_FILE_VERSION_NE) {
      fprintf(stderr, "model.cpp: WARNING: input is an old file that doesn't have scores; will add dummy scores\n");
    }
    uint32_t n_vocab = any_file_loader->hparams.n_vocab;
    file.write_raw(&(any_file_loader->vocab.bos_token_id), sizeof(model_vocab::id));
    file.write_raw(&(any_file_loader->vocab.eos_token_id), sizeof(model_vocab::id));
    file.write_raw(&(any_file_loader->vocab.pad_token_id), sizeof(model_vocab::id));
    file.write_raw(&(any_file_loader->vocab.sep_token_id), sizeof(model_vocab::id));
    for (uint32_t i = 0; i < n_vocab; i++) {
      const auto& token_score = any_file_loader->vocab.id_to_token.at(i);
      file.write_u32((uint32_t)token_score.tok.size());
      file.write_raw(token_score.tok.data(), token_score.tok.size());
      file.write_raw(&token_score.score, sizeof(token_score.score));
    }
  }
  void write_tensor(model_load_tensor& tensor, enum ne_type new_type, const void* new_data, size_t new_size) {
    switch (new_type) {
      case NE_TYPE_F32:
      case NE_TYPE_F16:
      case NE_TYPE_Q4_0:
      case NE_TYPE_Q4_1:
      case NE_TYPE_Q5_0:
      case NE_TYPE_Q5_1:
      case NE_TYPE_Q8_0:
      case NE_TYPE_BTLA:
        break;
      default:
        MODEL_ASSERT(false);
    }
    file.write_u32((uint32_t)tensor.ne.size());
    file.write_u32((uint32_t)tensor.name.size());
    file.write_u32(new_type);
    file.write_raw(tensor.ne.data(), sizeof(tensor.ne[0]) * tensor.ne.size());
    file.write_raw(tensor.name.data(), tensor.name.size());
    file.seek(-static_cast<ptrdiff_t>(file.tell()) & 31, SEEK_CUR);
    if (new_type != NE_TYPE_BTLA) MODEL_ASSERT(new_size == model_calc_tensor_size(tensor.ne, new_type));
    file.write_raw(new_data, new_size);
  }
};

struct model_model_loader {
  std::vector<std::unique_ptr<model_file_loader>> file_loaders;
  model_load_tensors_map tensors_map;
  bool use_mmap;
  size_t num_ne_tensors_created = 0;
  struct ne_context* ne_ctx = nullptr;
  std::unique_ptr<model_mmap> mapping;

  model_model_loader(const std::string& fname_base, bool use_mmap, bool vocab_only) {
    auto* first_file = new model_file_loader(fname_base.c_str(), 0, tensors_map);
    file_loaders.emplace_back(first_file);
    uint32_t n_parts = vocab_only ? 1 : guess_n_parts();
    for (uint32_t i = 1; i < n_parts; i++) {
      std::string fname = fname_base + "." + std::to_string(i);
      auto* ith_file = new model_file_loader(fname.c_str(), i, tensors_map);
      file_loaders.emplace_back(ith_file);
      if (ith_file->hparams != first_file->hparams) {
        throw format("model.cpp: hparams inconsistent between files");
      }
    }
    if (!model_mmap::SUPPORTED) {
      use_mmap = false;
    }
    if (use_mmap && alignment_prevents_mmap()) {
      fprintf(stderr,
              "model.cpp: can't use mmap because tensors are not aligned; convert to new format to avoid this\n");
      use_mmap = false;
    }
    this->use_mmap = use_mmap;
    for (model_load_tensor& lt : tensors_map.tensors) {
      lt.calc_all();
    }
  }

  bool alignment_prevents_mmap() {
    for (const model_load_tensor& lt : tensors_map.tensors) {
      for (const model_load_tensor_shard& shard : lt.shards) {
        if (shard.file_off & 3) {
          return true;
        }
      }
    }
    return false;
  }

  uint32_t guess_n_parts() const {
    auto it = tensors_map.name_to_idx.find("tok_embeddings.weight");
    if (it == tensors_map.name_to_idx.end()) {
      it = tensors_map.name_to_idx.find("transformer.wte.weight");
      if (it == tensors_map.name_to_idx.end()) {
        it = tensors_map.name_to_idx.find("gpt_neox.embed_in.weight");
        if (it == tensors_map.name_to_idx.end()) {
          it = tensors_map.name_to_idx.find("model/wte");
          if (it == tensors_map.name_to_idx.end()) {
            it = tensors_map.name_to_idx.find("token_embd.weight");  // llama-2-chat-hf
            if (it == tensors_map.name_to_idx.end()) {
              it = tensors_map.name_to_idx.find("model.embed_tokens.weight");  // baichuan13B
              if (it == tensors_map.name_to_idx.end()) {
                it = tensors_map.name_to_idx.find("transformer.word_embeddings.weight");  // ChatGLM-1
                if (it == tensors_map.name_to_idx.end()) {
                  it = tensors_map.name_to_idx.find("transformer.embedding.word_embeddings.weight");  // ChatGLM-2
                  if (it == tensors_map.name_to_idx.end()) {
                    it = tensors_map.name_to_idx.find("model.decoder.embed_tokens.weight");
                    if (it != tensors_map.name_to_idx.end()) return 1;  // hacky solution for OPT loading
                    if (it == tensors_map.name_to_idx.end()) {
                      throw std::string("missing tok_embeddings.weight");
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    const model_load_tensor& lt = tensors_map.tensors.at(it->second);
    return file_loaders.at(0)->hparams.n_embd / lt.shards.at(0).ne.at(0);
  }

  void calc_sizes(size_t* ctx_size_p, size_t* mmapped_size_p) const {
    *ctx_size_p = *mmapped_size_p = 0;
    size_t size_needed = 0;
    for (const model_load_tensor& lt : tensors_map.tensors) {
      *ctx_size_p += sizeof(struct ne_tensor) + NE_OBJECT_SIZE;
      if (lt.type == NE_TYPE_BTLA) {
        size_needed = lt.size;
      } else {
        size_needed = (lt.size + NE_MEM_ALIGN - 1) / NE_MEM_ALIGN * NE_MEM_ALIGN;
      }
      *(use_mmap ? mmapped_size_p : ctx_size_p) += size_needed;
    }
  }

  bool verify_tensor(const std::string& name) {
    auto it = tensors_map.name_to_idx.find(name);
    if (it == tensors_map.name_to_idx.end()) {
      return false;
    }

    return true;
  }

  struct ne_tensor* get_tensor(const std::string& name, const std::vector<uint32_t>& ne, ne_backend backend) {
    auto it = tensors_map.name_to_idx.find(name);
    if (it == tensors_map.name_to_idx.end()) {
      throw format("model.cpp: tensor '%s' is missing from model", name.c_str());
    }
    model_load_tensor& lt = tensors_map.tensors.at(it->second);
#ifdef NS_TP_MODEL
    if (lt.enable_tp && (lt.split_type == TP_1D_ROW || lt.split_type == TP_1D_COLUMN ||
                         lt.split_type == TP_1D_QKV_ROW || lt.split_type == TP_1D_QKV_COLUMN)) {
      // check the split dim
      size_t split_dim_size, origin_dim_size;
      if (lt.split_type == TP_1D_ROW || lt.split_type == TP_1D_QKV_ROW) {
        split_dim_size = lt.ne.size() == 1 ? lt.ne.at(0) : lt.ne.at(1);
        origin_dim_size = ne.size() == 1 ? ne.at(0) : ne.at(1);
      } else {
        split_dim_size = lt.ne.at(0);
        origin_dim_size = ne.at(0);
      }
      MODEL_ASSERT(split_dim_size == origin_dim_size / lt.world_size);
      return get_tensor_for(lt, backend);
    }
#endif
    if (lt.ne != ne) {
      throw format("model.cpp: tensor '%s' has wrong shape; expected %s, got %s", name.c_str(),
                   model_format_tensor_shape(ne).c_str(), model_format_tensor_shape(lt.ne).c_str());
    }

    return get_tensor_for(lt, backend);
  }

  struct ne_tensor* get_tensor_for(model_load_tensor& lt, ne_backend backend) {
    struct ne_tensor* tensor;
    if (lt.ne.size() == 2) {
      if (lt.type == NE_TYPE_BTLA) {
        tensor = ne_new_tensor_2d(ne_ctx, lt.type, lt.ne.at(0), lt.ne.at(1), lt.size);
      } else {
        tensor = ne_new_tensor_2d(ne_ctx, lt.type, lt.ne.at(0), lt.ne.at(1), NE_SIZE_CALC);
      }
    } else {
      MODEL_ASSERT(lt.ne.size() == 1);
      tensor = ne_new_tensor_1d(ne_ctx, lt.type, lt.ne.at(0), NE_SIZE_CALC);
    }
    ne_set_name(tensor, lt.name.c_str());
    MODEL_ASSERT(lt.ne_tensor == nullptr);  // if this fails, we called get_tensor twice on the same tensor
    tensor->backend = backend;
    lt.ne_tensor = tensor;
    num_ne_tensors_created++;
    return tensor;
  }

  void done_getting_tensors() const {
    if (num_ne_tensors_created != tensors_map.tensors.size()) {
      throw std::string("model.cpp: file contained more tensors than expected");
    }
  }

  void load_all_data(model_progress_callback progress_callback, void* progress_callback_user_data,
                     model_mlock* lmlock) {
    size_t data_size = 0;
    size_t prefetch_size = 0;
    for (const model_load_tensor& lt : tensors_map.tensors) {
      data_size += lt.size;
      if (lt.ne_tensor->backend == NE_BACKEND_CPU) {
        prefetch_size += lt.size;
      }
    }

    if (use_mmap) {
      mapping.reset(new model_mmap(&file_loaders.at(0)->file, prefetch_size));
      if (!lmlock) {
        // Don't call the callback since the actual loading will be lazy
        // and we can't measure it.
        progress_callback = nullptr;
      }
      if (lmlock) {
        lmlock->init(mapping->addr);
      }
    }

    size_t done_size = 0;
    for (model_load_tensor& lt : tensors_map.tensors) {
      if (lt.ne_tensor->backend != NE_BACKEND_CPU) {
        continue;
      }
      if (progress_callback) {
        progress_callback((float)done_size / data_size, progress_callback_user_data);
      }
      MODEL_ASSERT(lt.ne_tensor);  // unused tensors should have been caught by load_data already
      lt.data = (uint8_t*)lt.ne_tensor->data;
      load_data_for(lt);
      lt.ne_tensor->data = lt.data;
      done_size += lt.size;
      if (use_mmap && lmlock) {
        lmlock->grow_to(done_size);
      }
    }
  }

  void bestla_split_weight(void** src, void** dst, size_t src_n, size_t src_k, size_t dst_n, size_t dst_k,
                           size_t n_rank, size_t k_rank, bool qkv_fusion = false) {
    auto src_fp32 = (float*)malloc(src_n * src_k * sizeof(float));
    if (src_fp32 == nullptr) {
      assert(0);
    }
    bestla_unpackweight_fp32(*src, src_n, src_k, src_fp32, src_n);
    // layout will be K * N in the buffer
    float* dst_fp32;
    if (qkv_fusion) {
      dst_fp32 = (float*)malloc(dst_n * dst_k * sizeof(float));
      for (int i = 0; i < src_k; ++i) {
        for (int j = 0; j < 3; ++j) {
          float* dst_off = dst_fp32 + dst_n * i + j * dst_n / 3;
          float* src_off = src_fp32 + src_n * i + j * src_n / 3 + n_rank * dst_n / 3;
          memcpy(dst_off, src_off, dst_n * sizeof(float) / 3);
        }
      }
      bestla_packweight_copyattr(dst_fp32, *dst, dst_n, dst_k, dst_n, *src);
      free(dst_fp32);
    } else {
      dst_fp32 = src_fp32 + k_rank * dst_k * src_n + n_rank * dst_n;
      bestla_packweight_copyattr(dst_fp32, *dst, dst_n, dst_k, src_n, *src);
    }
    free(src_fp32);
  }
  void load_data_for(model_load_tensor& lt) {
    if (use_mmap) {
      MODEL_ASSERT(lt.shards.size() == 1);
      lt.data = (uint8_t*)mapping->addr + lt.shards.at(0).file_off;
    } else if (lt.split_type == SPLIT_NONE) {
      model_file& file = file_loaders.at(lt.shards.at(0).file_idx)->file;
      file.seek(lt.shards.at(0).file_off + file_loaders.at(0)->gguf_data_offset, SEEK_SET);
      file.read_raw(lt.data, lt.size);
    } else if (lt.split_type == SPLIT_BY_ROWS) {
      size_t offset = 0;
      for (model_load_tensor_shard& shard : lt.shards) {
        model_file& file = file_loaders.at(shard.file_idx)->file;
        file.seek(shard.file_off, SEEK_SET);
        file.read_raw(lt.data + offset, shard.size);
        offset += shard.size;
      }
      MODEL_ASSERT(offset == lt.size);
    } else if (lt.split_type == SPLIT_BY_COLUMNS) {
      // Let's load the data into temporary buffers to ensure the OS performs large loads.
      std::vector<model_buffer> tmp_bufs(lt.shards.size());
      for (size_t i = 0; i < lt.shards.size(); i++) {
        model_load_tensor_shard& shard = lt.shards.at(i);
        model_file& file = file_loaders.at(shard.file_idx)->file;
        file.seek(shard.file_off, SEEK_SET);
        tmp_bufs.at(i).resize(shard.size);
        file.read_raw(tmp_bufs.at(i).addr, shard.size);
      }
      // Then reshape.
      size_t num_rows = lt.ne.at(1);
      size_t per_shard_row_size = lt.shards.at(0).size / num_rows;
      size_t out_offset = 0;
      for (size_t row = 0; row < num_rows; row++) {
        for (model_buffer& tmp_buf : tmp_bufs) {
          memcpy(lt.data + out_offset, tmp_buf.addr + row * per_shard_row_size, per_shard_row_size);
          out_offset += per_shard_row_size;
        }
      }
      MODEL_ASSERT(out_offset == lt.size);
    }
#ifdef NS_TP_MODEL
    else if (lt.split_type == TP_1D_ROW || lt.split_type == TP_1D_QKV_ROW) {
      model_load_tensor_shard& shard = lt.shards.at(0);
      model_buffer tmp_buf;
      model_file& file = file_loaders.at(shard.file_idx)->file;
      file.seek(shard.file_off, SEEK_SET);
      size_t num_rows = lt.ne.size() == 1 ? 1 : lt.ne.at(1);
      if (lt.type == NE_TYPE_BTLA) {
        tmp_buf.resize(shard.size);
        file.read_raw(tmp_buf.addr, shard.size);
        void* dst_data = (void*)lt.data;
        void* src_data = (void*)(tmp_buf.addr);
        bestla_split_weight(&src_data, &dst_data, lt.world_size * num_rows, lt.ne.at(0), num_rows, lt.ne.at(0), lt.rank,
                            0, lt.split_type == TP_1D_QKV_ROW);
      } else {
        // only copy part of weight form the tmp_buf of origin file
        tmp_buf.resize(lt.size * lt.world_size);
        file.read_raw(tmp_buf.addr, lt.size * lt.world_size);
        if (lt.split_type == TP_1D_QKV_ROW) {
          for (int j = 0; j < 3; ++j) {
            auto dst_off = lt.data + j * lt.size / 3;
            auto src_off = tmp_buf.addr + (lt.rank + j * lt.world_size) * lt.size / 3;
            memcpy(dst_off, src_off, lt.size / 3);
          }
        } else {
          memcpy(lt.data, tmp_buf.addr + lt.rank * lt.size, lt.size);
        }
      }
    } else if (lt.split_type == TP_1D_COLUMN || lt.split_type == TP_1D_QKV_COLUMN) {
      if (lt.size == 0) {
        return;
      }
      model_load_tensor_shard& shard = lt.shards.at(0);
      model_buffer tmp_buf;
      model_file& file = file_loaders.at(shard.file_idx)->file;
      file.seek(shard.file_off, SEEK_SET);
      size_t num_rows = lt.ne.size() == 1 ? 1 : lt.ne.at(1);
      if (lt.type == NE_TYPE_BTLA) {
        tmp_buf.resize(shard.size);
        file.read_raw(tmp_buf.addr, shard.size);
        void* dst_data = (void*)lt.data;
        void* src_data = (void*)(tmp_buf.addr);
        // TODO support QKV COLUMN in bestla
        bestla_split_weight(&src_data, &dst_data, num_rows, lt.world_size * lt.ne.at(0), num_rows, lt.ne.at(0), 0,
                            lt.rank);
      } else {
        tmp_buf.resize(lt.size * lt.world_size);
        file.read_raw(tmp_buf.addr, lt.size * lt.world_size);
        size_t offset = 0;
        // different data type may have differnet per_row_size
        size_t per_row_size = lt.size / num_rows;
        if (lt.split_type == TP_1D_QKV_COLUMN) {
          for (size_t i = 0; i < num_rows; ++i) {
            for (int j = 0; j < 3; ++j) {
              auto dst_off = lt.data + i * per_row_size + j * per_row_size / 3;
              auto src_off = tmp_buf.addr + (lt.rank / 3 + j * lt.world_size / 3 + i * lt.world_size) * per_row_size;
              memcpy(dst_off, src_off, per_row_size / 3);
            }
            offset += per_row_size;
          }
        } else {
          for (size_t i = 0; i < num_rows; ++i) {
            memcpy(lt.data + offset, tmp_buf.addr + lt.rank * per_row_size + i * lt.world_size * per_row_size,
                   per_row_size);
            offset += per_row_size;
          }
        }
        MODEL_ASSERT(offset == lt.size);
      }
    } else if (lt.split_type == TP_1D_ONLY_MASTER) {
      // only master node load the tensor, other node set to zero
      model_file& file = file_loaders.at(lt.shards.at(0).file_idx)->file;
      file.seek(lt.shards.at(0).file_off, SEEK_SET);
      if (lt.rank == 0) {
        file.read_raw(lt.data, lt.size);
      } else {
        memset(lt.data, 0, lt.size);
      }
    }
#endif
    if (0) {
      print_checksum(lt);
    }
  }

  static void print_checksum(model_load_tensor& lt) {
    uint32_t sum = 0;
    for (size_t i = 0; i < lt.size; i++) {
      uint8_t byte = lt.data[i];
      sum = byte + (sum << 6) + (sum << 16) - sum;  // sdbm hash
    }
    fprintf(stderr, "%s checksum: %#08x (%s, size %zu)\n", lt.name.c_str(), sum,
            model_format_tensor_shape(lt.ne).c_str(), lt.size);
  }
};

#endif  // MODEL_FILES_H
