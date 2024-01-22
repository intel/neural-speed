#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import argparse
from .common import *
from tqdm import tqdm
import gguf
from transformers import AutoTokenizer

def permute_func(weights, n_head: int, n_head_kv: int):
    if n_head_kv is not None and n_head != n_head_kv:
        n_head //= n_head_kv
    return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2,
                            *weights.shape[1:]).swapaxes(1, 2).reshape(weights.shape))

def convert_fp32_tensor(src_name, dst_name, model, fout, n_head=0, n_head2=0, permute_func=None):
    if ".weight" not in src_name:
        src_name = src_name + ".weight"
    v = model[src_name]
    shape = v.shape
    # print("Processing non-Q4 variable: " + src_name +
    #       " with shape: ", shape, " and type: ", v.dtype)
    v = v.to(torch.float32)

    ftype_cur = {torch.float16: 1, torch.float32: 0}[v.dtype]

    # header
    write_header(fout, shape, dst_name, ftype_cur)
    if permute_func:
        v = permute_func(v, n_head, n_head2).contiguous()
    # data
    v.numpy().tofile(fout)

def quantize_ggml_tensor(src_name, dst_name, model, fout, q_config, n_head=0, n_head_kv=0, permute_func=None):
    if ".weight" not in src_name:
        src_name = src_name + ".weight"
    v = model[src_name]
    shape = v.shape
    # print("Processing non-Q4 variable: " + src_name +
    #       " with shape: ", shape, " and type: ", v.dtype)
    v = v.to(torch.float32)

    if permute_func:
        v = permute_func(v, n_head, n_head_kv).contiguous()

    qv = quantize_q4_0(v)
    ftype_cur = GGML_QK4_0_TYPE
    
    if isinstance(fout, gguf.gguf_writer.GGUFWriter):
        fout.add_tensor(dst_name, qv.numpy(), raw_shape=v.shape, raw_dtype=ftype_cur)
    else:
        # header
        write_header(fout, shape, dst_name, ftype_cur)

        # data
        qv.numpy().tofile(fout)
    print(f"quantizing {dst_name} float to q4_0 tensor")

def quantize_jblas_tensor(src_name, dst_name, model, fout, q_config, n_head, n_head_kv=0, permute_func=None):
    import neural_speed.llama_cpp as cpp_model
    if ".weight" not in src_name:
        src_name = src_name + ".weight"
    v = model[src_name]
    shape = v.shape
    v = v.to(torch.float32)

    if permute_func:
        v = permute_func(v, n_head, n_head_kv).contiguous()

    ftype_cur = GGML_QJBLAS_TYPE

    # header
    write_header(fout, shape, dst_name, ftype_cur)

    # pack int weight in bestla format
    dst = np.zeros((v.shape[0], v.shape[1] * 4), dtype=np.int8)
    byte_size = cpp_model.Model.np_bestla_quantize(v.numpy(), dst,
                                               weight_dtype=q_config.weight_dtype,
                                               group_size=q_config.group_size,
                                               alg=q_config.alg,
                                               compute_dtype=q_config.compute_dtype)
    dst.flatten()[:byte_size].tofile(fout)

def convert_quantized_tensor(src_name, dst_name, model, fout, q_config, n_head=0, n_head_kv=0, permute_func=None):
    # unpack weight and repack into jblas format
    import neural_speed.llama_cpp as cpp_model
    import pdb; pdb.set_trace()
    qzeros = model[f"{src_name}.qzeros"]
    zeros = qzeros_to_zeros(qzeros)
    scales = model[f"{src_name}.scales"]
    qweight = model[f"{src_name}.qweight"]

    int_weight, gptq_scales, gptq_zeros = unpack_weight(qweight, scales, qzeros, q_config)
    int_weight = int_weight.view(-1,int_weight.shape[-1])

    # permute_func for llama-like model
    if permute_func:
        int_weight = permute_func(int_weight.t(), n_head, n_head_kv).t().contiguous()
        gptq_scales = permute_func(gptq_scales.t(), n_head, n_head_kv).t().contiguous()
        gptq_zeros = permute_func(gptq_zeros.t(), n_head, n_head_kv).t().contiguous()

    # shuffle weight in GPTQ when act order is on
    if 'desc_act'in q_config and q_config['desc_act']:
        g_idx = model[f"{src_name}.g_idx"]
        int_weight2 = int_weight.clone()
        group_size=q_config['group_size']
        group_dict = {}
        for i in range(len(g_idx)):
            group_idx = g_idx[i].item()
            if group_idx not in group_dict:
                target_idx = group_idx * group_size
                group_dict[group_idx] = 0
            else:
                group_dict[group_idx] = group_dict[group_idx] + 1
                target_idx = group_idx * group_size + group_dict[group_idx]
            int_weight2[target_idx] = int_weight[i]
        int_weight = int_weight2

    shape = int_weight.shape
    write_header(fout, shape[::-1], dst_name, GGML_QJBLAS_TYPE)

    if q_config['bits'] == 4:
        int_weight = (int_weight - 8) * 16
        gptq_scales = gptq_scales / 16
        gptq_zeros = (gptq_zeros - 8) * 16
    dst = np.zeros((int_weight.shape[0], int_weight.shape[1] * 4), dtype=np.int8)
    int_weight = np.ascontiguousarray(int_weight.numpy())
    gptq_scales = np.ascontiguousarray((gptq_scales.float()).numpy())
    if q_config['sym']:
        gptq_zeros = np.empty(0, dtype=np.int8)
    else:
        gptq_zeros = np.ascontiguousarray(gptq_zeros.numpy())
    if 'desc_act'in q_config and q_config['desc_act']:
        g_idx = np.ascontiguousarray(g_idx.numpy())
    else:
        g_idx = np.empty(0, dtype=np.int32)

    # pack int weight in bestla format
    byte_size = cpp_model.Model.np_bestla_qpack(int_weight, gptq_scales, gptq_zeros, g_idx, dst,
                                               weight_dtype="int4" if q_config['bits'] == 4 else "int8",
                                               group_size=q_config['group_size'],
                                               alg="sym" if q_config['sym'] else "asym",
                                               compute_dtype="int8")
    dst.flatten()[:byte_size].tofile(fout)

def convert_llama_ggml(model_path, out_path, quant_config):
    print(quant_config)
    # TODO(zhenweil): refact load and convert function
    convert_func = convert_fp32_tensor
    if not quant_config.not_quant:
        if quant_config.use_ggml:
            convert_func = quantize_ggml_tensor
        else:
            convert_func = quantize_jblas_tensor

    if quant_config.use_gptq or quant_config.use_awq:
        convert_func = convert_quantized_tensor
        model, config, quantize_config = load_quantized_model(model_path)
        quant_config = quantize_config
    else:
        model, config, quantize_config = load_hf_model(model_path)
        config = config.to_dict()
        model = model.state_dict()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    f = open(out_path, "wb")

    # 1. write hparams
    n_vocab = config["vocab_size"]
    n_embd = config["hidden_size"]
    n_layer = config["num_hidden_layers"]
    n_head = config["num_attention_heads"]
    ffn_hidden_size = config["intermediate_size"]

    # hardcoded:
    n_mult = 256

    # 1. write head and params
    f.write(b"ggjt"[::-1])  # magic

    n_head = n_head
    n_head_kv = n_head
    values = [
        1,  # file version
        n_vocab,
        n_embd,
        256,  #hparams.n_mult,
        n_head,
        n_head_kv,  # n_head_kv (multi_query attention)
        n_layer,
        n_embd // n_head,  # rot (obsolete)
        0,  #file_type.value, # TODO
    ]
    f.write(struct.pack("i" * len(values), *values))
    f.write(struct.pack("i", 0))
    f.write(struct.pack("f", 0))
    f.write(struct.pack("f", 0))
    f.write(struct.pack("i", 0))
    f.write(struct.pack("i", 0))  # word_embed_proj_dim (for opt)
    f.write(struct.pack("i", 0))  # do_layer_norm_before (for opt)

    f.write(struct.pack("i", 0))
    f.write(struct.pack("i", ffn_hidden_size))
    f.write(struct.pack("i", 0))

    f.write(struct.pack("f", config["rms_norm_eps"]))
    f.write(struct.pack("f", config["rope_theta"] if "rope_theta" in config else 10000))
    rope_scale = 1
    if "rope_scaling" in config and config["rope_scaling"] is not None:
        rope_scale = config["rope_scaling"]["factor"] if "factor" in config["rope_scaling"] else 1
    f.write(struct.pack("f", rope_scale))

    # TODO, bos_token_id = 0 in https://huggingface.co/decapoda-research/llama-7b-hf/blob/main/config.json
    # but bos_token_id = 1 in llama.cpp
    f.write(struct.pack("i", config["bos_token_id"]))  
    f.write(struct.pack("i", config["eos_token_id"]))

    f.write(struct.pack("i", 0))
    f.write(struct.pack("i", 0))

    # 2. vocab
    tokenizer_path = tokenizer.vocab_file
    vocab = load_vocab(Path(tokenizer_path))
    for text, score in vocab.all_tokens():
        f.write(struct.pack("i", len(text)))
        f.write(text)
        f.write(struct.pack("f", score))

    # 3. write tensors
    list_vars = model
    convert_fp32_tensor("model.embed_tokens.weight", "tok_embeddings.weight", list_vars, f)
    convert_fp32_tensor("model.norm.weight", "norm.weight", list_vars, f)
    convert_fp32_tensor("lm_head.weight", "output.weight", list_vars, f)

    for i in tqdm(range(n_layer), desc="Processing layers"):
        convert_func(f"model.layers.{i}.self_attn.q_proj",
                    f"layers.{i}.attention.wq.weight", list_vars, f, quant_config, n_head, n_head,
                    permute_func=permute_func)
        convert_func(f"model.layers.{i}.self_attn.k_proj",
                    f"layers.{i}.attention.wk.weight", list_vars, f, quant_config, n_head, n_head_kv,
                    permute_func=permute_func)
        convert_func(f"model.layers.{i}.self_attn.v_proj",
                    f"layers.{i}.attention.wv.weight", list_vars, f, quant_config, n_head)
        convert_func(f"model.layers.{i}.self_attn.o_proj",
                    f"layers.{i}.attention.wo.weight", list_vars, f, quant_config, n_head)
        convert_func(f"model.layers.{i}.mlp.gate_proj",
                    f"layers.{i}.feed_forward.w1.weight", list_vars, f, quant_config, n_head)
        convert_func(f"model.layers.{i}.mlp.down_proj",
                    f"layers.{i}.feed_forward.w2.weight", list_vars, f, quant_config, n_head)
        convert_func(f"model.layers.{i}.mlp.up_proj",
                    f"layers.{i}.feed_forward.w3.weight", list_vars, f, quant_config, n_head)

        convert_fp32_tensor(f"model.layers.{i}.input_layernorm.weight",
                        f"layers.{i}.attention_norm.weight", list_vars, f)
        convert_fp32_tensor(f"model.layers.{i}.post_attention_layernorm.weight",
                        f"layers.{i}.ffn_norm.weight", list_vars, f)


    f.close()
    print(f"Success! saved as {out_path}")

def convert_llama(model_path, out_path, quant_config):
    # convert gguf
    gguf_writer = gguf.GGUFWriter(out_path, "llama")
    print(quant_config)
    # TODO(zhenweil): refact load and convert function
    convert_func = convert_fp32_tensor
    if not quant_config.not_quant:
        if quant_config.use_ggml:
            convert_func = quantize_ggml_tensor
        else:
            convert_func = quantize_jblas_tensor

    if quant_config.use_gptq or quant_config.use_awq:
        convert_func = convert_quantized_tensor
        model, config, quantize_config = load_quantized_model(model_path)
        quant_config = quantize_config
    else:
        model, config, quantize_config = load_hf_model(model_path)
        config = config.to_dict()
        model = model.state_dict()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 1. write hparams
    n_vocab = config["vocab_size"]
    n_embd = config["hidden_size"]
    n_layer = config["num_hidden_layers"]
    n_head = config["num_attention_heads"]
    ffn_hidden_size = config["intermediate_size"]
    n_head_kv = n_head
    rope_theta = config["rope_theta"] if "rope_theta" in config else 10000
    rope_scale = 1
    if "rope_scaling" in config and config["rope_scaling"] is not None:
        rope_scale = config["rope_scaling"]["factor"] if "factor" in config["rope_scaling"] else 1
    
    # Customized
    gguf_writer.add_uint32('magic', 0x67676d66)
    gguf_writer.add_uint32('version', 1)
    gguf_writer.add_uint32('n_vocab', n_vocab)
    gguf_writer.add_uint32('n_mult', 256)
    gguf_writer.add_uint32('ftype', 0)

    # LLM
    gguf_writer.add_embedding_length(n_embd)
    gguf_writer.add_context_length(4096)
    gguf_writer.add_block_count(n_layer)
    gguf_writer.add_feed_forward_length(ffn_hidden_size)

    # Attention
    gguf_writer.add_head_count(n_head)
    gguf_writer.add_head_count_kv(n_head_kv)
    gguf_writer.add_rope_dimension_count(n_embd // n_head)
    gguf_writer.add_layer_norm_rms_eps(config["rms_norm_eps"])
    gguf_writer.add_rope_freq_base(rope_theta)
    # TODO(lzw): update this
    # gguf_writer.add_rope_freq_base(rope_scale)

    # TODO:
    # bos_token_id = 0 in https://huggingface.co/decapoda-research/llama-7b-hf/blob/main/config.json
    # but bos_token_id = 1 in llama.cpp
    # Tokenizer
    gguf_writer.add_bos_token_id(config["bos_token_id"])
    gguf_writer.add_eos_token_id(config["eos_token_id"])
    gguf_writer.add_pad_token_id(0)
    gguf_writer.add_sep_token_id(0)

    # vocab
    print("gguf: get tokenizer metadata")

    tokens: List[bytes] = []
    scores: List[float] = []
    toktypes: List[int] = []

    tokenizer_path = tokenizer.vocab_file
    vocab = load_vocab(Path(tokenizer_path))
    for text, score in vocab.all_tokens():
        tokens.append(text)
        scores.append(score)

    gguf_writer.add_tokenizer_model("llama")
    gguf_writer.add_token_list(tokens)
    gguf_writer.add_token_scores(scores)

    print("gguf: get tokenizer metadata done")

    list_vars = model
    # import pudb; pudb.set_trace()
    gguf_writer.add_tensor("token_embd.weight", list_vars["model.embed_tokens.weight"].numpy())
    gguf_writer.add_tensor("output_norm.weight", list_vars["model.norm.weight"].numpy())
    gguf_writer.add_tensor("output.weight", list_vars["lm_head.weight"].numpy())

    for i in tqdm(range(n_layer), desc="Processing layers"):
        gguf_writer.add_tensor(f"blk.{i}.attn_q.weight", list_vars[f"model.layers.{i}.self_attn.q_proj.weight"].numpy())
        gguf_writer.add_tensor(f"blk.{i}.attn_k.weight", list_vars[f"model.layers.{i}.self_attn.k_proj.weight"].numpy())
        gguf_writer.add_tensor(f"blk.{i}.attn_v.weight", list_vars[f"model.layers.{i}.self_attn.v_proj.weight"].numpy())
        gguf_writer.add_tensor(f"blk.{i}.attn_output.weight", list_vars[f"model.layers.{i}.self_attn.o_proj.weight"].numpy())

        gguf_writer.add_tensor(f"blk.{i}.ffn_gate.weight", list_vars[f"model.layers.{i}.mlp.gate_proj.weight"].numpy())
        gguf_writer.add_tensor(f"blk.{i}.ffn_down.weight", list_vars[f"model.layers.{i}.mlp.down_proj.weight"].numpy())
        
        qv = quantize_q4_0(list_vars[f"model.layers.{i}.mlp.up_proj.weight"])
        ftype_cur = GGML_QK4_0_TYPE
        # import pudb; pudb.set_trace()
        # gguf_writer.add_tensor(f"blk.{i}.ffn_up.weight", qv.numpy(), raw_shape=list_vars[f"model.layers.{i}.mlp.up_proj.weight"].shape, raw_dtype=ftype_cur)
        quantize_ggml_tensor(f"model.layers.{i}.mlp.up_proj.weight", f"blk.{i}.ffn_up.weight", list_vars,
                             gguf_writer, quant_config)

        gguf_writer.add_tensor(f"blk.{i}.attn_norm.weight", list_vars[f"model.layers.{i}.input_layernorm.weight"].numpy())
        gguf_writer.add_tensor(f"blk.{i}.ffn_norm.weight", list_vars[f"model.layers.{i}.post_attention_layernorm.weight"].numpy())

    print("gguf: write header")
    gguf_writer.write_header_to_file()
    print("gguf: write metadata")
    gguf_writer.write_kv_data_to_file()
    print("gguf: write tensors")
    gguf_writer.write_tensors_to_file()

    gguf_writer.close()
    print(f"Success! saved as {out_path}")