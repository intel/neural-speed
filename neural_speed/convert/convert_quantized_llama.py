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
from common import *


def permute_func(weights, n_head: int, n_head_kv: int):
    if n_head_kv is not None and n_head != n_head_kv:
        n_head = n_head_kv
    return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2,
                            *weights.shape[1:]).swapaxes(1, 2).reshape(weights.shape))

def convert_to_q4_bestla_tensor(src_name, dst_name, model, fout, q_config, n_head, n_head_kv=0, permute_func=None):
    # unpack weight and repack into jblas format
    import neural_speed.llama_cpp as cpp_model
    if ".weight" in src_name:
        src_name = src_name.replace(".weight", "")
    qzeros = model[f"{src_name}.qzeros"]
    zeros = qzeros_to_zeros(qzeros)
    scales = model[f"{src_name}.scales"]
    qweight = model[f"{src_name}.qweight"]

    int_weight, gptq_scales, gptq_zeros = unpack_weight(qweight, scales, qzeros, q_config)
    int_weight = int_weight.view(-1,int_weight.shape[-1])

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

    # permute_func for llama-like model
    if permute_func:
        int_weight = permute_func(int_weight.t(), n_head, n_head_kv).t().contiguous()
        gptq_scales = permute_func(gptq_scales.t(), n_head, n_head_kv).t().contiguous()
        gptq_zeros = permute_func(gptq_zeros.t(), n_head, n_head_kv).t().contiguous()

    shape = int_weight.shape
    write_header(fout, shape[::-1], dst_name, GGML_QJBLAS_TYPE)

    weight_dtype = "int8"
    if q_config['bits'] == 4:
        int_weight = (int_weight - 8) * 16
        gptq_scales = gptq_scales / 16
        gptq_zeros = (gptq_zeros - 8) * 16
        weight_dtype == "int4"

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
                                               weight_dtype=weight_dtype,
                                               group_size=q_config['group_size'],
                                               alg="sym" if q_config['sym'] else "asym",
                                               compute_dtype="int8")
    dst.flatten()[:byte_size].tofile(fout)
    print(f"converting {dst_name} quantized tensor to bestla q4 block")

def main(args_in: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Convert a model to a NE compatible file")
    parser.add_argument("--outtype", choices=["f32", "f16"], help="output format (default: based on input)")
    parser.add_argument("--outfile", type=Path, help="path to write to; default: based on input")
    parser.add_argument("model", type=Path, help="directory containing model file")
    args = parser.parse_args(args_in)

    out_path = args.outfile.as_posix()
    model_path = args.model.as_posix()

    model, config, quantize_config = load_quantized_model(model_path)
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
    rope_scale = 1
    if "rope_scaling" in config and config["rope_scaling"] is not None:
        rope_scale = config["rope_scaling"]["factor"] if "factor" in config["rope_scaling"] else 1
    n_head = n_head
    n_head_kv = config["num_key_value_heads"] if "num_key_value_heads" in config else n_head
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
    f.write(struct.pack("i", 0))  # n_experts
    f.write(struct.pack("i", 0))  # n_expert_used

    f.write(struct.pack("f", config["rms_norm_eps"]))
    f.write(struct.pack("f", config["rope_theta"] if "rope_theta" in config else 10000))
    f.write(struct.pack("f", rope_scale))

    f.write(struct.pack("f", 0.0)) # config.json "rope_scaling.factor", not enabled
    f.write(struct.pack("i", 0))   # rope_scaling.original_max_position_embeddings
    f.write(struct.pack("i", 0))   # params["rope_scaling"]["type"] =="yarn" else 0))

    # TODO, bos_token_id = 0 in https://huggingface.co/decapoda-research/llama-7b-hf/blob/main/config.json
    # but bos_token_id = 1 in llama.cpp
    f.write(struct.pack("i", 1))
    f.write(struct.pack("i", 2))

    f.write(struct.pack("i", 0))
    f.write(struct.pack("i", 0))

    # 2. vocab
    tokenizer_path = os.path.join(model_path, "tokenizer.model")
    vocab = load_vocab(Path(tokenizer_path))
    for text, score in vocab.all_tokens():
        f.write(struct.pack("i", len(text)))
        f.write(text)
        f.write(struct.pack("f", score))

    # 3. write tensors
    list_vars = model
    convert_to_fp32_tensor("model.embed_tokens.weight", "tok_embeddings.weight", list_vars, f)
    convert_to_fp32_tensor("model.norm.weight", "norm.weight", list_vars, f)
    convert_to_fp32_tensor("lm_head.weight", "output.weight", list_vars, f)

    for i in range(n_layer):
        convert_to_q4_bestla_tensor(f"model.layers.{i}.self_attn.q_proj",
                    f"layers.{i}.attention.wq.weight", list_vars, f, quantize_config, n_head, n_head,
                    permute_func=permute_func)
        convert_to_q4_bestla_tensor(f"model.layers.{i}.self_attn.k_proj",
                    f"layers.{i}.attention.wk.weight", list_vars, f, quantize_config, n_head, n_head_kv,
                    permute_func=permute_func)
        convert_to_q4_bestla_tensor(f"model.layers.{i}.self_attn.v_proj",
                    f"layers.{i}.attention.wv.weight", list_vars, f, quantize_config, n_head)
        convert_to_q4_bestla_tensor(f"model.layers.{i}.self_attn.o_proj",
                    f"layers.{i}.attention.wo.weight", list_vars, f, quantize_config, n_head)
        convert_to_q4_bestla_tensor(f"model.layers.{i}.mlp.gate_proj",
                    f"layers.{i}.feed_forward.w1.weight", list_vars, f, quantize_config, n_head)
        convert_to_q4_bestla_tensor(f"model.layers.{i}.mlp.down_proj",
                    f"layers.{i}.feed_forward.w2.weight", list_vars, f, quantize_config, n_head)
        convert_to_q4_bestla_tensor(f"model.layers.{i}.mlp.up_proj",
                    f"layers.{i}.feed_forward.w3.weight", list_vars, f, quantize_config, n_head)

        convert_to_fp32_tensor(f"model.layers.{i}.input_layernorm.weight",
                        f"layers.{i}.attention_norm.weight", list_vars, f)
        convert_to_fp32_tensor(f"model.layers.{i}.post_attention_layernorm.weight",
                        f"layers.{i}.ffn_norm.weight", list_vars, f)


    f.close()
    print(f"Success! saved as {out_path}")


if __name__ == '__main__':
    main()
