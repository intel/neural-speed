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
import json
import sys
import re
import argparse
from common import *

def permute_func(weights, n_head: int, n_head_kv: int):
    if n_head_kv is not None and n_head != n_head_kv:
        n_head //= n_head_kv
    return (weights.reshape(n_head_kv, 2, weights.shape[0] // n_head_kv // 2, *weights.shape[1:])
                .swapaxes(1, 2)
                .reshape(weights.shape))


def convert_to_qx_bestla_tensor(src_name, dst_name, model, fout, q_config):
    # unpack weight and repack into 3bits / 4bits BestLA format
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

    shape = int_weight.shape
    write_header(fout, shape[::-1], dst_name, GGML_QJBLAS_TYPE)

    # INC stores sig-int4 value as u4(range 0~15, they add a offset),
    # BesTLA requires s4_clip((-8,7)*16), so we sub the offset and then mul 16.
    # Int3 is the same as int4, but offset=4, mul scale==32.
    weight_dtype = "int8"
    if q_config['bits'] == 4:
        int_weight = (int_weight - 8) * 16
        gptq_scales = gptq_scales / 16
        gptq_zeros = (gptq_zeros - 8) * 16
        weight_dtype = "int4"
    elif q_config['bits'] == 3:
        int_weight = (int_weight - 4) * 32
        gptq_scales = gptq_scales / 32
        gptq_zeros = (gptq_zeros - 4) * 32
        weight_dtype = "int3"

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

    # repack int weight in BesTLA format
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

    from transformers import AutoModelForCausalLM, AutoTokenizer
    def load_GPTQ_qwen(model_path):
        from safetensors.torch import load_file
        model_1 = load_file("/home/zhenzhong/model/Qwen-7B-Chat-GPTQ/model-00001-of-00002.safetensors")
        model = load_file("/home/zhenzhong/model/Qwen-7B-Chat-GPTQ/model-00002-of-00002.safetensors")
        model.update(model_1)

        with open(model_path + '/config.json', "r", encoding="utf-8") as f:
            config = json.load(f)

        quantize_config = config["quantization_config"]
        if "zero_point" in quantize_config:
            quantize_config["sym"] = not quantize_config["zero_point"]
        return model, config, config["quantization_config"]
    
    model, hparams, quantize_config = load_GPTQ_qwen(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    f = open(out_path, "wb")
    
    # possible data types
    #   ftype == 0 -> float32
    #   ftype == 1 -> float16
    ftype = 0
    if args.outtype == "f16":
        ftype = 1
        
    # 1. write hparams
    ne_file_magic = 0x67676d66
    f.write(struct.pack("i", ne_file_magic))  # magic: ne in hex
    f.write(struct.pack("i", 1))
    f.write(struct.pack("i", hparams["vocab_size"]))
    f.write(struct.pack("i", hparams["hidden_size"]))
    f.write(struct.pack("i", hparams["intermediate_size"]))  # dummy data
    f.write(struct.pack("i", hparams["num_attention_heads"]))
    f.write(struct.pack("i", 0))  # multi-query attention
    f.write(struct.pack("i", hparams["num_hidden_layers"]))
    f.write(struct.pack("i", hparams["kv_channels"]))
    f.write(struct.pack("i", ftype))
    f.write(struct.pack("i", hparams["seq_length"]))
    f.write(struct.pack("f", 0.0))
    f.write(struct.pack("f", 0.0))
    f.write(struct.pack("i", 0))
    f.write(struct.pack("i", 0))  # word_embed_proj_dim (for opt)
    f.write(struct.pack("i", 0))  # do_layer_norm_before (for opt)

    f.write(struct.pack("i", 0))
    f.write(struct.pack("i", hparams["intermediate_size"]))
    f.write(struct.pack("i", 0))
    f.write(struct.pack("f", hparams.get("rms_norm_eps", 1e-6)))  # rms norm eps
    f.write(struct.pack("f", 10000.0))  # freq_base
    f.write(struct.pack("f", 1.0))  # rope_factor

    f.write(struct.pack("i", tokenizer.special_tokens['<|endoftext|>']))
    f.write(struct.pack("i", tokenizer.special_tokens['<|endoftext|>']))
    f.write(struct.pack("i", tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1))
    f.write(struct.pack("i", tokenizer.sep_token_id if tokenizer.sep_token_id is not None else -1))

    # 2. vocab
    for i in range(hparams["vocab_size"]):
        if i < tokenizer.vocab_size:
            text = tokenizer.decode([i]).encode('utf-8')
            f.write(struct.pack("i", len(text)))
            f.write(text)
            f.write(struct.pack("f", 0.0 - i))
        else:
            text = tokenizer.decode([tokenizer.vocab_size - 1]).encode('utf-8')
            f.write(struct.pack("i", len(text)))
            f.write(text)
            f.write(struct.pack("f", -10000))
    
    
    # 3. write tensors
    list_vars = model

    convert_to_fp32_tensor("transformer.wte.weight", "tok_embeddings.weight", list_vars, f)
    convert_to_fp32_tensor("transformer.ln_f.weight", "output_norm.weight", list_vars, f)
    convert_to_fp32_tensor("lm_head.weight", "output.weight", list_vars, f)

    for i in range(hparams["num_hidden_layers"]):
        convert_to_fp32_tensor(f"transformer.h.{i}.ln_1.weight",
                    f"blk.h.{i}.attn_norm.weight", list_vars, f)
        convert_to_fp32_tensor(f"transformer.h.{i}.ln_2.weight",
                    f"blk.h.{i}.ffn_norm.weight", list_vars, f)
        
        # qkv GEMM
        convert_to_qx_bestla_tensor(f"transformer.h.{i}.attn.c_attn.weight",
                    f"blk.h.{i}.attn_qkv.weight", list_vars, f, quantize_config)
        convert_to_fp32_tensor(f"transformer.h.{i}.attn.c_attn.bias",
                        f"blk.h.{i}.attn_qkv.bias", list_vars, f)
        convert_to_qx_bestla_tensor(f"transformer.h.{i}.attn.c_proj.weight",
                    f"blk.h.{i}.attn_output.weight", list_vars, f, quantize_config)
        
        
        # ffn GEMM
        convert_to_qx_bestla_tensor(f"transformer.h.{i}.mlp.w1.weight",
                    f"blk.h.{i}.ffn_up.weight", list_vars, f, quantize_config)
        convert_to_qx_bestla_tensor(f"transformer.h.{i}.mlp.w2.weight",
                    f"blk.h.{i}.ffn_gate.weight", list_vars, f, quantize_config)
        convert_to_qx_bestla_tensor(f"transformer.h.{i}.mlp.c_proj.weight",
                    f"blk.h.{i}.ffn_down.weight", list_vars, f, quantize_config)


    f.close()
    print(f"Success! saved as {out_path}")

if __name__ == '__main__':
    main()
