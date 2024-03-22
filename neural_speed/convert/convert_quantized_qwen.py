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
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(args_in: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Convert a model to a NE compatible file")
    parser.add_argument("--outtype", choices=["f32", "f16"], help="output format (default: based on input)")
    parser.add_argument("--outfile", type=Path, help="path to write to; default: based on input")
    parser.add_argument("--model_hub",
                        choices=["huggingface", "modelscope"],
                        default="huggingface",
                        help="hub to load model")
    parser.add_argument("model", type=Path, help="directory containing model file")
    args = parser.parse_args(args_in)

    out_path = args.outfile.as_posix()
    model_path = args.model.as_posix()

    model, hparams, quantize_config = load_quantized_safetensors(model_path)
    list_vars = model

    print(hparams)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    f = open(out_path, "wb")

    # possible data types
    #   ftype == 0 -> float32, ftype == 1 -> float16
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
    f.write(
        struct.pack(
            "i", hparams["kv_channels"] if "kv_channels" in hparams else int(hparams["hidden_size"] /
                                                                             hparams["num_attention_heads"])))
    f.write(struct.pack("i", ftype))
    f.write(struct.pack("i", hparams["max_position_embeddings"]))
    f.write(struct.pack("f", 0.0))
    f.write(struct.pack("f", 0.0))
    f.write(struct.pack("i", 0))
    f.write(struct.pack("i", 0))  # word_embed_proj_dim (for opt)
    f.write(struct.pack("i", 0))  # do_layer_norm_before (for opt)

    f.write(struct.pack("i", 0))
    if hparams['model_type']=='qwen2':
        f.write(struct.pack("i", hparams["intermediate_size"]))
    else:
        f.write(struct.pack("i", int(hparams["intermediate_size"]/2)))
    f.write(struct.pack("i", 0))
    f.write(struct.pack("i", 0))  # n_experts
    f.write(struct.pack("i", 0))  # n_expert_used
    f.write(struct.pack("i", 0)) # n_embd_head_k for gemma
    f.write(struct.pack("f", hparams.get("rms_norm_eps", 1e-6)))  # rms norm eps
    f.write(struct.pack("f", 10000.0))  # freq_base
    f.write(struct.pack("f", 1.0))  # rope_factor

    f.write(struct.pack("f", 0.0))  # config.json "rope_scaling.factor", not enabled
    f.write(struct.pack("i", 0))  # rope_scaling.original_max_position_embeddings
    f.write(struct.pack("i", 0))  # params["rope_scaling"]["type"] =="yarn" else 0))

    if hparams['model_type'] == 'qwen2':
        f.write(struct.pack("i", hparams["bos_token_id"]))
        f.write(struct.pack("i", hparams["eos_token_id"]))
    else:
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

    def convert_qwen_to_fp32_tensor(src_name, dst_name, model, fout):
        # qwen-gptq is torch.bfloat16 mostly.
        if model[src_name].dtype == torch.float32:
            data = model[src_name].squeeze().numpy()
        else:
            data = model[src_name].squeeze().to(torch.float32).numpy()
        data = data.astype(np.float32)
        shape = data.shape
        n_dims = len(shape)
        print("convert_qwen_to_fp32_tensor:  %40s" % src_name + "-> %-40s" % dst_name + " shape: ", shape, " type: ",
              data.dtype)

        #ftype_cur = {torch.float16: 1, torch.float32: 0}[data.dtype]
        # default type is fp32
        ftype_cur = 0
        if ftype == 1 and n_dims > 1:
            data = data.astype(np.float16)
            ftype_cur = 1
        else:
            data = data.astype(np.float32)

        # header
        # write_header(fout, shape, dst_name, ftype_cur)
        str = src_name.encode('utf-8')
        fout.write(struct.pack("iii", n_dims, len(str), ftype_cur))
        for i in range(n_dims):
            fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
        fout.write(str)

        # data
        data.tofile(f)

    #3. write tensors
    if hparams['model_type'] == 'qwen':
        convert_qwen_to_fp32_tensor("transformer.wte.weight", "transformer.wte.weight", list_vars, f)
        convert_qwen_to_fp32_tensor("transformer.ln_f.weight", "transformer.ln_f.weight", list_vars, f)
        convert_qwen_to_fp32_tensor("lm_head.weight", "lm_head.weight", list_vars, f)

        for i in range(hparams["num_hidden_layers"]):
            convert_qwen_to_fp32_tensor(f"transformer.h.{i}.ln_1.weight", f"transformer.h.{i}.ln_1.weight", list_vars,
                                        f)
            convert_qwen_to_fp32_tensor(f"transformer.h.{i}.ln_2.weight", f"transformer.h.{i}.ln_2.weight", list_vars,
                                        f)

            # qkv GEMM
            convert_to_qx_bestla_tensor(f"transformer.h.{i}.attn.c_attn.weight",
                                        f"transformer.h.{i}.attn.c_attn.weight", list_vars, f, quantize_config)
            convert_qwen_to_fp32_tensor(f"transformer.h.{i}.attn.c_attn.bias", f"transformer.h.{i}.attn.c_attn.bias",
                                        list_vars, f)
            convert_to_qx_bestla_tensor(f"transformer.h.{i}.attn.c_proj.weight",
                                        f"transformer.h.{i}.attn.c_proj.weight", list_vars, f, quantize_config)

            # ffn GEMM
            convert_to_qx_bestla_tensor(f"transformer.h.{i}.mlp.w1.weight", f"transformer.h.{i}.mlp.w1.weight",
                                        list_vars, f, quantize_config)
            convert_to_qx_bestla_tensor(f"transformer.h.{i}.mlp.w2.weight", f"transformer.h.{i}.mlp.w2.weight",
                                        list_vars, f, quantize_config)
            convert_to_qx_bestla_tensor(f"transformer.h.{i}.mlp.c_proj.weight", f"transformer.h.{i}.mlp.c_proj.weight",
                                        list_vars, f, quantize_config)

        f.close()
        print(f"Success! saved as {out_path}")
    elif hparams['model_type'] == 'qwen2':
        # 3. write tensors
        convert_qwen_to_fp32_tensor("model.embed_tokens.weight", "model.embed_tokens.weight", list_vars, f)
        convert_qwen_to_fp32_tensor("model.norm.weight", "model.norm.weight", list_vars, f)
        convert_qwen_to_fp32_tensor("lm_head.weight", "lm_head.weight", list_vars, f)

        for i in range(hparams["num_hidden_layers"]):
            convert_qwen_to_fp32_tensor(f"model.layers.{i}.input_layernorm.weight",
                                        f"model.layers.{i}.input_layernorm.weight", list_vars, f)
            convert_qwen_to_fp32_tensor(f"model.layers.{i}.post_attention_layernorm.weight",
                                        f"model.layers.{i}.post_attention_layernorm.weight", list_vars, f)

            # qkv GEMM
            convert_to_qx_bestla_tensor(f"model.layers.{i}.self_attn.q_proj.weight",
                                        f"model.layers.{i}.self_attn.q_proj.weight", list_vars, f, quantize_config)
            convert_to_qx_bestla_tensor(f"model.layers.{i}.self_attn.k_proj.weight",
                                        f"model.layers.{i}.self_attn.k_proj.weight", list_vars, f, quantize_config)
            convert_to_qx_bestla_tensor(f"model.layers.{i}.self_attn.v_proj.weight",
                                        f"model.layers.{i}.self_attn.v_proj.weight", list_vars, f, quantize_config)
            convert_to_qx_bestla_tensor(f"model.layers.{i}.self_attn.o_proj.weight",
                                        f"model.layers.{i}.self_attn.o_proj.weight", list_vars, f, quantize_config)

            convert_qwen_to_fp32_tensor(f"model.layers.{i}.self_attn.q_proj.bias",
                                        f"model.layers.{i}.self_attn.q_proj.bias", list_vars, f)
            convert_qwen_to_fp32_tensor(f"model.layers.{i}.self_attn.k_proj.bias",
                                        f"model.layers.{i}.self_attn.k_proj.bias", list_vars, f)
            convert_qwen_to_fp32_tensor(f"model.layers.{i}.self_attn.v_proj.bias",
                                        f"model.layers.{i}.self_attn.v_proj.bias", list_vars, f)

            # ffn GEMM
            convert_to_qx_bestla_tensor(f"model.layers.{i}.mlp.down_proj.weight",
                                        f"model.layers.{i}.mlp.down_proj.weight", list_vars, f, quantize_config)
            convert_to_qx_bestla_tensor(f"model.layers.{i}.mlp.gate_proj.weight",
                                        f"model.layers.{i}.mlp.gate_proj.weight", list_vars, f, quantize_config)
            convert_to_qx_bestla_tensor(f"model.layers.{i}.mlp.up_proj.weight", f"model.layers.{i}.mlp.up_proj.weight",
                                        list_vars, f, quantize_config)

        f.close()
        print(f"Success! saved as {out_path}")


if __name__ == '__main__':
    main()
