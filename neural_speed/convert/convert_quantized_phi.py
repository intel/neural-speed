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
from sentencepiece import SentencePieceProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer


def convert_phi1_5_gptq_to_bestTLA(model_path, out_path, outtype, model, hparams, quantize_config):
    list_vars = model
    for name in list_vars.keys():
        print(name)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    fout = open(out_path, "wb")

    # possible data types
    #   ftype == 0 -> float32, ftype == 1 -> float16
    ftype = 0
    if outtype == "f16":
        ftype = 1

    # 1. write hparams
    print(hparams)
    ne_file_magic = 0x67676d66
    n_rot = int(hparams["partial_rotary_factor"] * hparams["hidden_size"] / hparams["num_attention_heads"])
    # n_rot = hparams['rotary_dim']
    fout.write(struct.pack("i", ne_file_magic))  # magic: ne in hex
    fout.write(struct.pack("i", 1))
    fout.write(struct.pack("i", hparams["vocab_size"]))
    fout.write(struct.pack("i", hparams["hidden_size"]))
    fout.write(struct.pack("i", hparams["intermediate_size"]))  # dummy data
    fout.write(struct.pack("i", hparams["num_attention_heads"]))
    fout.write(struct.pack("i", hparams["num_key_value_heads"]))  # multi-query attention
    fout.write(struct.pack("i", hparams["num_hidden_layers"]))
    fout.write(struct.pack("i", n_rot))
    fout.write(struct.pack("i", ftype))
    fout.write(struct.pack("i", hparams["max_position_embeddings"]))
    fout.write(struct.pack("f", 0.0))
    fout.write(struct.pack("f", 0.0))
    fout.write(struct.pack("i", 0))
    fout.write(struct.pack("i", 0))  # word_embed_proj_dim (for opt)
    fout.write(struct.pack("i", 0))  # do_layer_norm_before (for opt)

    fout.write(struct.pack("i", 0))
    fout.write(struct.pack("i", 0))
    fout.write(struct.pack("i", 0))
    fout.write(struct.pack("i", 0))  # n_experts
    fout.write(struct.pack("i", 0))  # n_expert_used
    fout.write(struct.pack("i", 0)) # n_embd_head_k for gemma
    fout.write(struct.pack("f", hparams.get("rms_norm_eps", 1e-6)))  # rms norm eps
    fout.write(struct.pack("f", 10000.0))  # freq_base
    fout.write(struct.pack("f", 1.0))  # rope_factor
    fout.write(struct.pack("f", 0.0))  # config.json "rope_scaling.factor", not enabled
    fout.write(struct.pack("i", 0))  # rope_scaling.original_max_position_embeddings
    fout.write(struct.pack("i", 0))  # params["rope_scaling"]["type"] =="yarn" else 0))
    fout.write(struct.pack("i", tokenizer.bos_token_id if tokenizer.bos_token_id is not None else -1))
    fout.write(struct.pack("i", tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1))
    fout.write(struct.pack("i", tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1))
    fout.write(struct.pack("i", tokenizer.sep_token_id if tokenizer.sep_token_id is not None else -1))

    # 2. vocab
    for i in range(hparams["vocab_size"]):
        if i < tokenizer.vocab_size:
            text = tokenizer.decode([i]).encode('utf-8')
            fout.write(struct.pack("i", len(text)))
            fout.write(text)
            fout.write(struct.pack("f", 0.0 - i))
        else:
            text = tokenizer.decode([tokenizer.vocab_size - 1]).encode('utf-8')
            fout.write(struct.pack("i", len(text)))
            fout.write(text)
            fout.write(struct.pack("f", -10000))

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
        str = dst_name.encode('utf-8')
        fout.write(struct.pack("iii", n_dims, len(str), ftype_cur))
        for i in range(n_dims):
            fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
        fout.write(str)

        # data
        data.tofile(fout)

    #3. write tensors
    convert_qwen_to_fp32_tensor("model.embed_tokens.weight", "model.embed_tokens.weight", list_vars, fout)
    convert_qwen_to_fp32_tensor("model.final_layernorm.weight", "model.final_layernorm.weight", list_vars, fout)
    convert_qwen_to_fp32_tensor("model.final_layernorm.bias", "model.final_layernorm.bias", list_vars, fout)
    convert_qwen_to_fp32_tensor("lm_head.weight", "lm_head.weight", list_vars, fout)
    convert_qwen_to_fp32_tensor("lm_head.bias", "lm_head.bias", list_vars, fout)

    for i in range(hparams["num_hidden_layers"]):
        prefix = "model.layers." + str(i)
        renamed_prefix = "model.layers." + str(i)

        convert_qwen_to_fp32_tensor(f"{prefix}.input_layernorm.weight", f"{renamed_prefix}.input_layernorm.weight",
                                    list_vars, fout)
        convert_qwen_to_fp32_tensor(f"{prefix}.input_layernorm.bias", f"{renamed_prefix}.input_layernorm.bias",
                                    list_vars, fout)

        # qkv GEMM
        convert_to_qx_bestla_tensor(f"{prefix}.self_attn.q_proj.weight", f"{prefix}.self_attn.q_proj.weight", list_vars,
                                    fout, quantize_config)
        convert_to_qx_bestla_tensor(f"{prefix}.self_attn.k_proj.weight", f"{prefix}.self_attn.k_proj.weight", list_vars,
                                    fout, quantize_config)
        convert_to_qx_bestla_tensor(f"{prefix}.self_attn.v_proj.weight", f"{prefix}.self_attn.v_proj.weight", list_vars,
                                    fout, quantize_config)
        convert_to_qx_bestla_tensor(f"{prefix}.self_attn.dense.weight", f"{prefix}.self_attn.dense.weight", list_vars,
                                    fout, quantize_config)

        convert_qwen_to_fp32_tensor(f"{prefix}.self_attn.q_proj.bias", f"{prefix}.self_attn.q_proj.bias", list_vars,
                                    fout)
        convert_qwen_to_fp32_tensor(f"{prefix}.self_attn.k_proj.bias", f"{prefix}.self_attn.k_proj.bias", list_vars,
                                    fout)
        convert_qwen_to_fp32_tensor(f"{prefix}.self_attn.v_proj.bias", f"{prefix}.self_attn.v_proj.bias", list_vars,
                                    fout)
        convert_qwen_to_fp32_tensor(f"{prefix}.self_attn.dense.bias", f"{prefix}.self_attn.dense.bias", list_vars, fout)

        # ffn GEMM
        convert_to_qx_bestla_tensor(f"{prefix}.mlp.fc1.weight", f"{renamed_prefix}.mlp.fc1.weight", list_vars, fout,
                                    quantize_config)
        convert_to_qx_bestla_tensor(f"{prefix}.mlp.fc2.weight", f"{renamed_prefix}.mlp.fc2.weight", list_vars, fout,
                                    quantize_config)

        convert_qwen_to_fp32_tensor(f"{prefix}.mlp.fc1.bias", f"{renamed_prefix}.mlp.fc1.bias", list_vars, fout)
        convert_qwen_to_fp32_tensor(f"{prefix}.mlp.fc2.bias", f"{renamed_prefix}.mlp.fc2.bias", list_vars, fout)

    fout.close()
    print(f"Success! saved as {out_path}")


def convert_phi2_gptq_to_bestTLA(model_path, model, out_path, hparams, quantize_config):
    list_vars = model
    for name in list_vars.keys():
        print(name)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    fout = open(out_path, "wb")

    # possible data types
    #   ftype == 0 -> float32, ftype == 1 -> float16
    ftype = 0
    if outtype == "f16":
        ftype = 1

    # 1. write hparams
    print(hparams)
    ne_file_magic = 0x67676d66
    #n_rot = int(hparams["partial_rotary_factor"]*hparams["hidden_size"]/hparams["num_attention_heads"])
    n_rot = hparams['rotary_dim']
    fout.write(struct.pack("i", ne_file_magic))  # magic: ne in hex
    fout.write(struct.pack("i", 1))
    fout.write(struct.pack("i", hparams["vocab_size"]))
    fout.write(struct.pack("i", hparams["n_embd"]))
    fout.write(struct.pack("i", hparams["n_embd"] * 4))  # dummy data
    fout.write(struct.pack("i", hparams["n_head"]))
    fout.write(struct.pack("i", hparams["n_head"]))  # multi-query attention
    fout.write(struct.pack("i", hparams["n_layer"]))
    fout.write(struct.pack("i", n_rot))
    fout.write(struct.pack("i", ftype))
    fout.write(struct.pack("i", hparams["n_positions"]))
    fout.write(struct.pack("f", 0.0))
    fout.write(struct.pack("f", 0.0))
    fout.write(struct.pack("i", 0))
    fout.write(struct.pack("i", 0))  # word_embed_proj_dim (for opt)
    fout.write(struct.pack("i", 0))  # do_layer_norm_before (for opt)

    fout.write(struct.pack("i", 0))
    fout.write(struct.pack("i", 0))
    fout.write(struct.pack("i", 0))
    fout.write(struct.pack("i", 0))  # n_experts
    fout.write(struct.pack("i", 0))  # n_expert_used
    fout.write(struct.pack("f", hparams.get("rms_norm_eps", 1e-6)))  # rms norm eps
    fout.write(struct.pack("f", 10000.0))  # freq_base
    fout.write(struct.pack("f", 1.0))  # rope_factor
    fout.write(struct.pack("f", 0.0))  # config.json "rope_scaling.factor", not enabled
    fout.write(struct.pack("i", 0))  # rope_scaling.original_max_position_embeddings
    fout.write(struct.pack("i", 0))  # params["rope_scaling"]["type"] =="yarn" else 0))
    fout.write(struct.pack("i", tokenizer.bos_token_id if tokenizer.bos_token_id is not None else -1))
    fout.write(struct.pack("i", tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1))
    fout.write(struct.pack("i", tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1))
    fout.write(struct.pack("i", tokenizer.sep_token_id if tokenizer.sep_token_id is not None else -1))

    # 2. vocab
    for i in range(hparams["vocab_size"]):
        if i < tokenizer.vocab_size:
            text = tokenizer.decode([i]).encode('utf-8')
            fout.write(struct.pack("i", len(text)))
            fout.write(text)
            fout.write(struct.pack("f", 0.0 - i))
        else:
            text = tokenizer.decode([tokenizer.vocab_size - 1]).encode('utf-8')
            fout.write(struct.pack("i", len(text)))
            fout.write(text)
            fout.write(struct.pack("f", -10000))

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
        str = dst_name.encode('utf-8')
        fout.write(struct.pack("iii", n_dims, len(str), ftype_cur))
        for i in range(n_dims):
            fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
        fout.write(str)

        # data
        data.tofile(fout)

    #3. write tensors
    convert_qwen_to_fp32_tensor("transformer.embd.wte.weight", "model.embed_tokens.weight", list_vars, fout)
    convert_qwen_to_fp32_tensor("lm_head.ln.weight", "model.final_layernorm.weight", list_vars, fout)
    convert_qwen_to_fp32_tensor("lm_head.ln.bias", "model.final_layernorm.bias", list_vars, fout)
    convert_qwen_to_fp32_tensor("lm_head.linear.weight", "lm_head.weight", list_vars, fout)
    convert_qwen_to_fp32_tensor("lm_head.linear.bias", "lm_head.bias", list_vars, fout)

    for i in range(hparams["n_layer"]):
        prefix = "transformer.h." + str(i)
        renamed_prefix = "model.layers." + str(i)

        convert_qwen_to_fp32_tensor(f"{prefix}.ln.weight", f"{renamed_prefix}.input_layernorm.weight", list_vars, fout)
        convert_qwen_to_fp32_tensor(f"{prefix}.ln.bias", f"{renamed_prefix}.input_layernorm.bias", list_vars, fout)

        # qkv GEMM
        convert_to_qx_bestla_tensor(f"{prefix}.mixer.Wqkv.weight", f"{renamed_prefix}.mixer.Wqkv.weight", list_vars,
                                    fout, quantize_config)
        convert_qwen_to_fp32_tensor(f"{prefix}.mixer.Wqkv.bias", f"{renamed_prefix}.mixer.Wqkv.bias", list_vars, fout)

        convert_to_qx_bestla_tensor(f"{prefix}.mixer.out_proj.weight", f"{renamed_prefix}.mixer.out_proj.weight",
                                    list_vars, fout, quantize_config)
        convert_qwen_to_fp32_tensor(f"{prefix}.mixer.out_proj.bias", f"{renamed_prefix}.mixer.out_proj.bias", list_vars,
                                    fout)

        # ffn GEMM
        convert_to_qx_bestla_tensor(f"{prefix}.mlp.fc1.weight", f"{renamed_prefix}.mlp.fc1.weight", list_vars, fout,
                                    quantize_config)
        convert_to_qx_bestla_tensor(f"{prefix}.mlp.fc2.weight", f"{renamed_prefix}.mlp.fc2.weight", list_vars, fout,
                                    quantize_config)

        convert_qwen_to_fp32_tensor(f"{prefix}.mlp.fc1.bias", f"{renamed_prefix}.mlp.fc1.bias", list_vars, fout)
        convert_qwen_to_fp32_tensor(f"{prefix}.mlp.fc2.bias", f"{renamed_prefix}.mlp.fc2.bias", list_vars, fout)

    fout.close()
    print(f"Success! saved as {out_path}")


def main(args_in: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Convert a model to a NE compatible file")
    parser.add_argument("--outtype", choices=["f32", "f16"], help="output format (default: based on input)")
    parser.add_argument("--outfile", type=Path, help="path to write to; default: based on input")
    parser.add_argument("--model_hub",
                        choices=["huggingface", "modelscope"],
                        default="huggingface",
                        help="hub to load model")
    parser.add_argument("--format",
                        type=str,
                        default="NE",
                        choices=["NE", "GGUF"],
                        help="convert to the GGUF or NE format")
    parser.add_argument("model", type=Path, help="directory containing model file")
    args = parser.parse_args(args_in)

    out_path = args.outfile.as_posix()
    model_path = args.model.as_posix()

    model, hparams, quantize_config = load_quantized_safetensors(model_path)

    if hparams['model_type'] == "phi":
        convert_phi1_5_gptq_to_bestTLA(model_path, out_path, args.outtype, model, hparams, quantize_config)
    elif hparams['model_type'] == "phi-msft":
        convert_phi2_gptq_to_bestTLA(model_path, out_path, args.outtype, model, hparams, quantize_config)


if __name__ == '__main__':
    main()
