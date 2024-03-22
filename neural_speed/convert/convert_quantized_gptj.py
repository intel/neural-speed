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
from tqdm import tqdm
from transformers import AutoTokenizer


def convert_to_qx_bestla_tensor(src_name, dst_name, model, fout, q_config):
    # unpack weight and repack into 3bits / 4bits BestLA format
    import neural_speed.llama_cpp as cpp_model
    if ".weight" in src_name:
        src_name = src_name.replace(".weight", "")
    qzeros = model[f"{src_name}.qzeros"]
    zeros = qzeros_to_zeros(qzeros)
    scales = model[f"{src_name}.scales"]
    qweight = model[f"{src_name}.qweight"]

    int_weight, gptq_scales, gptq_zeros = unpack_weight(
        qweight, scales, qzeros, q_config)
    int_weight = int_weight.view(-1, int_weight.shape[-1])

    # shuffle weight in GPTQ when act order is on
    if 'desc_act' in q_config and q_config['desc_act']:
        g_idx = model[f"{src_name}.g_idx"]
        int_weight2 = int_weight.clone()
        group_size = q_config['group_size']
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
    if 'desc_act' in q_config and q_config['desc_act']:
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
    print(f"converting {dst_name} quantized tensor to bestla q{q_config['bits']} block")


def main(args_in: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Convert a model to a NE compatible file")
    parser.add_argument("--outtype", choices=["f32", "f16"], help="output format (default: based on input)")
    parser.add_argument("--outfile", type=Path, help="path to write to; default: based on input")
    parser.add_argument("--model_hub", choices=["huggingface","modelscope"],
                        default="huggingface", help="hub to load model")
    parser.add_argument("model", type=Path, help="directory containing model file")
    args = parser.parse_args(args_in)

    out_path = args.outfile.as_posix()
    model_path = args.model.as_posix()

    model, config, quantize_config = load_quantized_model(model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True)
    fout = open(out_path, "wb")

    # 1. write hparams
    hparams = config
    n_layer = hparams["n_layer"]
    fout.write(b"ggjt"[::-1])  # 0x67676d6c)) # magic: ggml in hex
    values = [
        1,  # file version
        hparams["vocab_size"],
        hparams["n_embd"],
        hparams["n_embd"] // hparams["n_head"],
        hparams["n_head"],
        hparams.get("n_head_kv", 0),  # multi-query attention
        hparams["n_layer"],
        hparams["rotary_dim"],
        0
    ]
    fout.write(struct.pack("i" * len(values), *values))
    fout.write(struct.pack("i", 0))
    fout.write(struct.pack("f", 0))
    fout.write(struct.pack("f", 0))
    fout.write(struct.pack("i", 0))
    fout.write(struct.pack("i", 0))  # word_embed_proj_dim (for opt)
    fout.write(struct.pack("i", 0))  # do_layer_norm_before (for opt)

    fout.write(struct.pack("i", 0))
    fout.write(struct.pack("i", 0))
    fout.write(struct.pack("i", 0))
    fout.write(struct.pack("i", 0))  # n_experts
    fout.write(struct.pack("i", 0))  # n_expert_used
    fout.write(struct.pack("i", 0)) # n_embd_head_k for gemma
    fout.write(struct.pack("f", hparams.get("layer_norm_epsilon", 1e-5)))  # rms_norm_eps or layer_norm_eps
    fout.write(struct.pack("f", 10000.0))  # freq_base
    fout.write(struct.pack("f", 1.0))  # rope_factor
    fout.write(struct.pack("f", 0.0)) # config.json "rope_scaling.factor", not enabled
    fout.write(struct.pack("i", 0))   # rope_scaling.original_max_position_embeddings
    fout.write(struct.pack("i", 0))   # params["rope_scaling"]["type"] =="yarn" else 0))

    fout.write(struct.pack("i", tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1))
    fout.write(struct.pack("i", tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2))
    fout.write(struct.pack("i", tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1))
    fout.write(struct.pack("i", tokenizer.sep_token_id if tokenizer.sep_token_id is not None else -1))

    # 2. vocab
    byte_encoder = bytes_to_unicode()
    byte_decoder = {v: k for k, v in byte_encoder.items()}

    encoder = tokenizer.vocab
    # Add added_tokens (special tokens) to the encoder
    encoder_added = tokenizer.get_added_vocab()

    for i, key in enumerate(sorted(encoder, key=encoder.get)):
        # for key in encoder:
        text = bytearray([byte_decoder[c] for c in key])
        fout.write(struct.pack("i", len(text)))
        fout.write(text)
        if key not in encoder_added:
            fout.write(struct.pack("f", 0.0 - i))
        else:
            fout.write(struct.pack("f", -10000))

    # 3. write tensors
    list_vars = model

    convert_to_fp32_tensor("transformer.wte.weight", "transformer.wte.weight", list_vars, fout)
    convert_to_fp32_tensor("transformer.ln_f.weight", "transformer.ln_f.weight", list_vars, fout)
    convert_to_fp32_tensor("transformer.ln_f.bias", "transformer.ln_f.bias", list_vars, fout)
    convert_to_fp32_tensor("lm_head.bias", "lm_head.bias", list_vars, fout)
    convert_to_fp32_tensor("lm_head.weight", "lm_head.weight", list_vars, fout)

    for i in tqdm(range(n_layer), desc="Processing layers"):
        convert_to_qx_bestla_tensor(f"transformer.h.{i}.attn.q_proj.weight",
                    f"transformer.h.{i}.attn.q_proj.weight", list_vars, fout, quantize_config)
        convert_to_qx_bestla_tensor(f"transformer.h.{i}.attn.k_proj.weight",
                    f"transformer.h.{i}.attn.k_proj.weight", list_vars, fout, quantize_config)
        convert_to_qx_bestla_tensor(f"transformer.h.{i}.attn.v_proj.weight",
                    f"transformer.h.{i}.attn.v_proj.weight", list_vars, fout, quantize_config)

        convert_to_qx_bestla_tensor(f"transformer.h.{i}.attn.out_proj.weight",
                    f"transformer.h.{i}.attn.out_proj.weight", list_vars, fout, quantize_config)
        convert_to_qx_bestla_tensor(f"transformer.h.{i}.mlp.fc_in.weight",
                    f"transformer.h.{i}.mlp.fc_in.weight", list_vars, fout, quantize_config)
        convert_to_qx_bestla_tensor(f"transformer.h.{i}.mlp.fc_out.weight",
                    f"transformer.h.{i}.mlp.fc_out.weight", list_vars, fout, quantize_config)

        convert_to_fp32_tensor(f"transformer.h.{i}.mlp.fc_in.bias",
                        f"transformer.h.{i}.mlp.fc_in.bias", list_vars, fout)
        convert_to_fp32_tensor(f"transformer.h.{i}.mlp.fc_out.bias",
                        f"transformer.h.{i}.mlp.fc_out.bias", list_vars, fout)

        convert_to_fp32_tensor(f"transformer.h.{i}.ln_1.weight",
                        f"transformer.h.{i}.ln_1.weight", list_vars, fout)
        convert_to_fp32_tensor(f"transformer.h.{i}.ln_1.bias",
                        f"transformer.h.{i}.ln_1.bias", list_vars, fout)


    fout.close()
    print(f"Success! saved as {out_path}")


if __name__ == '__main__':
    main()
