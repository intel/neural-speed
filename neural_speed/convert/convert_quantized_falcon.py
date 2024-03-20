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
    fout = open(out_path, "wb")

    # possible data types
    #   ftype == 0 -> float32, ftype == 1 -> float16
    ftype = 0
    if args.outtype == "f16":
        ftype = 1

    # 1. write hparams
    n_head_kv = hparams.get("n_head_kv", 1)
    n_head = hparams["n_head"]
    head_dim = hparams["hidden_size"] // n_head

    fout.write(struct.pack("i", 0x67676d6c))  # magic: falcon in hex

    fout.write(struct.pack("i", hparams["vocab_size"]))
    fout.write(struct.pack("i", hparams["hidden_size"]))
    fout.write(struct.pack("i", 0))
    fout.write(struct.pack("i", n_head))
    fout.write(struct.pack("i", n_head_kv))  # multi-query attention
    fout.write(struct.pack("i", hparams["n_layer"]))
    fout.write(struct.pack("i", 0))
    fout.write(struct.pack("i", ftype))
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
    fout.write(struct.pack("f", hparams.get("rms_norm_eps", 1e-6)))  # rms norm eps
    fout.write(struct.pack("f", 10000.0))  # freq_base
    fout.write(struct.pack("f", 1.0))  # rope_factor

    fout.write(struct.pack("f", 0.0))  # config.json "rope_scaling.factor", not enabled
    fout.write(struct.pack("i", 0))  # rope_scaling.original_max_position_embeddings
    fout.write(struct.pack("i", 0))  # params["rope_scaling"]["type"] =="yarn" else 0))

    fout.write(struct.pack("i", tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1))
    fout.write(struct.pack("i", tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2))
    fout.write(struct.pack("i", tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1))
    fout.write(struct.pack("i", tokenizer.sep_token_id if tokenizer.sep_token_id is not None else -1))

    # 2. vocab
    reverse_vocab = {id: encoded_tok for encoded_tok, id in tokenizer.vocab.items()}
    byte_encoder = bytes_to_unicode()
    byte_decoder = {v: k for k, v in byte_encoder.items()}

    for i in range(hparams["vocab_size"]):
        text = bytearray([byte_decoder[c] for c in reverse_vocab[i]])
        fout.write(struct.pack("i", len(text)))
        fout.write(text)

    def convert_to_fp32_tensor(src_name, dst_name, model, fout):
        # qwen-gptq is torch.bfloat16 mostly.
        if model[src_name].dtype == torch.float32:
            data = model[src_name].squeeze().numpy()
        else:
            data = model[src_name].squeeze().to(torch.float32).numpy()
        data = data.astype(np.float32)
        shape = data.shape
        n_dims = len(shape)
        print("convert_to_fp32_tensor:  %45s" % src_name + "-> %-40s" % dst_name + " shape: ", shape, " type: ",
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
        data.tofile(fout)

    #3. write tensors
    convert_to_fp32_tensor("transformer.word_embeddings.weight", "transformer.word_embeddings.weight", list_vars, fout)
    convert_to_fp32_tensor("transformer.ln_f.weight", "transformer.ln_f.weight", list_vars, fout)
    convert_to_fp32_tensor("transformer.ln_f.bias", "transformer.ln_f.bias", list_vars, fout)
    convert_to_fp32_tensor("lm_head.weight", "lm_head.weight", list_vars, fout)

    for i in range(hparams["n_layer"]):
        prefix = "transformer.h." + str(i)

        if n_head_kv == 1:
            convert_to_fp32_tensor(f"{prefix}.input_layernorm.weight", f"{prefix}.input_layernorm.weight", list_vars,
                                   fout)
            convert_to_fp32_tensor(f"{prefix}.input_layernorm.bias", f"{prefix}.input_layernorm.bias", list_vars, fout)
        elif n_head_kv == 8:
            convert_to_fp32_tensor(f"{prefix}.ln_mlp.weight", f"{prefix}.ln_mlp.weight", list_vars, fout)
            convert_to_fp32_tensor(f"{prefix}.ln_mlp.bias", f"{prefix}.ln_mlp.bias", list_vars, fout)
            convert_to_fp32_tensor(f"{prefix}.ln_attn.weight", f"{prefix}.ln_attn.weight", list_vars, fout)
            convert_to_fp32_tensor(f"{prefix}.ln_attn.bias", f"{prefix}.ln_attn.bias", list_vars, fout)

        # qkv GEMM
        convert_to_qx_bestla_tensor(f"{prefix}.self_attention.query_key_value.weight",
                                    f"{prefix}.self_attention.query_key_value.weight", list_vars, fout, quantize_config)
        convert_to_qx_bestla_tensor(f"{prefix}.self_attention.dense.weight", f"{prefix}.self_attention.dense.weight",
                                    list_vars, fout, quantize_config)

        # ffn GEMM
        convert_to_qx_bestla_tensor(f"{prefix}.mlp.dense_h_to_4h", f"{prefix}.mlp.dense_h_to_4h.weight", list_vars,
                                    fout, quantize_config)
        convert_to_qx_bestla_tensor(f"{prefix}.mlp.dense_4h_to_h", f"{prefix}.mlp.dense_4h_to_h.weight", list_vars,
                                    fout, quantize_config)

    fout.close()
    print(f"Success! saved as {out_path}")


if __name__ == '__main__':
    main()
