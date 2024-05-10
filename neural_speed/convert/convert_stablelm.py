#  Copyright (c) 2023 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# Convert Hugging Face fine-tuned gpt-neox-like models to ne format
#
# Usage:
#
#   python3 models/convert-h5-to-ne.py
#
# This script is similar to "convert-pt-to-ne.py"
#
import os
import sys
import struct
import numpy as np
from pathlib import Path
import argparse
from typing import (IO, TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, TypeVar,
                    Union)
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ref: https://github.com/openai/gpt-2/blob/master/src/encoder.py
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def stack_qk_norm(block, name, n_head, norms, n_dims, ftype, layer_name="q_layernorm"):
    datas = []
    for i in range(n_head):
        ename = f"model.layers.{block}.self_attn.{layer_name}.norms.{i}.weight"
        print(f"-----> Merging Tensor {ename} with shape {norms[ename].shape}")
        datas.append(norms[ename])
        del norms[ename]
    data = np.stack(datas, axis=0)
    merged_name = f"model.layers.{block}.self_attn.{layer_name}.weight"

    # ftype == 0 -> float32, ftype == 1 -> float16
    if ftype != 0:
        if name.endswith(".weight") and not name.endswith("_norm.weight") and n_dims == 2:
            print("  Converting to float16")
            data = data.astype(np.float16)
        else:
            print("  Converting to float32")
            data = data.astype(np.float32)
    else:
        if data.dtype != np.float32:
            print("  Converting to float32")
            data = data.astype(np.float32)

    return merged_name, data


def stablelm_convert(model, tokenizer, dir_model, fname_out, ftype, hparams):
    print("stablelm ne converting: ")
    list_vars = model.state_dict()
    n_head = hparams["num_attention_heads"]
    n_head_kv = hparams["num_key_value_heads"]
    block_count = hparams["num_hidden_layers"]
    vocab_size = hparams["vocab_size"]
    n_rot = int(hparams["partial_rotary_factor"] * hparams["hidden_size"] / hparams["num_attention_heads"])
    print("Model loaded: ", dir_model)

    ne_file = fname_out + '.bin' if not fname_out.endswith(".bin") else fname_out
    fout = open(ne_file, "wb")

    # 0x67676d6c is unversioned ne
    # 0x67676d66 is versioned ggmf (requires token scores)
    ne_file_magic = 0x67676d66
    #ne_file_version = 0x00000001 # v1

    fout.write(struct.pack("i", ne_file_magic))  # magic: ne in hex
    fout.write(struct.pack("i", 1))
    fout.write(struct.pack("i", vocab_size))
    fout.write(struct.pack("i", hparams["hidden_size"]))
    fout.write(struct.pack("i", 0))
    fout.write(struct.pack("i", n_head))
    fout.write(struct.pack("i", n_head_kv))  # multi-query attention
    fout.write(struct.pack("i", hparams["num_hidden_layers"]))
    fout.write(struct.pack("i", n_rot))
    fout.write(struct.pack("i", ftype))
    fout.write(struct.pack("i", hparams["max_position_embeddings"]))
    fout.write(struct.pack("f", 0.0))
    fout.write(struct.pack("f", 0.0))
    fout.write(struct.pack("i", 0))
    fout.write(struct.pack("i", 0))  # word_embed_proj_dim (for opt)
    fout.write(struct.pack("i", 0))  # do_layer_norm_before (for opt)

    fout.write(struct.pack("i", 0)) # multi_query_group_num
    fout.write(struct.pack("i", hparams["intermediate_size"])) # ffn_hidden_size
    fout.write(struct.pack("i", 0)) # inner_hidden_size for ChatGLM2

    fout.write(struct.pack("i", 0))  # n_experts
    fout.write(struct.pack("i", 0))  # n_expert_used
    fout.write(struct.pack("i", hparams["hidden_size"] // n_head)) # n_embd_head_k for gemma
    fout.write(struct.pack("f", hparams.get("layer_norm_eps", 1e-5)))  # rms_norm_eps or layer_norm_eps
    fout.write(struct.pack("f", hparams["rope_theta"]))  # freq_base
    fout.write(struct.pack("f", 1.0))  # freq_scale, was removed in config.json (by default=1.0)
    fout.write(struct.pack("f", 1.0))  # rope_scaling_factor, was removed in config.json (by default=1.0)

    fout.write(struct.pack("i", 0))  # original_max_position_embeddings
    fout.write(struct.pack("i", 0))  # use_yarn

    fout.write(struct.pack("i", hparams["bos_token_id"]))
    fout.write(struct.pack("i", hparams["eos_token_id"]))
    fout.write(struct.pack("i", hparams["pad_token_id"] if hparams["pad_token_id"] else 0))
    fout.write(struct.pack("i", hparams["sep_token_id"] if hparams["sep_token_id"] else 0))

    for i in range(vocab_size):
        if i < vocab_size:
            text = tokenizer.decode([i]).encode('utf-8')
            fout.write(struct.pack("i", len(text)))
            fout.write(text)
            fout.write(struct.pack("f", 0.0 - i))
        else:
            text = tokenizer.decode([vocab_size - 1]).encode('utf-8')
            fout.write(struct.pack("i", len(text)))
            fout.write(text)
            fout.write(struct.pack("f", -10000))
    
    def write_header(name, data, ftype=0):
        str = name.encode('utf-8')
        n_dims = len(data.shape)
        fout.write(struct.pack("iii", n_dims, len(str), ftype))
        for i in range(n_dims):
            fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
        print(str)
        fout.write(str)

    print(hparams)
    q_norms, k_norms = dict(), dict()
    for name, data_torch in list_vars.items():
        # Convert any unsupported data types to float32
        if data_torch.dtype not in (torch.float16, torch.float32):
            data_torch = data_torch.to(torch.float32)

        # Skip some tensors
        if name.endswith((".attention.rotary_emb.inv_freq")):
            continue
        
        data = data_torch.squeeze().numpy()
        old_dtype = data.dtype
        n_dims = len(data.shape)
        if name.find("q_layernorm.norms") != -1:
            q_norms[name] = data
            if len(q_norms) >= (block_count * n_head):
                for block in range(block_count):
                    name, data = stack_qk_norm(block, name, n_head, q_norms, n_dims, ftype, layer_name="q_layernorm")
                    print(f"Processing variable {name} with shape {data.shape}, {old_dtype} --> {data.dtype}")
                    write_header(name, data)
                    data.tofile(fout)
            continue
        if name.find("k_layernorm.norms") != -1:
            k_norms[name] = data
            if len(k_norms) >= (block_count * n_head_kv):
                for block in range(block_count):
                    name, data = stack_qk_norm(block, name, n_head_kv, k_norms, n_dims, ftype, layer_name="k_layernorm")
                    print(f"Processing variable {name} with shape {data.shape}, {old_dtype} --> {data.dtype}")
                    write_header(name, data)
                    data.tofile(fout)
            continue

        # ftype == 0 -> float32, ftype == 1 -> float16
        ftype_cur = 0
        if ftype != 0:
            if name.endswith(".weight") and not name.endswith("_norm.weight") and n_dims == 2:
                print("  Converting to float16")
                data = data.astype(np.float16)
                ftype_cur = 1
            else:
                print("  Converting to float32")
                data = data.astype(np.float32)
        else:
            if data.dtype != np.float32:
                print("  Converting to float32")
                data = data.astype(np.float32)

        # header
        write_header(name, data, ftype_cur)

        # data
        data.tofile(fout)

    fout.close()

    print("Done. Output file: " + ne_file)
    print("")

def main(args_in: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Convert a model to an NE or GGUF compatible file")
    parser.add_argument(
        "--outtype",
        choices=["f32", "f16"],
        help="output format (default: based on input)"
    )
    parser.add_argument(
        "--outfile",
        type=Path,
        help="path to write to; default: based on input"
    )
    parser.add_argument(
        "--model_hub",
        choices=["huggingface","modelscope"],
        default="huggingface",
        help="hub to load model"
    )
    parser.add_argument(
        "model",
        type=Path,
        help="directory containing model file"
    )

    args = parser.parse_args(args_in)

    dir_model = args.model.as_posix()
    fname_out = args.outfile.as_posix()

    # possible data types
    #   ftype == 0 -> float32
    #   ftype == 1 -> float16
    ftype = 0
    if args.outtype == "f16":
        ftype = 1
    if args.model_hub == "modelscope":
        from modelscope import AutoModelForCausalLM, AutoTokenizer
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(dir_model, trust_remote_code=True)
    print("Loading model: ", dir_model)
    model = AutoModelForCausalLM.from_pretrained(dir_model, trust_remote_code=True)
    hparams = model.config.to_dict()
    stablelm_convert(model, tokenizer, dir_model, fname_out, ftype, hparams)


if __name__ == '__main__':
    main()
