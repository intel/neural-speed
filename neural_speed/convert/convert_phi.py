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

import struct
import numpy as np
from pathlib import Path
import argparse
from typing import (IO, TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, TypeVar,
                    Union)
from transformers import AutoModelForCausalLM, AutoTokenizer
import gguf

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

def phi_convert_gguf(model, tokenizer, dir_model, fname_out, ftype, hparams):
    print("phi.gguf converting: ")
    list_vars = model.state_dict()
    n_rot = int(hparams["partial_rotary_factor"]*hparams["hidden_size"]/hparams["num_attention_heads"])
    for name in list_vars.keys():
        print(name, list_vars[name].shape, list_vars[name].dtype)

    print(hparams)

    gguf_file = fname_out + '.gguf'
    gguf_writer = gguf.GGUFWriter(gguf_file, "phi")

    gguf_writer.add_uint32('magic', 0x67676d66)
    gguf_writer.add_uint32('version', 1)
    gguf_writer.add_uint32('n_vocab', hparams["vocab_size"])
    gguf_writer.add_embedding_length(hparams["hidden_size"])
    gguf_writer.add_head_count(hparams["num_attention_heads"])
    gguf_writer.add_head_count_kv(hparams["num_key_value_heads"])

    gguf_writer.add_block_count(hparams["num_hidden_layers"])
    gguf_writer.add_rope_dimension_count(n_rot)
    gguf_writer.add_uint32('ftype', ftype)
    gguf_writer.add_context_length(hparams["max_position_embeddings"])
    gguf_writer.add_feed_forward_length(hparams["intermediate_size"])

    gguf_writer.add_bos_token_id(tokenizer.bos_token_id)
    gguf_writer.add_eos_token_id(tokenizer.eos_token_id)
    gguf_writer.add_pad_token_id(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    gguf_writer.add_sep_token_id(tokenizer.sep_token_id if tokenizer.sep_token_id is not None else 0)

    def write_vocab_gguf(dir_model, hparams, gguf_writer):
        tokens: list[bytearray] = []
        toktypes: list[int] = []

        from transformers import AutoTokenizer  # type: ignore[attr-defined]
        tokenizer = AutoTokenizer.from_pretrained(dir_model, trust_remote_code=True)
        vocab_size = hparams.get("vocab_size", len(tokenizer.vocab))
        assert max(tokenizer.vocab.values()) < vocab_size

        reverse_vocab = {id_: encoded_tok for encoded_tok, id_ in tokenizer.vocab.items()}
        added_vocab = tokenizer.get_added_vocab()

        for i in range(vocab_size):
            if i not in reverse_vocab:
                pad_token = f"[PAD{i}]".encode('utf-8')
                tokens.append(bytearray(pad_token))
                toktypes.append(gguf.TokenType.USER_DEFINED)
            elif reverse_vocab[i] in added_vocab:
                tokens.append(reverse_vocab[i])
                if tokenizer.added_tokens_decoder[i].special:
                    toktypes.append(gguf.TokenType.CONTROL)
                else:
                    toktypes.append(gguf.TokenType.USER_DEFINED)
            else:
                tokens.append(reverse_vocab[i])
                toktypes.append(gguf.TokenType.NORMAL)

        gguf_writer.add_tokenizer_model("gpt2")
        gguf_writer.add_token_list(tokens)
        gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(dir_model, load_merges=True)
        special_vocab.add_to_gguf(gguf_writer)

    write_vocab_gguf(dir_model, hparams, gguf_writer)

    # tensor info
    print("gguf: get tensor metadata")
    for name in list_vars.keys():
        data = list_vars[name].squeeze().numpy()

        print("Processing variable: " + name + " with shape: ", data.shape)
        if 'inv_freq' in name:
            continue

        n_dims = len(data.shape)

        # ftype == 0 -> float32, ftype == 1 -> float16
        ftype_cur = 0
        if ftype != 0:
            if name[-7:] == ".weight" and n_dims == 2:
                print("  Converting to float16")
                data = data.astype(np.float16)
                ftype_cur = 1
            else:
                print("  Converting to float32")
                data = data.astype(np.float32)
                ftype_cur = 0
        else:
            if data.dtype != np.float32:
                print("  Converting to float32")
                data = data.astype(np.float32)
                ftype_cur = 0

        # print(f"[{i+1:{padi}d}/{len(model)}]
        # Writing tensor {name:38s} | size {size:16} | type {lazy_tensor.data_type.name:4}")

        gguf_writer.add_tensor(name, data)

    print("gguf: write header")
    gguf_writer.write_header_to_file()
    print("gguf: write metadata")
    gguf_writer.write_kv_data_to_file()
    print("gguf: write tensors")
    gguf_writer.write_tensors_to_file()

    gguf_writer.close()

    print("Done. Output file: " + fname_out)
    print("")

def phi_convert(model, tokenizer, dir_model, fname_out, ftype, hparams):
    n_rot = int(hparams["partial_rotary_factor"]*hparams["hidden_size"]/hparams["num_attention_heads"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    hparams = model.config.to_dict()
    print("Model loaded: ", dir_model)

    fout = open(fname_out, "wb")

    # 0x67676d6c is unversioned ne
    # 0x67676d66 is versioned ggmf (requires token scores)
    ne_file_magic = 0x67676d66
    #ne_file_version = 0x00000001 # v1

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
    fout.write(struct.pack("f", hparams.get("layer_norm_eps", 1e-5)))  # rms_norm_eps or layer_norm_eps
    fout.write(struct.pack("f", 10000.0))  # freq_base
    fout.write(struct.pack("f", 1.0))  # rope_factor
    fout.write(struct.pack("f", 0.0)) # config.json "rope_scaling.factor", not enabled
    fout.write(struct.pack("i", 0))   # rope_scaling.original_max_position_embeddings
    fout.write(struct.pack("i", 0))   # params["rope_scaling"]["type"] =="yarn" else 0))
    fout.write(struct.pack("i", tokenizer.bos_token_id if tokenizer.bos_token_id is not None else -1))
    fout.write(struct.pack("i", tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1))
    fout.write(struct.pack("i", tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1))
    fout.write(struct.pack("i", tokenizer.sep_token_id if tokenizer.sep_token_id is not None else -1))

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

    list_vars = model.state_dict()

    print(hparams)

    for name in list_vars.keys():
        # No gradients for these
        list_vars[name].requires_grad = False
        src = name
        print(src, ' -> ', name)
        data = list_vars[src].squeeze().numpy()
        data = data.astype(np.float32)

        n_dims = len(data.shape)
        print(name, n_dims, data.shape)

        # default type is fp32
        ftype_cur = 0
        if ftype == 1 and n_dims > 1:
            print("  Converting to float16", data.shape, data[:3, :3].tolist())
            data = data.astype(np.float16)
            ftype_cur = 1
        else:
            print("  Converting to float32", data.shape, data[:3, :3].tolist() if n_dims > 1 else data[:3].tolist())
            data = data.astype(np.float32)

        # header
        str = name.encode('utf-8')
        fout.write(struct.pack("iii", n_dims, len(str), ftype_cur))
        for i in range(n_dims):
            fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
        print(str)
        fout.write(str)

        # data
        data.tofile(fout)

    fout.close()

    print("Done. Output file: " + fname_out)
    print("")

def main(args_in: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Convert a model to a NE compatible file")
    parser.add_argument("--outtype", choices=["f32", "f16"], help="output format (default: based on input)")
    parser.add_argument("--outfile", type=Path, help="path to write to; default: based on input")
    parser.add_argument("--model_hub", choices=["huggingface","modelscope"], default="huggingface",
                        help="hub to load model")
    parser.add_argument("model", type=Path, help="directory containing model file")
    parser.add_argument("--format",
                        type=str,
                        default="NE",
                        choices=["NE", "GGUF"],
                        help="convert to the GGUF or NE format")
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
    print("Loading model: ", dir_model)
    model = AutoModelForCausalLM.from_pretrained(dir_model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(dir_model, trust_remote_code=True)
    hparams = model.config.to_dict()
    if args.format == "GGUF":
        phi_convert_gguf(model, tokenizer, dir_model, fname_out, ftype, hparams)
    else:
        phi_convert(model, tokenizer, dir_model, fname_out, ftype, hparams)



if __name__ == '__main__':
    main()
