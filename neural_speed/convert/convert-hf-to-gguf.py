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

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ContextManager, Iterator, Sequence, cast

import numpy as np
import torch

if TYPE_CHECKING:
    from torch import Tensor

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / 'gguf-py'))
import gguf


###### MODEL DEFINITIONS ######


class SentencePieceTokenTypes(IntEnum):
    NORMAL = 1
    UNKNOWN = 2
    CONTROL = 3
    USER_DEFINED = 4
    UNUSED = 5
    BYTE = 6


class HfVocab:
    def __init__(self, fname_tokenizer: Path, fname_added_tokens: Path | None = None) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "To use HfVocab, please install the `transformers` package. "
                "You can install it with `pip install transformers`."
            ) from e

        print("fname_tokenizer:", fname_tokenizer)
        # Allow the tokenizer to default to slow or fast versions.
        # Explicitly set tokenizer to use local paths.
        self.tokenizer = AutoTokenizer.from_pretrained(
            fname_tokenizer,
            cache_dir=fname_tokenizer,
            local_files_only=True,
        )

        # Initialize lists and dictionaries for added tokens
        self.added_tokens_list = []
        self.added_tokens_dict = dict()
        self.added_tokens_ids  = set()

        # Process added tokens
        for tok, tokidx in sorted(
            self.tokenizer.get_added_vocab().items(), key=lambda x: x[1]
        ):
            # Only consider added tokens that are not in the base vocabulary
            if tokidx >= self.tokenizer.vocab_size:
                self.added_tokens_list.append(tok)
                self.added_tokens_dict[tok] = tokidx
                self.added_tokens_ids.add(tokidx)

        # Store special tokens and their IDs
        self.specials = {
            tok: self.tokenizer.get_vocab()[tok]
            for tok in self.tokenizer.all_special_tokens
        }
        self.special_ids = set(self.tokenizer.all_special_ids)

        # Set vocabulary sizes
        self.vocab_size_base = self.tokenizer.vocab_size
        self.vocab_size      = self.vocab_size_base + len(self.added_tokens_list)

        self.fname_tokenizer    = fname_tokenizer
        self.fname_added_tokens = fname_added_tokens

    def hf_tokens(self) -> Iterable[tuple[bytes, float, gguf.TokenType]]:
        reverse_vocab = {
            id: encoded_tok for encoded_tok, id in self.tokenizer.get_vocab().items()
        }

        for token_id in range(self.vocab_size_base):
            # Skip processing added tokens here
            if token_id in self.added_tokens_ids:
                continue

            # Convert token text to bytes
            token_text = reverse_vocab[token_id].encode("utf-8")

            # Yield token text, score, and type
            yield token_text, self.get_token_score(token_id), self.get_token_type(
                token_id, token_text, self.special_ids  # Reuse already stored special IDs
            )

    def get_token_type(self, token_id: int, token_text: bytes, special_ids: set[int]) -> gguf.TokenType:
        # Special case for byte tokens
        if re.fullmatch(br"<0x[0-9A-Fa-f]{2}>", token_text):
            return gguf.TokenType.BYTE

        # Determine token type based on whether it's a special token
        return gguf.TokenType.CONTROL if token_id in special_ids else gguf.TokenType.NORMAL

    def get_token_score(self, token_id: int) -> float:
        # Placeholder for actual logic to determine the token's score
        # This needs to be implemented based on specific requirements
        return -1000.0  # Default score

    def added_tokens(self) -> Iterable[tuple[bytes, float, gguf.TokenType]]:
        for text in self.added_tokens_list:
            if text in self.specials:
                toktype = self.get_token_type(self.specials[text], b'', self.special_ids)
                score = self.get_token_score(self.specials[text])
            else:
                toktype = gguf.TokenType.USER_DEFINED
                score = -1000.0

            yield text.encode("utf-8"), score, toktype

    def has_newline_token(self):
        return "<0x0A>" in self.tokenizer.vocab or "\n" in self.tokenizer.vocab

    def all_tokens(self) -> Iterable[tuple[bytes, float, gguf.TokenType]]:
        yield from self.hf_tokens()
        yield from self.added_tokens()

    def __repr__(self) -> str:
        return f"<HfVocab with {self.vocab_size_base} base tokens and {len(self.added_tokens_list)} added tokens>"

class Model:
    def __init__(self, dir_model: Path, ftype: int, fname_out: Path, is_big_endian: bool):
        self.dir_model = dir_model
        self.ftype = ftype
        self.fname_out = fname_out
        self.is_big_endian = is_big_endian
        self.endianess = gguf.GGUFEndian.BIG if is_big_endian else gguf.GGUFEndian.LITTLE
        self.is_safetensors = self._is_model_safetensors()
        self.num_parts = Model.count_model_parts(self.dir_model, ".safetensors" if self.is_safetensors else ".bin")
        self.part_names = self._get_part_names()
        self.hparams = Model.load_hparams(self.dir_model)
        self.model_arch = self._get_model_architecture()
        self.gguf_writer = gguf.GGUFWriter(fname_out,
                                           gguf.MODEL_ARCH_NAMES[self.model_arch],
                                           endianess=self.endianess,
                                           use_temp_file=False)
        self.block_count = self.find_hparam(["n_layers", "num_hidden_layers", "n_layer"])

    def find_hparam(self, keys: Sequence[str], optional: bool = False) -> Any:
        key = next((k for k in keys if k in self.hparams), None)
        if key is not None:
            return self.hparams[key]
        if optional:
            return None
        raise KeyError(f"could not find any of: {keys}")

    def set_vocab(self):
        self._set_vocab_gpt2()

    def get_tensors(self) -> Iterator[tuple[str, Tensor]]:
        for part_name in self.part_names:
            print(f"gguf: loading model part '{part_name}'")
            ctx: ContextManager[Any]
            if self.is_safetensors:
                from safetensors import safe_open
                ctx = cast(ContextManager[Any], safe_open(self.dir_model / part_name, framework="pt", device="cpu"))
            else:
                ctx = contextlib.nullcontext(
                    torch.load(str(self.dir_model / part_name), map_location="cpu", mmap=True, weights_only=True))

            with ctx as model_part:
                for name in model_part.keys():
                    data = model_part.get_tensor(name) if self.is_safetensors else model_part[name]
                    yield name, data

    def set_gguf_parameters(self):
        self.gguf_writer.add_name(self.dir_model.name)
        self.gguf_writer.add_block_count(self.block_count)

        if (n_ctx := self.find_hparam(["max_position_embeddings", "n_ctx"], optional=True)) is not None:
            self.gguf_writer.add_context_length(n_ctx)

        n_embd = self.find_hparam(["hidden_size", "n_embd"])
        self.gguf_writer.add_embedding_length(n_embd)

        if (n_ff := self.find_hparam(["intermediate_size", "n_inner"], optional=True)) is not None:
            self.gguf_writer.add_feed_forward_length(n_ff)

        n_head = self.find_hparam(["num_attention_heads", "n_head"])
        self.gguf_writer.add_head_count(n_head)

        if (n_head_kv := self.hparams.get("num_key_value_heads")) is not None:
            self.gguf_writer.add_head_count_kv(n_head_kv)

        if (f_rms_eps := self.hparams.get("rms_norm_eps")) is not None:
            self.gguf_writer.add_layer_norm_rms_eps(f_rms_eps)
        if (f_norm_eps := self.find_hparam(["layer_norm_eps", "layer_norm_epsilon"], optional=True)) is not None:
            self.gguf_writer.add_layer_norm_eps(f_norm_eps)
        if (n_experts := self.hparams.get("num_local_experts")) is not None:
            self.gguf_writer.add_expert_count(n_experts)
        if (n_experts_used := self.hparams.get("num_experts_per_tok")) is not None:
            self.gguf_writer.add_expert_used_count(n_experts_used)

        self.gguf_writer.add_file_type(self.ftype)

    def write_tensors(self):
        block_count = self.hparams.get("n_layers", self.hparams.get("num_hidden_layers", self.hparams.get("n_layer")))
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)
        for name, data_torch in self.get_tensors():
            # we don't need these
            if name.endswith((".attention.masked_bias", ".attention.bias", ".attention.rotary_emb.inv_freq")):
                continue

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            data = data_torch.squeeze().numpy()

            # map tensor names
            new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                print(f"Can not map tensor {name!r}")
                sys.exit()

            n_dims = len(data.shape)
            data_dtype = data.dtype

            # if f32 desired, convert any float16 to float32
            if self.ftype == 0 and data_dtype == np.float16:
                data = data.astype(np.float32)

            # TODO: Why can't we use these float16 as-is? There should be not reason to store float16 as float32
            if self.ftype == 1 and data_dtype == np.float16 and n_dims == 1:
                data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if self.ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
                data = data.astype(np.float16)

            print(f"{name} -> {new_name}, n_dims = {n_dims}, data.shape = {data.shape}, {old_dtype} --> {data.dtype}")

            self.gguf_writer.add_tensor(new_name, data)

    def write(self):
        self.write_tensors()
        self.gguf_writer.write_header_to_file()
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.write_tensors_to_file()
        self.gguf_writer.close()

    def write_vocab(self):
        self.gguf_writer.write_header_to_file()
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.close()

    @staticmethod
    def count_model_parts(dir_model: Path, prefix: str) -> int:
        num_parts = 0
        for filename in os.listdir(dir_model):
            if filename.endswith(prefix):
                num_parts += 1

        return num_parts

    @staticmethod
    def load_hparams(dir_model):
        with open(dir_model / "config.json", "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def from_model_architecture(model_architecture):
        if model_architecture == "GPTNeoXForCausalLM":
            return GPTNeoXModel
        if model_architecture == "BloomForCausalLM":
            return BloomModel
        if model_architecture == "MPTForCausalLM":
            return MPTModel
        if model_architecture in ("BaichuanForCausalLM", "BaiChuanForCausalLM"):
            return BaichuanModel
        if model_architecture in ("FalconForCausalLM", "RWForCausalLM"):
            return FalconModel
        if model_architecture == "GPTBigCodeForCausalLM":
            return StarCoderModel
        if model_architecture == "GPTRefactForCausalLM":
            return RefactModel
        if model_architecture == "PersimmonForCausalLM":
            return PersimmonModel
        if model_architecture in ("StableLMEpochForCausalLM", "LlavaStableLMEpochForCausalLM"):
            return StableLMModel
        if model_architecture == "QWenLMHeadModel":
            return QwenModel
        if model_architecture == "Qwen2ForCausalLM":
            return Model
        if model_architecture == "MixtralForCausalLM":
            return MixtralModel
        if model_architecture == "GPT2LMHeadModel":
            return GPT2Model
        if model_architecture == "PhiForCausalLM":
            return Phi2Model
        if model_architecture == "PlamoForCausalLM":
            return PlamoModel
        if model_architecture == "CodeShellForCausalLM":
            return CodeShellModel
        if model_architecture == "OrionForCausalLM":
            return OrionModel
        if model_architecture == "InternLM2ForCausalLM":
            return InternLM2Model
        if model_architecture == "MiniCPMForCausalLM":
            return MiniCPMModel
        if model_architecture == "BertModel":
            return BertModel
        if model_architecture == "NomicBertModel":
            return NomicBertModel
        return Model

    def _is_model_safetensors(self) -> bool:
        return Model.count_model_parts(self.dir_model, ".safetensors") > 0

    def _get_part_names(self):
        if self.is_safetensors:
            if self.num_parts == 1:  # there's only one .safetensors file
                return ("model.safetensors", )
            return (f"model-{n:05}-of-{self.num_parts:05}.safetensors" for n in range(1, self.num_parts + 1))

        if self.num_parts == 1:  # there's only one .bin file
            return ("pytorch_model.bin", )
        return (f"pytorch_model-{n:05}-of-{self.num_parts:05}.bin" for n in range(1, self.num_parts + 1))

    def _get_model_architecture(self) -> gguf.MODEL_ARCH:
        arch = self.hparams["architectures"][0]
        if arch == "GPTNeoXForCausalLM":
            return gguf.MODEL_ARCH.GPTNEOX
        if arch == "BloomForCausalLM":
            return gguf.MODEL_ARCH.BLOOM
        if arch == "MPTForCausalLM":
            return gguf.MODEL_ARCH.MPT
        if arch in ("BaichuanForCausalLM", "BaiChuanForCausalLM"):
            return gguf.MODEL_ARCH.BAICHUAN
        if arch in ("FalconForCausalLM", "RWForCausalLM"):
            return gguf.MODEL_ARCH.FALCON
        if arch == "GPTBigCodeForCausalLM":
            return gguf.MODEL_ARCH.STARCODER
        if arch == "GPTRefactForCausalLM":
            return gguf.MODEL_ARCH.REFACT
        if arch == "PersimmonForCausalLM":
            return gguf.MODEL_ARCH.PERSIMMON
        if arch in ("StableLMEpochForCausalLM", "LlavaStableLMEpochForCausalLM"):
            return gguf.MODEL_ARCH.STABLELM
        if arch == "QWenLMHeadModel":
            return gguf.MODEL_ARCH.QWEN
        if arch == "Qwen2ForCausalLM":
            return gguf.MODEL_ARCH.QWEN2
        if arch == "MixtralForCausalLM":
            return gguf.MODEL_ARCH.LLAMA
        if arch == "GPT2LMHeadModel":
            return gguf.MODEL_ARCH.GPT2
        if arch == "PhiForCausalLM":
            return gguf.MODEL_ARCH.PHI2
        if arch == "PlamoForCausalLM":
            return gguf.MODEL_ARCH.PLAMO
        if arch == "CodeShellForCausalLM":
            return gguf.MODEL_ARCH.CODESHELL
        if arch == "OrionForCausalLM":
            return gguf.MODEL_ARCH.ORION
        if arch == "InternLM2ForCausalLM":
            return gguf.MODEL_ARCH.INTERNLM2
        if arch == "MiniCPMForCausalLM":
            return gguf.MODEL_ARCH.MINICPM
        if arch == "BertModel":
            return gguf.MODEL_ARCH.BERT
        if arch == "NomicBertModel":
            return gguf.MODEL_ARCH.NOMIC_BERT

        raise NotImplementedError(f'Architecture "{arch}" not supported!')

    def _set_vocab_gpt2(self):
        dir_model = self.dir_model
        hparams = self.hparams
        tokens: list[bytearray] = []
        toktypes: list[int] = []

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(dir_model)
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

        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(dir_model, load_merges=True)
        special_vocab.add_to_gguf(self.gguf_writer)

    def _set_vocab_qwen(self):
        dir_model = self.dir_model
        hparams = self.hparams
        tokens: list[bytearray] = []
        toktypes: list[int] = []

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(dir_model, trust_remote_code=True)
        vocab_size = hparams["vocab_size"]
        assert max(tokenizer.get_vocab().values()) < vocab_size

        merges = []
        vocab = {}
        mergeable_ranks = tokenizer.mergeable_ranks
        for token, rank in mergeable_ranks.items():
            vocab[QwenModel.token_bytes_to_string(token)] = rank
            if len(token) == 1:
                continue
            merged = QwenModel.bpe(mergeable_ranks, token, max_rank=rank)
            assert len(merged) == 2
            merges.append(' '.join(map(QwenModel.token_bytes_to_string, merged)))

        # for this kind of tokenizer, added_vocab is not a subset of vocab, so they need to be combined
        added_vocab = tokenizer.special_tokens
        reverse_vocab = {id_: encoded_tok for encoded_tok, id_ in (vocab | added_vocab).items()}

        for i in range(vocab_size):
            if i not in reverse_vocab:
                pad_token = f"[PAD{i}]".encode("utf-8")
                tokens.append(bytearray(pad_token))
                toktypes.append(gguf.TokenType.USER_DEFINED)
            elif reverse_vocab[i] in added_vocab:
                tokens.append(reverse_vocab[i])
                toktypes.append(gguf.TokenType.CONTROL)
            else:
                tokens.append(reverse_vocab[i])
                toktypes.append(gguf.TokenType.NORMAL)

        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(dir_model, load_merges=False)
        special_vocab.merges = merges
        # only add special tokens when they were not already loaded from config.json
        if len(special_vocab.special_token_ids) == 0:
            special_vocab._set_special_token("bos", tokenizer.special_tokens["<|endoftext|>"])
            special_vocab._set_special_token("eos", tokenizer.special_tokens["<|endoftext|>"])
        # this one is usually not in config.json anyway
        special_vocab._set_special_token("unk", tokenizer.special_tokens["<|endoftext|>"])
        special_vocab.add_to_gguf(self.gguf_writer)

    def _set_vocab_sentencepiece(self):
        from sentencepiece import SentencePieceProcessor

        tokenizer_path = self.dir_model / 'tokenizer.model'

        tokens: list[bytes] = []
        scores: list[float] = []
        toktypes: list[int] = []

        if not tokenizer_path.is_file():
            print(f'Error: Missing {tokenizer_path}', file=sys.stderr)
            sys.exit(1)

        tokenizer = SentencePieceProcessor(str(tokenizer_path))
        vocab_size = self.hparams.get('vocab_size', tokenizer.vocab_size())

        for token_id in range(vocab_size):
            piece = tokenizer.id_to_piece(token_id)
            text = piece.encode("utf-8")
            score = tokenizer.get_score(token_id)

            toktype = SentencePieceTokenTypes.NORMAL
            if tokenizer.is_unknown(token_id):
                toktype = SentencePieceTokenTypes.UNKNOWN
            elif tokenizer.is_control(token_id):
                toktype = SentencePieceTokenTypes.CONTROL
            elif tokenizer.is_unused(token_id):
                toktype = SentencePieceTokenTypes.UNUSED
            elif tokenizer.is_byte(token_id):
                toktype = SentencePieceTokenTypes.BYTE

            tokens.append(text)
            scores.append(score)
            toktypes.append(toktype)

        added_tokens_file = self.dir_model / 'added_tokens.json'
        if added_tokens_file.is_file():
            with open(added_tokens_file, "r", encoding="utf-8") as f:
                added_tokens_json = json.load(f)

                for key in added_tokens_json:
                    tokens.append(key.encode("utf-8"))
                    scores.append(-1000.0)
                    toktypes.append(SentencePieceTokenTypes.USER_DEFINED)

        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

    def _set_vocab_hf(self):
        path = self.dir_model
        added_tokens_path = self.dir_model
        vocab = HfVocab(path, added_tokens_path if added_tokens_path.exists() else None)
        tokens = []
        scores = []
        toktypes = []

        for text, score, toktype in vocab.all_tokens():
            tokens.append(text)
            scores.append(score)
            toktypes.append(toktype)

        assert len(tokens) == vocab.vocab_size

        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)


class GPTNeoXModel(Model):
    def set_gguf_parameters(self):
        block_count = self.hparams["num_hidden_layers"]

        self.gguf_writer.add_name(self.dir_model.name)
        self.gguf_writer.add_context_length(self.hparams["max_position_embeddings"])
        self.gguf_writer.add_embedding_length(self.hparams["hidden_size"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_feed_forward_length(self.hparams["intermediate_size"])
        self.gguf_writer.add_rope_dimension_count(
            int(self.hparams["rotary_pct"] * (self.hparams["hidden_size"] // self.hparams["num_attention_heads"])), )
        self.gguf_writer.add_head_count(self.hparams["num_attention_heads"])
        self.gguf_writer.add_parallel_residual(self.hparams.get("use_parallel_residual", True))
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_eps"])


class BloomModel(Model):
    def set_gguf_parameters(self):
        self.gguf_writer.add_name("Bloom")
        n_embed = self.hparams.get("hidden_size", self.hparams.get("n_embed"))
        n_head = self.hparams.get("n_head", self.hparams.get("num_attention_heads"))
        self.gguf_writer.add_context_length(self.hparams.get("seq_length", n_embed))
        self.gguf_writer.add_embedding_length(n_embed)
        self.gguf_writer.add_feed_forward_length(4 * n_embed)
        self.gguf_writer.add_block_count(self.hparams["n_layer"])
        self.gguf_writer.add_head_count(n_head)
        self.gguf_writer.add_head_count_kv(n_head)
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)

    def write_tensors(self):
        block_count = self.hparams["n_layer"]
        tensors = dict(self.get_tensors())
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)
        has_lm_head = True
        n_head = self.hparams.get("n_head", self.hparams.get("num_attention_heads"))
        n_embed = self.hparams.get("hidden_size", self.hparams.get("n_embed"))

        for name, data_torch in tensors.items():
            if "lm_head.weight" not in tensors.keys() and "output.weight" not in tensors.keys():
                has_lm_head = False

            name = re.sub(r'transformer\.', '', name)

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            data = data_torch.squeeze().numpy()

            # Map bloom-style qkv_linear to gpt-style qkv_linear
            # bloom:
            # github.com/huggingface/transformers/blob/main/src/transformers/models/bloom/modeling_bloom.py#L238-L252
            # gpt-2:
            # github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py #L312
            if re.match(r"h\.\d+\.self_attention\.query_key_value\.weight", name):
                qkv_weights = data.reshape((n_head, 3, n_embed // n_head, n_embed))
                data = np.concatenate(
                    (
                        qkv_weights[:, 0, :, :].reshape((-1, n_embed)),
                        qkv_weights[:, 1, :, :].reshape((-1, n_embed)),
                        qkv_weights[:, 2, :, :].reshape((-1, n_embed)),
                    ),
                    axis=0,
                )
                print("re-format attention.linear_qkv.weight")
            elif re.match(r"h\.\d+\.self_attention\.query_key_value\.bias", name):
                qkv_bias = data.reshape((n_head, 3, n_embed // n_head))
                data = np.concatenate(
                    (
                        qkv_bias[:, 0, :].reshape((n_embed, )),
                        qkv_bias[:, 1, :].reshape((n_embed, )),
                        qkv_bias[:, 2, :].reshape((n_embed, )),
                    ),
                    axis=0,
                )
                print("re-format attention.linear_qkv.bias")

            # map tensor names
            new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                print(f"Can not map tensor {name!r}")
                sys.exit()

            n_dims = len(data.shape)
            data_dtype = data.dtype

            # if f32 desired, convert any float16 to float32
            if self.ftype == 0 and data_dtype == np.float16:
                data = data.astype(np.float32)

            # TODO: Why can't we use these float16 as-is? There should be not reason to store float16 as float32
            if self.ftype == 1 and data_dtype == np.float16 and n_dims == 1:
                data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if self.ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
                data = data.astype(np.float16)

            print(f"=> {new_name}, shape = {data.shape}, {old_dtype} --> {data.dtype}")

            self.gguf_writer.add_tensor(new_name, data)

            if not has_lm_head and name == "word_embeddings.weight":
                self.gguf_writer.add_tensor("output.weight", data)
                print(name, f"=> output.weight, shape = {data.shape}, {old_dtype} --> {data.dtype}")


class MPTModel(Model):
    def set_gguf_parameters(self):
        block_count = self.hparams["n_layers"]
        self.gguf_writer.add_name(self.dir_model.name)
        self.gguf_writer.add_context_length(self.hparams["max_seq_len"])
        self.gguf_writer.add_embedding_length(self.hparams["d_model"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_feed_forward_length(4 * self.hparams["d_model"])
        self.gguf_writer.add_head_count(self.hparams["n_heads"])
        if kv_n_heads := self.hparams["attn_config"].get("kv_n_heads"):
            self.gguf_writer.add_head_count_kv(kv_n_heads)
        self.gguf_writer.add_layer_norm_eps(1e-5)
        if self.hparams["attn_config"]["clip_qkv"] is not None:
            self.gguf_writer.add_clamp_kqv(self.hparams["attn_config"]["clip_qkv"])
        self.gguf_writer.add_max_alibi_bias(self.hparams["attn_config"]["alibi_bias_max"])

    def write_tensors(self):
        block_count = self.hparams.get("n_layers", self.hparams.get("num_hidden_layers"))
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)
        for name, data_torch in self.get_tensors():
            # we don't need these
            if name.endswith((".attention.masked_bias", ".attention.bias", ".attention.rotary_emb.inv_freq")):
                continue

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            data = data_torch.squeeze().numpy()

            # map tensor names
            if "scales" in name:
                new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias", ".scales"))
                if new_name is not None:
                    new_name = new_name.replace("scales", "act.scales")
            else:
                new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                print(f"Can not map tensor {name!r}")
                sys.exit()

            n_dims = len(data.shape)
            data_dtype = data.dtype

            # if f32 desired, convert any float16 to float32
            if self.ftype == 0 and data_dtype == np.float16:
                data = data.astype(np.float32)

            # TODO: Why can't we use these float16 as-is? There should be not reason to store float16 as float32
            if self.ftype == 1 and data_dtype == np.float16 and n_dims == 1:
                data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if self.ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
                data = data.astype(np.float16)

            print(f"{name} -> {new_name}, n_dims = {n_dims}, data.shape = {data.shape}, {old_dtype} --> {data.dtype}")

            self.gguf_writer.add_tensor(new_name, data)

            # note: MPT output is tied to (same as) wte in original model;
            # for easier implementation in llama.cpp it's duplicated in GGUF, though :/
            if new_name == "token_embd.weight":
                self.gguf_writer.add_tensor("output.weight", data)


class OrionModel(Model):
    def set_vocab(self):
        self._set_vocab_sentencepiece()

    def set_gguf_parameters(self):
        block_count = self.hparams["num_hidden_layers"]
        head_count = self.hparams["num_attention_heads"]
        head_count_kv = self.hparams.get("num_key_value_heads", head_count)
        hf_repo = self.hparams.get("_name_or_path", "")

        ctx_length = 0
        if "max_sequence_length" in self.hparams:
            ctx_length = self.hparams["max_sequence_length"]
        elif "max_position_embeddings" in self.hparams:
            ctx_length = self.hparams["max_position_embeddings"]
        elif "model_max_length" in self.hparams:
            ctx_length = self.hparams["model_max_length"]
        else:
            print("gguf: can not find ctx length parameter.")
            sys.exit()

        self.gguf_writer.add_file_type(self.ftype)
        self.gguf_writer.add_name(self.dir_model.name)
        self.gguf_writer.add_source_hf_repo(hf_repo)
        self.gguf_writer.add_tensor_data_layout("Meta AI original pth")
        self.gguf_writer.add_context_length(ctx_length)
        self.gguf_writer.add_embedding_length(self.hparams["hidden_size"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_feed_forward_length(self.hparams["intermediate_size"])
        self.gguf_writer.add_head_count(head_count)
        self.gguf_writer.add_head_count_kv(head_count_kv)
        self.gguf_writer.add_layer_norm_eps(self.hparams["rms_norm_eps"])

    def write_tensors(self):
        # Collect tensors from generator object
        model_kv = dict(self.get_tensors())
        block_count = self.hparams["num_hidden_layers"]
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)

        for name, data_torch in model_kv.items():
            # we don't need these
            if name.endswith(".rotary_emb.inv_freq"):
                continue

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            data = data_torch.squeeze().numpy()

            # map tensor names
            new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                print(f"Can not map tensor {name!r}")
                sys.exit()

            n_dims = len(data.shape)
            data_dtype = data.dtype

            # if f32 desired, convert any float16 to float32
            if self.ftype == 0 and data_dtype == np.float16:
                data = data.astype(np.float32)

            # TODO: Why can't we use these float16 as-is? There should be not reason to store float16 as float32
            if self.ftype == 1 and data_dtype == np.float16 and n_dims == 1:
                data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if self.ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
                data = data.astype(np.float16)

            print(f"{name} -> {new_name}, n_dims = {n_dims}, {old_dtype} --> {data.dtype}")
            self.gguf_writer.add_tensor(new_name, data)


class BaichuanModel(Model):
    def set_vocab(self):
        self._set_vocab_sentencepiece()

    def set_gguf_parameters(self):
        block_count = self.hparams["num_hidden_layers"]
        head_count = self.hparams["num_attention_heads"]
        head_count_kv = self.hparams.get("num_key_value_heads", head_count)
        hf_repo = self.hparams.get("_name_or_path", "")

        ctx_length = 0
        if "max_sequence_length" in self.hparams:
            ctx_length = self.hparams["max_sequence_length"]
        elif "max_position_embeddings" in self.hparams:
            ctx_length = self.hparams["max_position_embeddings"]
        elif "model_max_length" in self.hparams:
            ctx_length = self.hparams["model_max_length"]
        else:
            print("gguf: can not find ctx length parameter.")
            sys.exit()

        self.gguf_writer.add_name(self.dir_model.name)
        self.gguf_writer.add_source_hf_repo(hf_repo)
        self.gguf_writer.add_tensor_data_layout("Meta AI original pth")
        self.gguf_writer.add_context_length(ctx_length)
        self.gguf_writer.add_embedding_length(self.hparams["hidden_size"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_feed_forward_length(self.hparams["intermediate_size"])
        self.gguf_writer.add_rope_dimension_count(self.hparams["hidden_size"] // self.hparams["num_attention_heads"])
        self.gguf_writer.add_head_count(head_count)
        self.gguf_writer.add_head_count_kv(head_count_kv)
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["rms_norm_eps"])

        if self.hparams.get("rope_scaling") is not None and "factor" in self.hparams["rope_scaling"]:
            if self.hparams["rope_scaling"].get("type") == "linear":
                self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
                self.gguf_writer.add_rope_scaling_factor(self.hparams["rope_scaling"]["factor"])

    def write_tensors(self):
        # Collect tensors from generator object
        model_kv = dict(self.get_tensors())
        block_count = self.hparams["num_hidden_layers"]
        head_count = self.hparams["num_attention_heads"]
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)
        head_count_kv = self.hparams.get("num_key_value_heads", head_count)

        for i in range(block_count):
            if (w := model_kv.get(f"model.layers.{i}.self_attn.W_pack.weight")) is not None:
                print(f"Unpacking and permuting layer {i}")
                model_kv[f"model.layers.{i}.self_attn.q_proj.weight"] = \
                    self._reverse_hf_permute_part(w, 0, head_count, head_count)
                model_kv[f"model.layers.{i}.self_attn.k_proj.weight"] = \
                    self._reverse_hf_permute_part(w, 1, head_count, head_count_kv)
                model_kv[f"model.layers.{i}.self_attn.v_proj.weight"] = \
                    self._reverse_hf_part(w, 2)
                del model_kv[f"model.layers.{i}.self_attn.W_pack.weight"]

        for name, data_torch in model_kv.items():
            # we don't need these
            if name.endswith(".rotary_emb.inv_freq"):
                continue

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            data = data_torch.squeeze().numpy()

            # map tensor names
            new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                print(f"Can not map tensor {name!r}")
                sys.exit()

            n_dims = len(data.shape)
            data_dtype = data.dtype

            # if f32 desired, convert any float16 to float32
            if self.ftype == 0 and data_dtype == np.float16:
                data = data.astype(np.float32)

            # TODO: Why can't we use these float16 as-is? There should be not reason to store float16 as float32
            if self.ftype == 1 and data_dtype == np.float16 and n_dims == 1:
                data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if self.ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
                data = data.astype(np.float16)

            print(f"{name} -> {new_name}, n_dims = {n_dims}, {old_dtype} --> {data.dtype}")
            self.gguf_writer.add_tensor(new_name, data)

    def _reverse_hf_permute(self, weights: Tensor, n_head: int, n_kv_head: int | None = None) -> Tensor:
        if n_kv_head is not None and n_head != n_kv_head:
            n_head //= n_kv_head

        return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2,
                                *weights.shape[1:]).swapaxes(1, 2).reshape(weights.shape))

    def _reverse_hf_permute_part(
        self,
        weights: Tensor,
        n_part: int,
        n_head: int,
        n_head_kv: int | None = None,
    ) -> Tensor:
        r = weights.shape[0] // 3
        return self._reverse_hf_permute(weights[r * n_part:r * n_part + r, ...], n_head, n_head_kv)

    def _reverse_hf_part(self, weights: Tensor, n_part: int) -> Tensor:
        r = weights.shape[0] // 3
        return weights[r * n_part:r * n_part + r, ...]


class FalconModel(Model):
    def set_gguf_parameters(self):
        block_count = self.hparams.get("num_hidden_layers")
        if block_count is None:
            block_count = self.hparams["n_layer"]  # old name

        n_head = self.hparams.get("num_attention_heads")
        if n_head is None:
            n_head = self.hparams["n_head"]  # old name

        n_head_kv = self.hparams.get("num_kv_heads")
        if n_head_kv is None:
            n_head_kv = self.hparams.get("n_head_kv", 1)  # old name

        self.gguf_writer.add_name("Falcon")
        self.gguf_writer.add_context_length(2048)  # not in config.json
        self.gguf_writer.add_tensor_data_layout("jploski")  # qkv tensor transform
        self.gguf_writer.add_embedding_length(self.hparams["hidden_size"])
        self.gguf_writer.add_feed_forward_length(4 * self.hparams["hidden_size"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_head_count(n_head)
        self.gguf_writer.add_head_count_kv(n_head_kv)
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)

    def write_tensors(self):
        block_count = self.hparams.get("num_hidden_layers")
        if block_count is None:
            block_count = self.hparams["n_layer"]  # old name

        n_head = self.hparams.get("num_attention_heads")
        if n_head is None:
            n_head = self.hparams["n_head"]  # old name

        n_head_kv = self.hparams.get("num_kv_heads")
        if n_head_kv is None:
            n_head_kv = self.hparams.get("n_head_kv", 1)  # old name

        head_dim = self.hparams["hidden_size"] // n_head
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)

        for name, data_torch in self.get_tensors():
            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            # QKV tensor transform
            # The original query_key_value tensor contains n_head_kv "kv groups",
            # each consisting of n_head/n_head_kv query weights followed by one key
            # and one value weight (shared by all query heads in the kv group).
            # This layout makes it a big pain to work with in GGML.
            # So we rearrange them here,, so that we have n_head query weights
            # followed by n_head_kv key weights followed by n_head_kv value weights,
            # in contiguous fashion.
            # ref: https://github.com/jploski/ggml/blob/falcon40b/examples/falcon/convert-hf-to-ggml.py

            if "query_key_value" in name:
                qkv = data_torch.view(n_head_kv, n_head // n_head_kv + 2, head_dim, head_dim * n_head)
                q = qkv[:, :-2].reshape(n_head * head_dim, head_dim * n_head)
                k = qkv[:, [-2]].reshape(n_head_kv * head_dim, head_dim * n_head)
                v = qkv[:, [-1]].reshape(n_head_kv * head_dim, head_dim * n_head)
                data_torch = torch.cat((q, k, v)).reshape_as(data_torch)

            data = data_torch.squeeze().numpy()

            # map tensor names
            new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                print(f"Can not map tensor {name!r}")
                sys.exit()

            n_dims = len(data.shape)
            data_dtype = data.dtype

            # if f32 desired, convert any float16 to float32
            if self.ftype == 0 and data_dtype == np.float16:
                data = data.astype(np.float32)

            # TODO: Why can't we use these float16 as-is? There should be not reason to store float16 as float32
            if self.ftype == 1 and data_dtype == np.float16 and n_dims == 1:
                data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if self.ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
                data = data.astype(np.float16)

            print(f"{name} -> {new_name}, n_dims = {n_dims}, data.shape = {data.shape}, {old_dtype} --> {data.dtype}")

            self.gguf_writer.add_tensor(new_name, data)


class StarCoderModel(Model):
    def set_gguf_parameters(self):
        block_count = self.hparams["n_layer"]

        self.gguf_writer.add_name("StarCoder")
        self.gguf_writer.add_context_length(self.hparams["n_positions"])
        self.gguf_writer.add_embedding_length(self.hparams["n_embd"])
        self.gguf_writer.add_feed_forward_length(4 * self.hparams["n_embd"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_head_count(self.hparams["n_head"])
        self.gguf_writer.add_head_count_kv(1)
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)


class RefactModel(Model):
    def set_gguf_parameters(self):
        hidden_dim = self.hparams["n_embd"]
        inner_dim = 4 * hidden_dim
        hidden_dim = int(2 * inner_dim / 3)
        multiple_of = 256
        ff_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        block_count = self.hparams["n_layer"]

        self.gguf_writer.add_name("Refact")
        # refact uses Alibi. So this is from config.json which might be used by training.
        self.gguf_writer.add_context_length(self.hparams["n_positions"])
        self.gguf_writer.add_embedding_length(self.hparams["n_embd"])

        self.gguf_writer.add_feed_forward_length(ff_dim)
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_head_count(self.hparams["n_head"])
        self.gguf_writer.add_head_count_kv(1)
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)

    def write_tensors(self):
        hidden_dim = self.hparams["n_embd"]
        inner_dim = 4 * hidden_dim
        hidden_dim = int(2 * inner_dim / 3)
        multiple_of = 256
        ff_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        n_head = self.hparams["n_head"]
        n_head_kv = 1
        head_dim = self.hparams["n_embd"] // n_head
        block_count = self.hparams["n_layer"]

        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)

        tensors = dict(self.get_tensors())
        for i in range(block_count):
            if (w := tensors.get(f"transformer.h.{i}.attn.kv.weight")) is not None:
                tensors[f"model.layers.{i}.self_attn.k_proj.weight"] = w[:n_head_kv * head_dim]
                tensors[f"model.layers.{i}.self_attn.v_proj.weight"] = w[n_head_kv * head_dim:]
                del tensors[f"transformer.h.{i}.attn.kv.weight"]
            if (w := tensors.get(f"transformer.h.{i}.attn.q.weight")) is not None:
                tensors[f"model.layers.{i}.self_attn.q_proj.weight"] = w
                del tensors[f"transformer.h.{i}.attn.q.weight"]
            if (w := tensors.get(f"transformer.h.{i}.mlp.gate_up_proj.weight")) is not None:
                tensors[f"model.layers.{i}.mlp.gate_proj.weight"] = w[:ff_dim]
                tensors[f"model.layers.{i}.mlp.up_proj.weight"] = w[ff_dim:]
                del tensors[f"transformer.h.{i}.mlp.gate_up_proj.weight"]

        for name, data_torch in tensors.items():
            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            data = data_torch.squeeze().numpy()

            # map tensor names
            new_name = tensor_map.get_name(name, try_suffixes=(".weight", ))
            if new_name is None:
                print(f"Can not map tensor {name!r}")
                sys.exit()

            n_dims = len(data.shape)
            data_dtype = data.dtype

            # if f32 desired, convert any float16 to float32
            if self.ftype == 0 and data_dtype == np.float16:
                data = data.astype(np.float32)

            # TODO: Why can't we use these float16 as-is? There should be not reason to store float16 as float32
            if self.ftype == 1 and data_dtype == np.float16 and n_dims == 1:
                data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if self.ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
                data = data.astype(np.float16)

            print(f"{name} -> {new_name}, n_dims = {n_dims}, data.shape = {data.shape}, {old_dtype} --> {data.dtype}")

            self.gguf_writer.add_tensor(new_name, data)


class PersimmonModel(Model):
    def set_gguf_parameters(self):
        block_count = self.hparams.get("num_layers", self.hparams.get("num_hidden_layers"))
        head_count = self.hparams["num_attention_heads"]
        head_count_kv = head_count
        hidden_size = self.hparams["hidden_size"]

        self.gguf_writer.add_name('persimmon-8b-chat')
        self.gguf_writer.add_context_length(self.hparams["max_position_embeddings"])
        self.gguf_writer.add_embedding_length(hidden_size)
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_feed_forward_length(self.hparams["intermediate_size"])

        # NOTE: not sure about this change - why does the model not have a rope dimension count when it is smaller
        #       than the head size?
        #       ref: https://github.com/ggerganov/llama.cpp/pull/4889
        # self.gguf_writer.add_rope_dimension_count(hidden_size // head_count)
        self.gguf_writer.add_rope_dimension_count(hidden_size // head_count // 2)

        self.gguf_writer.add_head_count(head_count)
        self.gguf_writer.add_head_count_kv(head_count_kv)
        self.gguf_writer.add_rope_freq_base(self.hparams["rope_theta"])
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_eps"])
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["rms_norm_eps"])

    def set_vocab(self):
        self._set_vocab_sentencepiece()
        # self.gguf_writer.add_bos_token_id(71013)
        # self.gguf_writer.add_eos_token_id(71013)

    def write_tensors(self):
        block_count = self.hparams.get("num_layers", self.hparams.get("num_hidden_layers"))
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)

        for name, data_torch in self.get_tensors():
            if name.endswith(".self_attention.rotary_emb.inv_freq"):
                continue
            old_dtype = data_torch.dtype
            # TODO: FP16 conversion produces garbage outputs. (Q8_0 does not, so..?)
            data = data_torch.to(torch.float32).squeeze().numpy()
            new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                print(f"Can not map tensor {name!r}")
                sys.exit()
            n_dims = len(data.shape)
            print(f"{name} -> {new_name}, n_dims = {n_dims}, data.shape = {data.shape}, {old_dtype} --> {data.dtype}")
            self.gguf_writer.add_tensor(new_name, data)


class StableLMModel(Model):
    def set_vocab(self):
        if (self.dir_model / "tokenizer.json").is_file():
            self._set_vocab_gpt2()
        else:
            # StableLM 2 1.6B uses a vocab in a similar format to Qwen's vocab
            self._set_vocab_qwen()

    def set_gguf_parameters(self):
        hparams = self.hparams
        block_count = hparams["num_hidden_layers"]

        self.gguf_writer.add_name(self.dir_model.name)
        self.gguf_writer.add_context_length(hparams["max_position_embeddings"])
        self.gguf_writer.add_embedding_length(hparams["hidden_size"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_feed_forward_length(hparams["intermediate_size"])
        self.gguf_writer.add_rope_dimension_count(
            int(hparams["rope_pct"] * (hparams["hidden_size"] // hparams["num_attention_heads"])))
        self.gguf_writer.add_head_count(hparams["num_attention_heads"])
        self.gguf_writer.add_parallel_residual(hparams["use_parallel_residual"] if "use_parallel_residual" in
                                               hparams else True)
        self.gguf_writer.add_layer_norm_eps(1e-5)


class MixtralModel(Model):
    def set_vocab(self):
        self._set_vocab_sentencepiece()


class MiniCPMModel(Model):
    def set_gguf_parameters(self):
        block_count = self.hparams["num_hidden_layers"]
        self.gguf_writer.add_name("MiniCPM")
        self.gguf_writer.add_context_length(self.hparams["max_position_embeddings"])
        self.gguf_writer.add_embedding_length(self.hparams["hidden_size"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_feed_forward_length(self.hparams["intermediate_size"])
        self.gguf_writer.add_rope_dimension_count(self.hparams["hidden_size"] // self.hparams["num_attention_heads"])
        self.gguf_writer.add_head_count(self.hparams["num_attention_heads"])
        self.gguf_writer.add_head_count_kv(self.hparams["num_key_value_heads"])
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["rms_norm_eps"])
        self.gguf_writer.add_file_type(self.ftype)

    def set_vocab(self):
        self._set_vocab_hf()

    def _reverse_hf_permute(self, weights: Tensor, n_head: int, n_kv_head: int | None = None) -> Tensor:
        if n_kv_head is not None and n_head != n_kv_head:
            n_head //= n_kv_head

        return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2,
                                *weights.shape[1:]).swapaxes(1, 2).reshape(weights.shape))

    def write_tensors(self):
        block_count = self.hparams.get("n_layers", self.hparams.get("num_hidden_layers", self.hparams.get("n_layer")))
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)
        n_head = self.hparams.get("num_attention_heads")
        n_kv_head = self.hparams.get("num_key_value_heads")
        for name, data_torch in self.get_tensors():
            # we don't need these
            if name.endswith((".attention.masked_bias", ".attention.bias", ".attention.rotary_emb.inv_freq")):
                continue

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            # HF models permute some of the tensors, so we need to undo that
            if name.endswith(("q_proj.weight")):
                data_torch = self._reverse_hf_permute(data_torch, n_head, n_head)
            if name.endswith(("k_proj.weight")):
                data_torch = self._reverse_hf_permute(data_torch, n_head, n_kv_head)

            data = data_torch.squeeze().numpy()

            # map tensor names
            new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                print(f"Can not map tensor {name!r}")
                sys.exit()

            n_dims = len(data.shape)
            data_dtype = data.dtype

            # if f32 desired, convert any float16 to float32
            if self.ftype == 0 and data_dtype == np.float16:
                data = data.astype(np.float32)

            # TODO: Why can't we use these float16 as-is? There should be not reason to store float16 as float32
            if self.ftype == 1 and data_dtype == np.float16 and n_dims == 1:
                data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if self.ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
                data = data.astype(np.float16)

            print(f"{name} -> {new_name}, n_dims = {n_dims}, data.shape = {data.shape}, {old_dtype} --> {data.dtype}")

            self.gguf_writer.add_tensor(new_name, data)


class QwenModel(Model):
    @staticmethod
    def token_bytes_to_string(b):
        from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
        byte_encoder = bytes_to_unicode()
        return ''.join([byte_encoder[ord(char)] for char in b.decode('latin-1')])

    @staticmethod
    def bpe(mergeable_ranks: dict[bytes, int], token: bytes, max_rank: int | None = None) -> list[bytes]:
        parts = [bytes([b]) for b in token]
        while True:
            min_idx = None
            min_rank = None
            for i, pair in enumerate(zip(parts[:-1], parts[1:])):
                rank = mergeable_ranks.get(pair[0] + pair[1])
                if rank is not None and (min_rank is None or rank < min_rank):
                    min_idx = i
                    min_rank = rank
            if min_rank is None or (max_rank is not None and min_rank >= max_rank):
                break
            assert min_idx is not None
            parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
        return parts

    def set_vocab(self):
        self._set_vocab_qwen()

    def set_gguf_parameters(self):
        self.gguf_writer.add_name("Qwen")
        self.gguf_writer.add_context_length(self.hparams["max_position_embeddings"])
        self.gguf_writer.add_block_count(self.hparams["num_hidden_layers"])
        self.gguf_writer.add_embedding_length(self.hparams["hidden_size"])
        self.gguf_writer.add_feed_forward_length(self.hparams["intermediate_size"])
        self.gguf_writer.add_rope_freq_base(self.hparams["rotary_emb_base"])
        self.gguf_writer.add_rope_dimension_count(self.hparams["hidden_size"] // self.hparams["num_attention_heads"])
        self.gguf_writer.add_head_count(self.hparams["num_attention_heads"])
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["layer_norm_epsilon"])

    def write_tensors(self):
        block_count = self.hparams["num_hidden_layers"]
        model_kv = dict(self.get_tensors())
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)
        for name, data_torch in model_kv.items():
            # we don't need these
            if name.endswith(".rotary_emb.inv_freq"):
                continue

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            data = data_torch.squeeze().numpy()

            # map tensor names
            new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                print(f"Can not map tensor {name!r}")
                sys.exit()

            n_dims = len(data.shape)
            data_dtype = data.dtype

            # if f32 desired, convert any float16 to float32
            if self.ftype == 0 and data_dtype == np.float16:
                data = data.astype(np.float32)

            # TODO: Why can't we use these float16 as-is? There should be not reason to store float16 as float32
            if self.ftype == 1 and data_dtype == np.float16 and n_dims == 1:
                data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if self.ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
                data = data.astype(np.float16)

            print(f"{name} -> {new_name}, n_dims = {n_dims}, data.shape = {data.shape}, {old_dtype} --> {data.dtype}")
            self.gguf_writer.add_tensor(new_name, data)


class GPT2Model(Model):
    def set_gguf_parameters(self):
        self.gguf_writer.add_name(self.dir_model.name)
        self.gguf_writer.add_block_count(self.hparams["n_layer"])
        self.gguf_writer.add_context_length(self.hparams["n_ctx"])
        self.gguf_writer.add_embedding_length(self.hparams["n_embd"])
        self.gguf_writer.add_feed_forward_length(4 * self.hparams["n_embd"])
        self.gguf_writer.add_head_count(self.hparams["n_head"])
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)

    def write_tensors(self):
        block_count = self.hparams.get("n_layers", self.hparams.get("num_hidden_layers", self.hparams.get("n_layer")))
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)

        for name, data_torch in self.get_tensors():
            # we don't need these
            if name.endswith((".attention.masked_bias", ".attention.bias", ".attention.rotary_emb.inv_freq",
                              ".attn.bias", ".attn.masked_bias")):
                continue

            if name.endswith((".c_attn.weight", ".c_proj.weight", ".c_fc.weight", ".c_proj.weight")):
                data_torch = data_torch.transpose(1, 0)

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            data = data_torch.squeeze().numpy()

            # map tensor names
            new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                print(f"Can not map tensor {name!r}")
                sys.exit()

            n_dims = len(data.shape)
            data_dtype = data.dtype

            # if f32 desired, convert any float16 to float32
            if self.ftype == 0 and data_dtype == np.float16:
                data = data.astype(np.float32)

            # TODO: Why can't we use these float16 as-is? There should be not reason to store float16 as float32
            if self.ftype == 1 and data_dtype == np.float16 and n_dims == 1:
                data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if self.ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
                data = data.astype(np.float16)

            print(f"{name} -> {new_name}, n_dims = {n_dims}, data.shape = {data.shape}, {old_dtype} --> {data.dtype}")

            self.gguf_writer.add_tensor(new_name, data)

            # note: GPT2 output is tied to (same as) wte in original model
            if new_name == "token_embd.weight":
                print(f"output.weight, n_dims = {n_dims}, {old_dtype} --> {data.dtype}")
                self.gguf_writer.add_tensor("output.weight", data)


class Phi2Model(Model):
    def set_gguf_parameters(self):
        block_count = self.find_hparam(["num_hidden_layers", "n_layer"])

        rot_pct = self.find_hparam(["partial_rotary_factor"])
        n_embd = self.find_hparam(["hidden_size", "n_embd"])
        n_head = self.find_hparam(["num_attention_heads", "n_head"])

        self.gguf_writer.add_name("Phi2")
        self.gguf_writer.add_context_length(self.find_hparam(["n_positions", "max_position_embeddings"]))

        self.gguf_writer.add_embedding_length(n_embd)
        self.gguf_writer.add_feed_forward_length(4 * n_embd)
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_head_count(n_head)
        self.gguf_writer.add_head_count_kv(n_head)
        self.gguf_writer.add_layer_norm_eps(self.find_hparam(["layer_norm_epsilon", "layer_norm_eps"]))
        self.gguf_writer.add_rope_dimension_count(int(rot_pct * n_embd) // n_head)
        self.gguf_writer.add_file_type(self.ftype)
        self.gguf_writer.add_add_bos_token(False)


class PlamoModel(Model):
    def set_vocab(self):
        self._set_vocab_sentencepiece()

    def set_gguf_parameters(self):
        hparams = self.hparams
        block_count = hparams["num_hidden_layers"]

        self.gguf_writer.add_name("PLaMo")
        self.gguf_writer.add_context_length(4096)  # not in config.json
        self.gguf_writer.add_embedding_length(hparams["hidden_size"])
        self.gguf_writer.add_feed_forward_length(hparams["intermediate_size"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_head_count(hparams["num_attention_heads"])
        self.gguf_writer.add_head_count_kv(5)  # hparams["num_key_value_heads"]) is wrong
        self.gguf_writer.add_layer_norm_rms_eps(hparams["rms_norm_eps"])

    def shuffle_attn_q_weight(self, data_torch):
        assert data_torch.size() == (5120, 5120)
        data_torch = data_torch.reshape(8, 5, 128, 5120)
        data_torch = torch.permute(data_torch, (1, 0, 2, 3))
        data_torch = torch.reshape(data_torch, (5120, 5120))
        return data_torch

    def shuffle_attn_output_weight(self, data_torch):
        assert data_torch.size() == (5120, 5120)
        data_torch = data_torch.reshape(5120, 8, 5, 128)
        data_torch = torch.permute(data_torch, (0, 2, 1, 3))
        data_torch = torch.reshape(data_torch, (5120, 5120))
        return data_torch

    def write_tensors(self):
        block_count = self.hparams.get("num_layers", self.hparams.get("num_hidden_layers"))
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)

        for name, data_torch in self.get_tensors():
            if "self_attn.rotary_emb.inv_freq" in name:
                continue

            # map tensor names
            new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                print(f"Can not map tensor {name!r}")
                sys.exit()

            # shuffle for broadcasting of gqa in ggml_mul_mat
            if new_name.endswith("attn_q.weight"):
                data_torch = self.shuffle_attn_q_weight(data_torch)
            elif new_name.endswith("attn_output.weight"):
                data_torch = self.shuffle_attn_output_weight(data_torch)

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            data = data_torch.squeeze().numpy()

            n_dims = len(data.shape)
            data_dtype = data.dtype

            # if f32 desired, convert any float16 to float32
            if self.ftype == 0 and data_dtype == np.float16:
                data = data.astype(np.float32)

            # TODO: Why can't we use these float16 as-is? There should be not reason to store float16 as float32
            if self.ftype == 1 and data_dtype == np.float16 and n_dims == 1:
                data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if self.ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
                data = data.astype(np.float16)

            print(f"{name} -> {new_name}, n_dims = {n_dims}, data.shape = {data.shape}, {old_dtype} --> {data.dtype}")

            self.gguf_writer.add_tensor(new_name, data)


class CodeShellModel(Model):
    def set_gguf_parameters(self):
        block_count = self.hparams["n_layer"]

        self.gguf_writer.add_name("CodeShell")
        self.gguf_writer.add_context_length(self.hparams["n_positions"])
        self.gguf_writer.add_embedding_length(self.hparams["n_embd"])
        self.gguf_writer.add_feed_forward_length(4 * self.hparams["n_embd"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_head_count(self.hparams["n_head"])
        self.gguf_writer.add_head_count_kv(self.hparams["num_query_groups"])
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)
        self.gguf_writer.add_rope_freq_base(10000.0)
        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
        self.gguf_writer.add_rope_scaling_factor(1.0)

    def write_tensors(self):
        block_count = self.hparams.get("n_layers", self.hparams.get("num_hidden_layers", self.hparams.get("n_layer")))
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)
        tensors = dict(self.get_tensors())
        has_lm_head = "lm_head.weight" in tensors.keys() or "output.weight" in tensors.keys()
        for name, data_torch in tensors.items():
            # we don't need these
            if name.endswith((".attn.rotary_emb.inv_freq")):
                continue

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            data = data_torch.squeeze().numpy()

            # map tensor names
            new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                print(f"Can not map tensor {name!r}")
                sys.exit()

            n_dims = len(data.shape)
            data_dtype = data.dtype

            # if f32 desired, convert any float16 to float32
            if self.ftype == 0 and data_dtype == np.float16:
                data = data.astype(np.float32)

            # TODO: Why can't we use these float16 as-is? There should be not reason to store float16 as float32
            if self.ftype == 1 and data_dtype == np.float16 and n_dims == 1:
                data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if self.ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
                data = data.astype(np.float16)

            print(f"{name} -> {new_name}, n_dims = {n_dims}, data.shape = {data.shape}, {old_dtype} --> {data.dtype}")

            self.gguf_writer.add_tensor(new_name, data)

            if not has_lm_head and name == "transformer.wte.weight":
                self.gguf_writer.add_tensor("output.weight", data)
                print(name, f"=> output.weight, shape = {data.shape}, {old_dtype} --> {data.dtype}")


class InternLM2Model(Model):
    def set_vocab(self):
        # (TODO): Is there a better way?
        # Copy from _set_vocab_sentencepiece, The only difference is that we will treat the character
        # \x00 specially and convert it into an emoji character to prevent it from being mistakenly
        # recognized as an empty string in C++.
        from sentencepiece import SentencePieceProcessor
        from sentencepiece import sentencepiece_model_pb2 as model

        tokenizer_path = self.dir_model / 'tokenizer.model'

        tokens: list[bytes] = []
        scores: list[float] = []
        toktypes: list[int] = []

        if not tokenizer_path.is_file():
            print(f'Error: Missing {tokenizer_path}', file=sys.stderr)
            sys.exit(1)

        sentencepiece_model = model.ModelProto()
        sentencepiece_model.ParseFromString(open(tokenizer_path, "rb").read())
        add_prefix = sentencepiece_model.normalizer_spec.add_dummy_prefix

        tokenizer = SentencePieceProcessor(str(tokenizer_path))
        vocab_size = self.hparams.get('vocab_size', tokenizer.vocab_size())

        for token_id in range(vocab_size):
            piece = tokenizer.id_to_piece(token_id)
            text = piece.encode("utf-8")
            score = tokenizer.get_score(token_id)
            if text == b"\x00":
                # (TODO): fixme
                # Hack here and replace the \x00 characters.
                print(f"InternLM2 convert token '{text}' to '🐉'!")
                text = "🐉"

            toktype = SentencePieceTokenTypes.NORMAL
            if tokenizer.is_unknown(token_id):
                toktype = SentencePieceTokenTypes.UNKNOWN
            elif tokenizer.is_control(token_id):
                toktype = SentencePieceTokenTypes.CONTROL
            elif tokenizer.is_unused(token_id):
                toktype = SentencePieceTokenTypes.UNUSED
            elif tokenizer.is_byte(token_id):
                toktype = SentencePieceTokenTypes.BYTE

            tokens.append(text)
            scores.append(score)
            toktypes.append(toktype)

        added_tokens_file = self.dir_model / 'added_tokens.json'
        if added_tokens_file.is_file():
            with open(added_tokens_file, "r", encoding="utf-8") as f:
                added_tokens_json = json.load(f)

                for key in added_tokens_json:
                    tokens.append(key.encode("utf-8"))
                    scores.append(-1000.0)
                    toktypes.append(SentencePieceTokenTypes.USER_DEFINED)

        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)
        self.gguf_writer.add_add_space_prefix(add_prefix)

        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        old_eos = special_vocab.special_token_ids["eos"]
        if "chat" in os.path.basename(self.dir_model.absolute()):
            # For the chat model, we replace the eos with '<|im_end|>'.
            special_vocab.special_token_ids["eos"] = self._try_get_sft_eos(tokenizer)
            print(f"Replace eos:{old_eos} with a special token:{special_vocab.special_token_ids['eos']} \
in chat mode so that the conversation can end normally.")

        special_vocab.add_to_gguf(self.gguf_writer)

    def _try_get_sft_eos(self, tokenizer):
        unused_145_list = tokenizer.encode('[UNUSED_TOKEN_145]')
        im_end_list = tokenizer.encode('<|im_end|>')
        assert (len(unused_145_list) == 1) ^ (len(im_end_list) == 1)
        if len(unused_145_list) == 1:
            eos_token = unused_145_list[0]
        if len(im_end_list) == 1:
            eos_token = im_end_list[0]
        return eos_token

    def _hf_permute_qk(self, weights, n_head: int, n_head_kv: int):
        if n_head_kv is not None and n_head != n_head_kv:
            n_head = n_head_kv
        return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2,
                                *weights.shape[1:]).swapaxes(1, 2).reshape(weights.shape))

    def set_gguf_parameters(self):
        self.gguf_writer.add_name("InternLM2")
        self.gguf_writer.add_context_length(self.hparams["max_position_embeddings"])
        self.gguf_writer.add_block_count(self.hparams["num_hidden_layers"])
        self.gguf_writer.add_embedding_length(self.hparams["hidden_size"])
        self.gguf_writer.add_feed_forward_length(self.hparams["intermediate_size"])
        self.gguf_writer.add_rope_freq_base(self.hparams["rope_theta"])
        self.gguf_writer.add_head_count(self.hparams["num_attention_heads"])
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["rms_norm_eps"])
        self.gguf_writer.add_head_count_kv(self.hparams["num_key_value_heads"])

    def post_write_tensors(self, tensor_map, name, data_torch):
        old_dtype = data_torch.dtype

        # convert any unsupported data types to float32
        if data_torch.dtype not in (torch.float16, torch.float32):
            data_torch = data_torch.to(torch.float32)

        data = data_torch.squeeze().numpy()

        # map tensor names
        new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
        if new_name is None:
            print(f"Can not map tensor {name!r}")
            sys.exit()

        n_dims = len(data.shape)
        data_dtype = data.dtype

        # if f32 desired, convert any float16 to float32
        if self.ftype == 0 and data_dtype == np.float16:
            data = data.astype(np.float32)

        # TODO: Why can't we use these float16 as-is? There should be not reason to store float16 as float32
        if self.ftype == 1 and data_dtype == np.float16 and n_dims == 1:
            data = data.astype(np.float32)

        # if f16 desired, convert any float32 2-dim weight tensors to float16
        if self.ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
            data = data.astype(np.float16)

        print(f"{name} -> {new_name}, n_dims = {n_dims}, data.shape = {data.shape}, {old_dtype} --> {data.dtype}")
        self.gguf_writer.add_tensor(new_name, data)

    def write_tensors(self):
        from einops import rearrange

        num_heads = self.hparams.get("num_attention_heads")
        num_kv_heads = self.hparams.get("num_key_value_heads")
        hidden_size = self.hparams.get("hidden_size")
        q_per_kv = num_heads // num_kv_heads
        head_dim = hidden_size // num_heads
        num_groups = num_heads // q_per_kv

        block_count = self.hparams["num_hidden_layers"]
        model_kv = dict(self.get_tensors())
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)
        qkv_pattern = r"model\.layers\.(\d+)\.attention\.wqkv"
        for name, data_torch in model_kv.items():
            # we don't need these
            if name.endswith(".rotary_emb.inv_freq"):
                continue

            if re.match(qkv_pattern, name):
                bid = re.findall(qkv_pattern, name)[0]
                qkv = data_torch
                qkv = rearrange(qkv.T, " o (g n i) ->o g n i", g=num_groups, n=q_per_kv + 2, i=head_dim)
                q, k, v = qkv[..., :q_per_kv, :], qkv[..., q_per_kv:q_per_kv + 1, :], qkv[...,
                                                                                          q_per_kv + 1:q_per_kv + 2, :]
                # The model weights of q and k require additional reshape.
                q = self._hf_permute_qk(rearrange(q, " o g n i ->  o (g n i)").T, num_heads, num_heads)
                k = self._hf_permute_qk(rearrange(k, " o g n i ->  o (g n i)").T, num_heads, num_kv_heads)
                v = rearrange(v, " o g n i ->  o (g n i)").T
                self.post_write_tensors(tensor_map, f"model.layers.{bid}.attention.wq.weight", q)
                self.post_write_tensors(tensor_map, f"model.layers.{bid}.attention.wk.weight", k)
                self.post_write_tensors(tensor_map, f"model.layers.{bid}.attention.wv.weight", v)
            else:
                self.post_write_tensors(tensor_map, name, data_torch)


class BertModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_size = None

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_causal_attention(False)

        # get pooling path
        with open(self.dir_model / "modules.json", encoding="utf-8") as f:
            modules = json.load(f)
        pooling_path = None
        for mod in modules:
            if mod["type"] == "sentence_transformers.models.Pooling":
                pooling_path = mod["path"]
                break

        # get pooling type
        pooling_type = gguf.PoolingType.NONE
        if pooling_path is not None:
            with open(self.dir_model / pooling_path / "config.json", encoding="utf-8") as f:
                pooling = json.load(f)
            if pooling["pooling_mode_mean_tokens"]:
                pooling_type = gguf.PoolingType.MEAN
            elif pooling["pooling_mode_cls_token"]:
                pooling_type = gguf.PoolingType.CLS
            else:
                raise NotImplementedError("Only MEAN and CLS pooling types supported")

        self.gguf_writer.add_pooling_type(pooling_type.value)

    def set_vocab(self):
        path = self.dir_model
        added_tokens_path = self.dir_model if self.dir_model.exists() else None

        # use huggingface vocab to get all tokens
        vocab = HfVocab(path, added_tokens_path)
        tokens, scores, toktypes = zip(*vocab.all_tokens())
        assert len(tokens) == vocab.vocab_size
        self.vocab_size = vocab.vocab_size

        # we need this to validate the size of the token_type embeddings
        # though currently we are passing all zeros to the token_type embeddings
        n_token_types = len(set(toktypes))
        self.gguf_writer.add_token_type_count(n_token_types)

        # convert to phantom space vocab
        def phantom(tok, typ):
            if tok.startswith(b"[") and tok.endswith(b"]"):
                return tok
            if tok.startswith(b"##"):
                return tok[2:]
            return b"\xe2\x96\x81" + tok

        tokens = tuple(phantom(t, y) for t, y in zip(tokens, toktypes))

        # set up bos and eos tokens (cls and sep)
        self.gguf_writer.add_bos_token_id(vocab.tokenizer.cls_token_id)
        self.gguf_writer.add_eos_token_id(vocab.tokenizer.sep_token_id)

        # add vocab to gguf
        self.gguf_writer.add_tokenizer_model("bert")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        # handle special tokens
        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

    def write_tensors(self):
        tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)
        tensors = dict(self.get_tensors())
        for name, data_torch in tensors.items():
            # we are only using BERT for embeddings so we don't need the pooling layer
            if name in ("embeddings.position_ids", "pooler.dense.weight", "pooler.dense.bias"):
                continue  # we don't need these

            # map tensor names
            new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                print(f"Can not map tensor {name!r}")
                sys.exit()

            data = data_torch.squeeze().numpy()
            n_dims = len(data.shape)
            new_dtype: type[np.floating[Any]]

            if (self.ftype == 1 and name.endswith(".weight") and n_dims == 2
                    and name != "embeddings.token_type_embeddings.weight"  # not used with get_rows, must be F32
                ):
                # if f16 desired, convert any float32 2-dim weight tensors to float16
                new_dtype = np.float16
            else:
                # if f32 desired, convert any float16 to float32
                new_dtype = np.float32

            print(f"{new_name}, n_dims = {n_dims}, {data_torch.dtype} --> {new_dtype}")

            if data.dtype != new_dtype:
                data = data.astype(new_dtype)

            self.gguf_writer.add_tensor(new_name, data)


class NomicBertModel(BertModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # the HF config claims n_ctx=8192, but it uses RoPE scaling
        self.hparams["n_ctx"] = 2048

        # SwigLU activation
        assert self.hparams["activation_function"] == "swiglu"
        # this doesn't do anything in the HF version
        assert self.hparams["causal"] is False
        # no bias tensors
        assert self.hparams["qkv_proj_bias"] is False
        assert self.hparams["mlp_fc1_bias"] is False
        assert self.hparams["mlp_fc2_bias"] is False
        # norm at end of layer
        assert self.hparams["prenorm"] is False
        # standard RoPE
        assert self.hparams["rotary_emb_fraction"] == 1.0
        assert self.hparams["rotary_emb_interleaved"] is False
        assert self.hparams["rotary_emb_scale_base"] is None

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_rope_freq_base(self.hparams["rotary_emb_base"])

    def get_tensors(self):
        assert self.vocab_size is not None
        for name, data in super().get_tensors():
            # Nomic Embed's token embeddings tensor is padded, but llama.cpp wants tensor sizes to match exactly.
            if name == 'embeddings.word_embeddings.weight' and data.shape[1] != self.vocab_size:
                rounded_vocab_size = (self.vocab_size + 63) // 64 * 64
                assert data.shape == (rounded_vocab_size, self.hparams["n_embd"])
                data = data[:self.vocab_size, :]
            yield name, data


###### CONVERSION LOGIC ######


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a huggingface model to a GGML compatible file")
    parser.add_argument(
        "--vocab-only",
        action="store_true",
        help="extract only the vocab",
    )
    parser.add_argument("--awq-path", type=Path, default=None, help="Path to scale awq cache file")
    parser.add_argument(
        "--outfile",
        type=Path,
        help="path to write to; default: based on input",
    )
    parser.add_argument(
        "--outtype",
        type=str,
        choices=["f32", "f16"],
        default="f16",
        help="output format - use f32 for float32, f16 for float16",
    )
    parser.add_argument("--bigendian", action="store_true", help="model is executed on big endian machine")
    parser.add_argument(
        "model",
        type=Path,
        help="directory containing model file",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dir_model = args.model

    if args.awq_path:
        sys.path.insert(1, str(Path(__file__).parent / 'awq-py'))
        from awq.apply_awq import add_scale_weights  # type: ignore[import-not-found]
        tmp_model_path = args.model / "weighted_model"
        dir_model = tmp_model_path
        if tmp_model_path.is_dir():
            print(f"{tmp_model_path} exists as a weighted model.")
        else:
            tmp_model_path.mkdir(parents=True, exist_ok=True)
            print("Saving new weighted model ...")
            add_scale_weights(str(args.model), str(args.awq_path), str(tmp_model_path))
            print(f"Saved weighted model at {tmp_model_path}.")

    if not dir_model.is_dir():
        print(f'Error: {args.model} is not a directory', file=sys.stderr)
        sys.exit(1)

    ftype_map = {
        "f32": gguf.GGMLQuantizationType.F32,
        "f16": gguf.GGMLQuantizationType.F16,
    }

    if args.outfile is not None:
        fname_out = args.outfile
    else:
        # output in the same directory as the model by default
        fname_out = dir_model / f'ggml-model-{args.outtype}.gguf'

    print(f"Loading model: {dir_model.name}")

    hparams = Model.load_hparams(dir_model)

    with torch.inference_mode():
        model_class = Model.from_model_architecture(hparams["architectures"][0])
        model_instance = model_class(dir_model, ftype_map[args.outtype], fname_out, args.bigendian)

        print("Set model parameters")
        model_instance.set_gguf_parameters()

        print("Set model tokenizer")
        model_instance.set_vocab()

        if args.vocab_only:
            print(f"Exporting model vocab to '{fname_out}'")
            model_instance.write_vocab()
        else:
            print(f"Exporting model to '{fname_out}'")
            model_instance.write()

        print(f"Model successfully exported to '{fname_out}'")


if __name__ == '__main__':
    main()
