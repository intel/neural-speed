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
import sys
import argparse
from pathlib import Path
from typing import List, Optional
from transformers import AutoTokenizer, TextStreamer, AutoModelForCausalLM
from neural_speed import Model

# Usage:
# python python_api_example_for_bin.py \
# --model_name falcon \
# --model_path /model_path/falcon-7b \
# -m /model_path/falcon-7b/ggml-model-f32.gguf

import torch
def build_chat_input(model, tokenizer, messages: List[dict], max_new_tokens: int=0):
    def _parse_messages(messages, split_role="user"):
        system, rounds = "", []
        round = []
        for i, message in enumerate(messages):
            if message["role"] == "system":
                assert i == 0
                system = message["content"]
                continue
            if message["role"] == split_role and round:
                rounds.append(round)
                round = []
            round.append(message)
        if round:
            rounds.append(round)
        return system, rounds

    max_new_tokens = max_new_tokens or model.generation_config.max_new_tokens
    max_input_tokens = model.config.model_max_length - max_new_tokens
    system, rounds = _parse_messages(messages, split_role="user")
    system_tokens = tokenizer.encode(system)
    max_history_tokens = max_input_tokens - len(system_tokens)

    history_tokens = []
    for round in rounds[::-1]:
        round_tokens = []
        for message in round:
            if message["role"] == "user":
                round_tokens.append(model.generation_config.user_token_id)
            else:
                round_tokens.append(model.generation_config.assistant_token_id)
            round_tokens.extend(tokenizer.encode(message["content"]))
        if len(history_tokens) == 0 or len(history_tokens) + len(round_tokens) <= max_history_tokens:
            history_tokens = round_tokens + history_tokens  # concat left
            if len(history_tokens) < max_history_tokens:
                continue
        break

    input_tokens = system_tokens + history_tokens
    if messages[-1]["role"] != "assistant":
        input_tokens.append(model.generation_config.assistant_token_id)
    input_tokens = input_tokens[-max_input_tokens:]  # truncate left
    return torch.LongTensor([input_tokens]).to(model.device)


def main(args_in: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="main program llm running")
    parser.add_argument("--model_name", type=str, help="Model name: String", required=True)
    parser.add_argument("--model_path", type=Path, help="Path to the model: String", required=True)
    parser.add_argument("-m", "--model", type=Path, help="Path to the executed model: String", required=True)
    parser.add_argument("--format",
                        type=str,
                        default="GGUF",
                        choices=["NE", "GGUF"],
                        help="convert to the GGUF or NE format")
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="Prompt to start generation with: String (default: empty)",
        default="Once upon a time",
    )

    args = parser.parse_args(args_in)
    print(args)

    gguf_path = args.model.as_posix()

    # prompt = args.prompt
    # tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    # inputs = tokenizer(prompt, return_tensors="pt").input_ids
    # import pdb;pdb.set_trace()
    # streamer = TextStreamer(tokenizer)

    # model = Model()
    # model.init_from_bin(args.model_name, gguf_path)
    # outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300, do_sample=True)


    # prompt = args.prompt
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    transformer_model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    # generation_config = GenerationConfig.from_pretrained(args.model_path)
    
    ## inputs = torch.tensor([[ 195, 16829, 92361,   196]])
    messages = []
    messages.append({"role": "user", "content": "你好”"})

    from transformers.generation.utils import GenerationConfig
    inputs_ids = build_chat_input(transformer_model, tokenizer, messages, max_new_tokens = 300)
    print(inputs_ids)
    model = Model()
    model.init_from_bin(args.model_name, gguf_path)
    outputs = model.generate(inputs_ids, max_new_tokens=300, do_sample=True)
    words = tokenizer.decode(outputs[0])
    print(words)
    
if __name__ == "__main__":
    main()
