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
from transformers import AutoTokenizer, TextStreamer
from neural_speed import Model

# Usage:
# python python_api_example_for_bin.py \
# --model_name falcon \
# --model_path /model_path/falcon-7b \
# -m /model_path/falcon-7b/ggml-model-f32.gguf

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

    prompt = "Once upon a time"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    streamer = TextStreamer(tokenizer)

    model = Model()
    model.init_from_bin(args.model_name, gguf_path)
    outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300, do_sample=True)


if __name__ == "__main__":
    main()
