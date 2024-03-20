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
from transformers import AutoTokenizer, TextStreamer
from neural_speed import Model
from typing import List, Optional


def main(args_in: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="pythonAPI example for gptq")
    parser.add_argument("model", type=Path, help="directory containing model file")
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="Prompt to start generation with: String (default: empty)",
        default="Once upon a time, a little girl",
    )
    args = parser.parse_args(args_in)

    prompt = args.prompt
    model_name = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    streamer = TextStreamer(tokenizer)

    model = Model()
    # If you want to run AWQ models, just set use_awq = True.
    model.init(model_name, weight_dtype="int4", compute_dtype="int8", use_gptq=True)
    outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300, do_sample=True)


if __name__ == "__main__":
    main()
