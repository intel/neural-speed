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
from transformers import AutoTokenizer, TextStreamer
from neural_speed import Model

if len(sys.argv) != 2:
    print("Usage: python python_api_example.py model_path")
model_name = sys.argv[1]

prompt = "Once upon a time, a little girl"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)

model = Model()
model.init(model_name, weight_dtype="int4", compute_dtype="int8")
outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300, do_sample=True)
