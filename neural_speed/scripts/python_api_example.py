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

from transformers import AutoTokenizer, TextStreamer
from neural_speed import Model

model_name = "/mnt/disk1/data2/zhenweil/models/llama/Llama-2-7b-chat-hf"  # or local path to model
# int4 weight_only quantization
# woq_config = WeightOnlyQuantConfig(compute_dtype="int8", weight_dtype="int4", use_gptq=True)
# fp4 weight_only quantization
# woq_config = WeightOnlyQuantConfig(compute_dtype="fp32", weight_dtype="fp4")
# nf4 weight_only quantization
# woq_config = WeightOnlyQuantConfig(compute_dtype="fp32", weight_dtype="nf4")
# fp8 weight_only quantization
# woq_config = WeightOnlyQuantConfig(compute_dtype="fp32", weight_dtype="fp8")
# for more data types combinations, please refer to
# `Supported Matrix Multiplication Data Types Combinations` section in README
prompt = "Once upon a time, a little girl"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)

# top_k_top_p sample or greedy_search
model = Model()
# import pdb; pdb.set_trace()
model.init(model_name,
            weight_dtype="int4",
            alg="sym",
            group_size=32,
            scale_dtype="fp32",
            compute_dtype="int8",
            use_ggml=False)
outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300, do_sample=True)
# beam search
# model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=woq_config, trust_remote_code=True)
# outputs = model.generate(inputs, num_beams=4, max_new_tokens=128, min_new_tokens=30, early_stopping=True)
# ans = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
# print(ans)

# while True:
#     prompt = input("> ").strip()
#     if prompt == "quit":
#         break
#     b_prompt = "[INST]{}[/INST]".format(prompt)  # prompt template for llama2
#     inputs = tokenizer(b_prompt, return_tensors="pt").input_ids
#     outputs = model.generate(inputs, streamer=streamer, interactive=True, ignore_prompt=True, do_sample=True)