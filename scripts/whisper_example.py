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

if len(sys.argv) != 3:
    print("Usage: python whisper_example.py model_path and audio_file")
model_name = sys.argv[1]
audio_file = sys.argv[2]

model = Model()
model.init(model_name, use_ggml=True)
model(audio_file)
