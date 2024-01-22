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

from pathlib import Path
from transformers import AutoConfig
import subprocess

model_maps = {"gpt_neox": "gptneox", "gpt_bigcode": "starcoder", "whisper": "whisper"}


def convert_model(model, outfile, outtype, whisper_repo_path=None):
    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    model_type = model_maps.get(config.model_type, config.model_type)

    quantized_model = 'gptq' in str(model).lower() or 'awq' in str(model).lower()
    if quantized_model:
        path = Path(Path(__file__).parent.absolute(), "convert_quantized_{}.py".format(model_type))
    else:
        path = Path(Path(__file__).parent.absolute(), "convert_{}.py".format(model_type))
    cmd = []
    cmd.extend(["python", path])
    cmd.extend(["--outfile", outfile])
    cmd.extend(["--outtype", outtype])
    cmd.extend([model])

    print("cmd:", cmd)
    subprocess.run(cmd)
