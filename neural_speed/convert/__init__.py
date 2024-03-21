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
import subprocess

model_maps = {
    "gpt_neox": "gptneox",
    "gpt_bigcode": "starcoder",
    "whisper": "whisper",
    "qwen2": "qwen",
    "RefinedWebModel": "falcon",
    "RefinedWeb": "falcon",
    "phi-msft": "phi"
}


def convert_model(model, outfile, outtype="f32", format="NE", model_hub="huggingface", use_quantized_model=False):
    if model_hub == "modelscope":
        from modelscope import AutoConfig
    else:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    model_type = model_maps.get(config.model_type, config.model_type)

    if use_quantized_model:
        path = Path(Path(__file__).parent.absolute(), "convert_quantized_{}.py".format(model_type))
    else:
        path = Path(Path(__file__).parent.absolute(), "convert_{}.py".format(model_type))
    cmd = []
    cmd.extend(["python", path])
    cmd.extend(["--outfile", outfile])
    cmd.extend(["--outtype", outtype])
    if model_type in {"phi", "stablelm"}:
        cmd.extend(["--format", format])
    cmd.extend(["--model_hub", model_hub])
    cmd.extend([model])

    print("cmd:", cmd)
    subprocess.run(cmd)
