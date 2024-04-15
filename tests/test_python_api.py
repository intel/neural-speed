#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
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

import numpy as np
import shutil
import torch
import unittest

from transformers import AutoTokenizer, TextStreamer
from neural_speed import Model

def cmpData(numa, numb):
    totalErr = ((np.abs(numa - numb))**2).sum()
    totalNum = (np.abs(numa)**2).sum()
    diff2 = np.sqrt(totalErr/totalNum)

    cos = np.dot(numa, numb)/(np.linalg.norm(numa)*np.linalg.norm(numb))
    return {"diff2": diff2, "cos": cos}

class TestLLMRUNTIME(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree("./runtime_outs", ignore_errors=True)

    def test_llm_runtime(self):
        model_name = "/tf_dataset2/models/nlp_toolkit/llama-2-7b-chat/Llama-2-7b-chat-hf"
        prompt = "What is the meaning of life?"

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        inputs = tokenizer(prompt, return_tensors="pt")

        pt_logits = torch.load("/tf_dataset2/inc-ut/nlptoolkit_ut_model/llama2_pt_logits.pth")[:,-1]
        pt_generate_ids = torch.load("/tf_dataset2/inc-ut/nlptoolkit_ut_model/llama2_pt_generate_ids.pth")[0].tolist()
        print(tokenizer.decode(pt_generate_ids))

        # check output ids
        woq_config_fp32 = {"use_quant":False, "compute_dtype":"fp32", "weight_dtype":"fp32", "use_ggml":False, "group_size":128}
        itrex_model = Model()

        itrex_model.init(model_name, use_quant=False)

        itrex_generate_ids = itrex_model.generate(inputs.input_ids, do_sample=False, max_new_tokens=100)[0]
        print(tokenizer.decode(itrex_generate_ids))
        for i in range(len(pt_generate_ids)):
            self.assertEqual(pt_generate_ids[i], itrex_generate_ids[i])

        # check diff of logits
        woq_configs = {
            "fp32": {"use_quant":False},
            # "ggml_int4": {"compute_dtype":"int8", "weight_dtype":"int4", "use_ggml":True},
            "jblas_int4": {"compute_dtype":"int8", "weight_dtype":"int4"},
            # "jblas_int8": {"compute_dtype":"bf16", "weight_dtype":"int8"},
        }
        for config_type in woq_configs:
            itrex_model = Model()
            itrex_model.init(model_name, **woq_configs[config_type])
            itrex_logits = itrex_model(inputs.input_ids)
            diff_data = cmpData(pt_logits.detach().numpy().flatten(), itrex_logits.flatten())
            print(config_type, diff_data)


    def test_multi_batch_inference(self):
        model_name = "/tf_dataset2/models/pytorch/gpt-j-6B"  # or local path to model
        prompts = [
           "she opened the door and see",
           "tell me 10 things about jazz music",
           "What is the meaning of life?",
           "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer"\
            " The slings and arrows of outrageous fortune, "\
            "Or to take arms against a sea of troubles."\
            "And by opposing end them. To die—to sleep,"
            ]

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,
                                                  padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        pad_token = tokenizer.pad_token_id
        inputs = tokenizer(prompts, padding=True, return_tensors='pt')

        # pytorch fp32
        pt_generate_ids = torch.load("/tf_dataset2/inc-ut/nlptoolkit_ut_model/beam_pt_generate_ids.pth").tolist()

        # llm runtime fp32 beam search
        itrex_model = Model()
        itrex_model.init(model_name, use_quant=False)
        itrex_generate_ids_padded = itrex_model.generate(
            inputs.input_ids, num_beams=4, max_new_tokens=128, min_new_tokens=30, early_stopping=True,
            pad_token=pad_token, continuous_batching=False)
        for i in range(len(itrex_generate_ids_padded)):
            self.assertListEqual(pt_generate_ids[i], itrex_generate_ids_padded[i])
        itrex_model.model = None
        itrex_generate_ids_cont = itrex_model.generate(
            inputs.input_ids, num_beams=4, max_new_tokens=128, min_new_tokens=30, early_stopping=True,
            pad_token=pad_token, continuous_batching=True)
        for i in range(len(itrex_generate_ids_cont)):
            self.assertListEqual(itrex_generate_ids_cont[i], itrex_generate_ids_cont[i])

        # llm runtime int4 greedy search
        itrex_model = Model()
        itrex_model.init(model_name, use_quant=True, weight_dtype="int4", compute_dtype="int8")
        outputs = itrex_model.generate(inputs.input_ids, num_beams=1, max_new_tokens=128, pad_token=pad_token,
                                       continuous_batching=True, memory_dtype="f16", do_sample=False)
        for i in range(len(prompts)):
            input_ids = tokenizer(prompts[i], return_tensors='pt').input_ids
            output = itrex_model.generate(input_ids, num_beams=1, max_new_tokens=128, pad_token=pad_token,
                                          memory_dtype="f16", do_sample=False)
            # ignore pad token
            gen_len = len(output[0]) - input_ids.shape[-1]
            self.assertListEqual(outputs[i][inputs.input_ids.shape[-1]: inputs.input_ids.shape[-1] + gen_len],
                                 output[0][input_ids.shape[-1]:])

if __name__ == "__main__":
    unittest.main()
