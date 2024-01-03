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
import os

import torch
from neural_speed.convert import convert_model
from transformers import AutoConfig, AutoTokenizer

model_maps = {"gpt_neox": "gptneox", "gpt_bigcode": "starcoder"}


class Model:
    def __init__(self):
        self.module = None
        self.model = None
        self.model_type = None
        self.bin_file = None
        self.generate_round = 0

    def __import_package(self, model_type):
        if self.module:
            return
        if model_type == "gptj":
            import neural_speed.gptj_cpp as cpp_model
        elif model_type == "falcon":
            import neural_speed.falcon_cpp as cpp_model
        elif model_type == "gptneox":
            import neural_speed.gptneox_cpp as cpp_model
        elif model_type == "dolly":
            import neural_speed.dolly_cpp as cpp_model
        elif model_type == "llama" or model_type == "llama2":
            import neural_speed.llama_cpp as cpp_model
        elif model_type == "mpt":
            import neural_speed.mpt_cpp as cpp_model
        elif model_type == "gpt_bigcode" or model_type == "starcoder":
            import neural_speed.starcoder_cpp as cpp_model
        elif model_type == "opt":
            import neural_speed.opt_cpp as cpp_model
        elif model_type == "bloom":
            import neural_speed.bloom_cpp as cpp_model
        elif model_type == "chatglm":
            import neural_speed.chatglm_cpp as cpp_model
        elif model_type == "chatglm2":
            import neural_speed.chatglm2_cpp as cpp_model
        elif model_type == "baichuan":
            import neural_speed.baichuan_cpp as cpp_model
        elif model_type == "polyglot":
            import neural_speed.polyglot_cpp as cpp_model
        elif model_type == "mistral":
            import neural_speed.mistral_cpp as cpp_model
        else:
            raise TypeError("Unspported model type {}!".format(model_type))
        self.module = cpp_model

    @staticmethod
    def get_model_type(model_config):
        model_type = model_maps.get(model_config.model_type, model_config.model_type)
        if model_type == "chatglm" and "chatglm2" in model_config._name_or_path:
            model_type = "chatglm2"
        return model_type

    def init(self, model_name, not_quant=False, use_cache=False, use_gptq=False, use_awq=False,
            weight_dtype="int4", alg="sym", group_size=32,
            scale_dtype="fp32", compute_dtype="int8", use_ggml=False):
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model_type = Model.get_model_type(self.config)
        self.__import_package(model_type)

        # check cache and quantization
        output_path = "runtime_outs"
        os.makedirs(output_path, exist_ok=True)
        fp32_bin = "{}/ne_{}_f32.bin".format(output_path, model_type)
        quant_desc = weight_dtype
        if use_ggml:
            quant_desc += "_ggml"
        else:
            quant_desc += "_bestla_c" + compute_dtype
            if group_size == -1:
                quant_desc += "_pc"
            else:
                quant_desc += "_g{}".format(group_size)
        if use_gptq:
            quant_desc = "gptq"
        if use_awq:
            quant_desc = "awq"
        quant_bin = "{}/ne_{}_q_{}.bin".format(output_path, model_type, quant_desc)

        if not_quant:
            self.bin_file = fp32_bin
        else:
            self.bin_file = quant_bin
        if use_cache and os.path.exists(self.bin_file):
            return

        if use_gptq or use_awq:
            convert_model(model_name, quant_bin, "f32")
            return
        
        if not use_cache or not os.path.exists(fp32_bin):
            convert_model(model_name, fp32_bin, "f32")
            assert os.path.exists(fp32_bin), "Fail to convert pytorch model"

        if not_quant:
            print("FP32 model will be used.")
            return
        self.module.Model.quant_model(model_path=fp32_bin, out_path=quant_bin,
                                    weight_dtype=weight_dtype, alg=alg, group_size=group_size,
                                    scale_dtype=scale_dtype, compute_dtype=compute_dtype, use_ggml=use_ggml)
        assert os.path.exists(quant_bin), "Fail to quantize model"

        # clean
        if not use_cache:
            os.remove(fp32_bin)

    def init_from_bin(self, model_type, model_path, **generate_kwargs):
        self.__import_package(model_type)
        self.model = self.module.Model()
        if "threads" not in generate_kwargs:
            threads = os.getenv("OMP_NUM_THREADS")
            if threads is None:
                generate_kwargs["threads"] = len(os.sched_getaffinity(0))
            else:
                generate_kwargs["threads"] = int(threads)
        self.model.init_model(model_path, **generate_kwargs)

    def quant_model(self, model_type, model_path, out_path, **quant_kwargs):
        self.__import_package(model_type)
        self.module.Model.quant_model(model_path=model_path, out_path=out_path, **quant_kwargs)

    def generate(self, input_ids, streamer=None, interactive=False, ignore_prompt=False,
                 stopping_criteria=None,  **generate_kwargs):
        max_new_tokens = generate_kwargs.get("max_new_tokens", -1)
        if self.model is None:
            self.init_from_bin(self.model_type, self.bin_file, batch_size=input_ids.shape[0],
                               **generate_kwargs)
            self.generate_round = 0
        elif not interactive:
            self.model.reinit()
            self.generate_round = 0

        ret = [[]]
        if self.generate_round == 0 and not ignore_prompt:
            ret = input_ids.tolist()

        beam_search = False
        if (generate_kwargs.get("num_beams", 1) > 1) and not generate_kwargs.get("do_sample", False):
            beam_search = True
        if not beam_search:
            # TODO support multi batch
            assert input_ids.shape[0] == 1, "Unsupport multi-batch input ids."

        if streamer:
            assert input_ids.shape[0] == 1, "Streamer only supports batch size 1."
            assert beam_search == False, "ERROR, can not use streamer when use beam search for generation! \
                Make sure that `num_beams` is set to 1."
            if self.generate_round == 0 and not ignore_prompt:
                streamer.put(input_ids)

        if interactive:
            self.model.reset_token_end()
        out_count = 0
        input_list = None
        pad_token_id = generate_kwargs.get("pad_token", None)
        if generate_kwargs.get("continuous_batching", False):
            input_list = self._cont_batching_input(input_ids, pad_token_id)
        else:
            input_list = input_ids.tolist()
        while True:
            response = self.model.generate(input_ids=input_list)
            input_list = []  # next-token stage will use previous output
            if len(response) == 0:
                break
            if streamer:
                streamer.put(torch.tensor([response[0]]))
            for i in range(len(response)):
                ret[i].extend(response[i])
            if beam_search:
                break
            if stopping_criteria is not None:
                if stopping_criteria(torch.tensor(ret), None):
                    break
            elif ret[0][-1] == self.tokenizer.eos_token_id or \
                    (max_new_tokens != -1 and out_count > max_new_tokens):
                break
            out_count += 1
        if streamer:
            streamer.end()

        self.generate_round += 1
        if os.getenv("NEURAL_SPEED_VERBOSE") and os.getenv("NEURAL_SPEED_VERBOSE") in ["1", "0"]:
            self.model.print_time()
        return ret

    def is_token_end(self):
        return self.model.is_token_end()

    def _cont_batching_input(self, input_ids, pad_token_id=None):
        assert isinstance(input_ids, torch.Tensor), "Input must be torch.Tensor."
        input_list = input_ids.tolist()
        pti = pad_token_id
        if pti == None:
            pti = self.tokenizer.pad_token_id
        assert pti != None, "Please supply pad token id."
        for il in range(len(input_list)):
            count = input_list[il].count(pti)
            # padding left
            del input_list[il][0: count]
            assert input_list[il] != [], "there are all pad tokens in batch {}.".format(il)
        return input_list

    def __call__(self, input_ids, reinit=False, **kwargs):
        if self.model is None:
            self.init_from_bin(self.model_type, self.bin_file, **kwargs)
            self.generate_round = 0
        elif reinit:
            self.model.reinit()
            self.generate_round = 0
        return self.model.evaluate(input_ids.tolist())
