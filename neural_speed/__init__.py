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

model_maps = {"gpt_neox": "gptneox", "gpt_bigcode": "starcoder"}
max_request_num_default = 1


class Model:

    def __init__(self):
        self.module = None
        self.model = None
        self.model_type = None
        self.bin_file = None
        self.generate_round = 0
        self.max_request_num = -1

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
        elif model_type == "chatglm2" or model_type == "chatglm3":
            import neural_speed.chatglm2_cpp as cpp_model
        elif model_type == "baichuan":
            import neural_speed.baichuan_cpp as cpp_model
        elif model_type == "polyglot":
            import neural_speed.polyglot_cpp as cpp_model
        elif model_type == "qwen":
            import neural_speed.qwen_cpp as cpp_model
        elif model_type == "mistral":
            import neural_speed.mistral_cpp as cpp_model
        elif model_type == "qwen2":
            import neural_speed.qwen_cpp as cpp_model
        elif model_type == "phi":
            import neural_speed.phi_cpp as cpp_model
        elif model_type == "gemma":
            import neural_speed.gemma_cpp as cpp_model
        elif model_type == "stablelm":
            import neural_speed.stablelm_cpp as cpp_model
        elif model_type == "whisper":
            import neural_speed.whisper_cpp as cpp_model
        elif model_type == "mixtral":
            import neural_speed.mixtral_cpp as cpp_model
        else:
            raise TypeError("Unsupported model type {}!".format(model_type))
        self.module = cpp_model

    @staticmethod
    def get_model_type(model_config):
        model_type = model_maps.get(model_config.model_type, model_config.model_type)
        if model_type == "chatglm" and "chatglm2" in model_config._name_or_path:
            model_type = "chatglm2"

        # For ChatGLM3
        if model_type == "chatglm" and "chatglm3" in model_config._name_or_path:
            # due to the same model architecture.
            model_type = "chatglm2"

        # for TheBloke/falcon-40b-instruct-GPTQ & TheBloke/Falcon-7B-Instruct-GPTQ
        if model_type == "RefinedWebModel" or model_type == "RefinedWeb":
            model_type = "falcon"

        # for TheBloke/phi-2-GPTQ
        if model_type == "phi-msft":
            model_type = "phi"

        return model_type

    def init(self,
             model_name,
             use_quant=True,
             use_gptq=False,
             use_awq=False,
             use_autoround=False,
             weight_dtype="int4",
             alg="sym",
             group_size=32,
             scale_dtype="fp32",
             compute_dtype="int8",
             use_ggml=False,
             model_hub="huggingface"):
        if model_hub == "modelscope":
            from modelscope import AutoConfig
            self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        else:
            from transformers import AutoConfig
            self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model_type = Model.get_model_type(self.config)
        self.model_type = model_type
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
        if use_awq:
            quant_desc = "autoround"
        quant_bin = "{}/ne_{}_q_{}.bin".format(output_path, model_type, quant_desc)

        if not use_quant:
            self.bin_file = fp32_bin
        else:
            self.bin_file = quant_bin

        if os.path.exists(self.bin_file):
            print("{} existed, will use cache file. Otherwise please remove the file".format(self.bin_file))
            return

        if use_gptq or use_awq or use_autoround:
            convert_model(model_name, quant_bin, use_quantized_model=True)
            return

        if not os.path.exists(fp32_bin):
            convert_model(model_name, fp32_bin, "f32", model_hub=model_hub)
            assert os.path.exists(fp32_bin), "Fail to convert pytorch model"

        if not use_quant:
            print("FP32 model will be used.")
            return
        self.module.Model.quant_model(model_path=fp32_bin,
                                      out_path=quant_bin,
                                      weight_dtype=weight_dtype,
                                      alg=alg,
                                      group_size=group_size,
                                      scale_dtype=scale_dtype,
                                      compute_dtype=compute_dtype,
                                      use_ggml=use_ggml)
        assert os.path.exists(quant_bin), "Fail to quantize model"

        # clean
        os.remove(fp32_bin)

    def init_from_bin(self, model_type, model_path, **generate_kwargs):
        self.__import_package(model_type)
        self.model = self.module.Model()

        if self.max_request_num == -1:
            self.max_request_num = max(generate_kwargs.get("max_request_num", max_request_num_default),
                                       generate_kwargs.get("batch_size", 1))

        if "threads" not in generate_kwargs:
            threads = os.getenv("OMP_NUM_THREADS")
            import platform
            sys_platform = platform.platform().lower()
            if threads is None:
                if "windows" in sys_platform:
                    cpu_count = os.cpu_count()
                    generate_kwargs["threads"] = int(cpu_count)
                else:
                    generate_kwargs["threads"] = len(os.sched_getaffinity(0))
            else:
                generate_kwargs["threads"] = int(threads)

        # Setting scratch_size_ratio according to the ctx_size & tokens_length
        # If scratch_size_ratio has been set, will not enter this branch.
        if generate_kwargs.get("ctx_size") is not None and generate_kwargs.get(
                "ctx_size") > 2048 and generate_kwargs.get("scratch_size_ratio") is None:

            def get_max_seq_length():
                config = self.config.to_dict()
                # chatglm2, bloom, chatglm3
                if 'seq_length' in config:
                    return config['seq_length']
                # qwen2, llama-2, llama, dolly, gptneox, qwen, qwen1.5, opt, phi
                elif 'max_position_embeddings' in config:
                    return config['max_position_embeddings']
                # baichuan, baichuan2
                elif 'model_max_length' in config:
                    return config['model_max_length']
                # gptj
                elif 'n_positions' in config:
                    return config['n_positions']
                # mpt
                elif 'max_seq_len' in config:
                    return config['max_seq_len']
                # chatglm
                elif 'max_sequence_length' in config:
                    return config['max_sequence_length']
                # whisper
                elif 'max_length' in config:
                    return config['max_length']
                # Falcon does not have these parameters.
                elif model_type == "falcon":
                    return 2048
                else:
                    print("Not found max seq length, setting to default 512")
                    return 512

            # when tokens less than 10240
            def get_scratch_size_ratio(size):
                if size > 2048 and size <= 4096:
                    return 2
                elif size > 4096 and size <= 8192:
                    return 4
                elif size > 8192 and size <= 10240:
                    return 8
                else:
                    # more than 10240
                    return -1

            max_seq_length = get_max_seq_length()
            ctx_size = generate_kwargs.get("ctx_size")

            if ctx_size > max_seq_length:
                print(f'max_seq_length is {max_seq_length}, but ctx_size is {ctx_size}. Please reduce ctx_size.')
                exit(0)

            if max_seq_length > 2048 and max_seq_length <= 4096:
                generate_kwargs["scratch_size_ratio"] = 2
            elif max_seq_length > 4096 and max_seq_length <= 8192:
                generate_kwargs["scratch_size_ratio"] = 4
            elif max_seq_length > 8192:
                if get_scratch_size_ratio(ctx_size) != -1:
                    generate_kwargs["scratch_size_ratio"] = get_scratch_size_ratio(ctx_size)
                else:
                    if max_seq_length == 16384:
                        generate_kwargs["scratch_size_ratio"] = 12
                    elif max_seq_length == 32768:
                        if ctx_size < 20480:
                            generate_kwargs["scratch_size_ratio"] = 20
                        else:
                            generate_kwargs["scratch_size_ratio"] = 35

        self.model.init_model(model_path, **generate_kwargs)

    def quant_model(self, model_type, model_path, out_path, **quant_kwargs):
        self.__import_package(model_type)
        self.module.Model.quant_model(model_path=model_path, out_path=out_path, **quant_kwargs)

    def generate(self,
                 input_ids,
                 streamer=None,
                 interactive=False,
                 ignore_prompt=False,
                 stopping_criteria=None,
                 **generate_kwargs):
        batch_size = input_ids.shape[0]

        max_new_tokens = generate_kwargs.get("max_new_tokens", -1)
        max_request_num = generate_kwargs.pop("max_request_num", max_request_num_default)
        reinit_from_bin = False
        if max_request_num > self.max_request_num or batch_size > self.max_request_num:
            reinit_from_bin = True
            if self.max_request_num > 0:
                print("Will start to reinit model from bin due to different max request num.")
            self.max_request_num = max(batch_size, max_request_num)

        if self.model is None or reinit_from_bin:
            self.init_from_bin(self.model_type,
                               self.bin_file,
                               batch_size=batch_size,
                               max_request_num=self.max_request_num,
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
            assert input_ids.shape[0] == 1, "Unsupported multi-batch input ids."

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
        if input_ids.shape[0] > 1 and generate_kwargs.get("continuous_batching", True):
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
            elif ret[0][-1] == self.__get_eos_id() or \
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

    def __get_eos_id(self):
        return self.model.get_eos_id()

    def __call__(self, model_input, reinit=False, logits_all=False, **kwargs):
        if self.model_type == 'whisper':
            if self.model is None:
                self.model = self.module.Model()
                self.model.init_model(self.bin_file)
            if os.path.isfile(model_input):
                self.model.inference(model_input)
            else:
                print("Please input an audio file")
            return
        if isinstance(model_input, torch.Tensor):
            if self.model is None:
                self.init_from_bin(self.model_type, self.bin_file, **kwargs)
                self.generate_round = 0
            elif reinit:
                self.model.reinit()
                self.generate_round = 0
            return self.model.evaluate(model_input.tolist(), logits_all)
        else:
            print("Please input torch.Tensor")
        return

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
            del input_list[il][0:count]
            assert input_list[il] != [], "there are all pad tokens in batch {}.".format(il)
        return input_list
