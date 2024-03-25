# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import unittest
import shutil
from neural_speed import Model, ModelServer
from transformers import AutoTokenizer

class TestModelServer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree("./runtime_outs", ignore_errors=True)

    def test_model_server(self):
        prompts = [
                "she opened the door and see",
                "tell me 10 things about jazz music",
                "What is the meaning of life?",
                "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer"\
                " The slings and arrows of outrageous fortune, "\
                "Or to take arms against a sea of troubles."\
                "And by opposing end them. To die—to sleep,",
                "Tell me an interesting fact about llamas.",
                "What is the best way to cook a steak?",
                "Are you familiar with the Special Theory of Relativity and can you explain it to me?",
                "Recommend some interesting books to read.",
                "What is the best way to learn a new language?",
                "How to get a job at Intel?",
                "If you could have any superpower, what would it be?",
                "I want to learn how to play the piano.",
                ]
        model_name = "/tf_dataset2/models/nlp_toolkit/llama-2-7b-chat/Llama-2-7b-chat-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = Model()
        # get quantized model
        model.init(model_name, use_quant=True, weight_dtype="int4", compute_dtype="int8")
        print("=======REFERENCE RESULTS FOR COMPARISON=========", flush=True)
        print("=======FOR LOOP GREEDY SEARCH GENERATION RESULTS WITH MHA==========", flush=True)
        for i in range(len(prompts)):
            p_token_ids = tokenizer(prompts[i], return_tensors='pt').input_ids
            output = model.generate(p_token_ids, num_beams=1, max_new_tokens=128, do_sample=False)
            ans = tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            print(ans, flush=True)
            print("================================", flush=True)
        print("=======FOR LOOP BEAM SEARCH GENERATION RESULTS WITH MHA==========", flush=True)
        for i in range(len(prompts)):
            p_token_ids = tokenizer(prompts[i], return_tensors='pt').input_ids
            output = model.generate(p_token_ids, num_beams=4, max_new_tokens=128, min_new_tokens=30,
                                    early_stopping=True, do_sample=False, continuous_batching=True,
                                    max_request_num=4)
            ans = tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            print(ans, flush=True)
            print("================================", flush=True)
        del model
        model_path = "./runtime_outs/ne_llama_q_int4_bestla_cint8_g32.bin"

        res_collect = []
        # response function (deliver generation results and current remain working size in server)
        def f_response(res, working):
            ret_token_ids = [r.token_ids for r in res]
            res_collect.extend(ret_token_ids)
            ans = tokenizer.batch_decode(ret_token_ids, skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)
            print(f"working_size: {working}, ans:", flush=True)
            for a in ans:
                print(a, flush=True)
                print("=====================================", flush=True)

        log_map = {"auto": "MHA", "f16": "NON-MHA",
                   "greedy": "GREEDY SEARCH", "beam": "BEAM SEARCH"}
        for md in ["auto", "f16"]:
            for policy in ["greedy", "beam"]:
                print("============={} {} MODEL SERVER TESTING========".format(log_map[md],
                                                                        log_map[policy]), flush=True)
                added_count = 0
                s = ModelServer(model_name,
                                f_response,
                                model_path,
                                max_new_tokens=128,
                                num_beams=4 if policy == "beam" else 1,
                                min_new_tokens=30,
                                early_stopping=True,
                                do_sample=False,
                                continuous_batching=True,
                                return_prompt=True,
                                max_request_num=8,
                                threads=56,
                                print_log=False,
                                scratch_size_ratio = 1.0,
                                memory_dtype= md,
                                )
                for i in range(len(prompts)):
                    p_token_ids = tokenizer(prompts[i], return_tensors='pt').input_ids.tolist()
                    s.issueQuery(i, p_token_ids)
                    added_count += 1
                    time.sleep(2)  # adjust query sending time interval

                # recommend to use time.sleep in while loop to exit program
                # let cpp server owns more resources
                while (added_count != len(prompts) or not s.Empty()):
                    time.sleep(1)
                del s
                print("should finished", flush=True)

if __name__ == "__main__":
    unittest.main()
