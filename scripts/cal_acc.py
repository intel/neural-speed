#  Copyright (c) 2023 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import sys
import argparse
from ns_evaluator import LMEvalParser
from accuracy import cli_evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate accuracy for a model")
    parser.add_argument('--model_name', type=str, default="~/Llama-2-7b-chat-hf")
    parser.add_argument('--tasks', type=str, default="lambada_openai")
    parser.add_argument("--use_gptq", action="store_true")
    parser.add_argument("--use_awq", action="store_true")
    parser.add_argument("--use_autoround", action="store_true")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--weight_dtype', type=str, default="int4")
    parser.add_argument('--compute_dtype', type=str, default="int8")
    parser.add_argument('--group_size', type=int, default=32)
    parser.add_argument('--use_ggml', action="store_true")
    parser.add_argument('--alg', type=str, default="sym")
    parser.add_argument('--scale_dtype', type=str, default="fp32")
    args = parser.parse_args()

    model_args=f'pretrained={args.model_name},dtype=float32,trust_remote_code=True'
    # model_args += f'use_gptq={args.use_gptq},use_awq={args.use_awq},use_autoround={args.use_autoround}'
    eval_args = LMEvalParser(model="hf",
                        model_args=model_args,
                        tasks=f"{args.tasks}",
                        device="cpu",
                        batch_size=args.batch_size,
                        use_gptq=args.use_gptq,
                        use_autoround=args.use_autoround,
                        use_awq=args.use_awq,
                        weight_dtype=args.weight_dtype,
                        compute_dtype=args.compute_dtype,
                        group_size=args.group_size,
                        use_ggml=args.use_ggml,
                        alg=args.alg,
                        scale_dtype=args.scale_dtype,
                        )
    results = cli_evaluate(eval_args)

    print(results)
