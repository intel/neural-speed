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

import argparse
import sys
from pathlib import Path
from typing import List, Optional
from huggingface_hub import snapshot_download
from neural_speed.convert import convert_model

def main(args_in: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Convert a PyTorch model to a NE compatible file")
    parser.add_argument(
        "--outtype",
        choices=["f32", "f16"],
        help="output format, default: f32",
        default="f32",
    )
    parser.add_argument(
        "--token",
        type=str,
        help="Access token ID for models that require it (LLaMa2, etc..)",
    )
    parser.add_argument(
        "--outfile",
        type=Path,
        required=True,
        help="path to write to"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="NE",
        choices=["NE", "GGUF"],
        help="Convert to the GGUF or NE format"
    )
    parser.add_argument(
        "--use_quantized_model",
        action="store_true",
        help="use quantized model: awq/gptq/autoround"
    )
    parser.add_argument(
        "model",
        type=Path,
        help="directory containing model file or model id"
    )

    args = parser.parse_args(args_in)

    if args.model.exists():
        dir_model = args.model.as_posix()
    else:
        try:
            dir_model = snapshot_download(repo_id=str(args.model), resume_download=True, token=args.token)
        except Exception as e:
            if e.response.status_code == 401:
                print("You are required to input an access token ID for {}, please add it in option --token or download model weights locally".format(args.model))
            sys.exit(f"{e}")

    convert_model(dir_model, args.outfile, args.outtype, format=args.format, use_quantized_model=args.use_quantized_model)


if __name__ == "__main__":
    main()
