## Advanced Usage

### One-click scripts

Argument description of run.py ([supported MatMul combinations](#supported-matrix-multiplication-data-types-combinations)):
| Argument                    | Description                                                                                                   |
| --------------              | ---------------------------------------------------------------------                                         |
| model                       | Directory containing model file or model id: String                                                           |
| --weight_dtype              | Data type of quantized weight: int4/int8/fp8(=fp8_e4m3)/fp8_e5m2/fp4(=fp4e2m1)/nf4 (default int4)                                                       |
| --alg                       | Quantization algorithm: sym/asym (default sym)                                                                |
| --group_size                | Group size: Int, 32/128/-1 (per channel) (default: 32)                                                                                 |
| --scale_dtype               | Data type of scales: fp32/bf16/fp8 (default fp32)                                                                 |
| --compute_dtype             | Data type of Gemm computation: int8/bf16/fp16/fp32 (default: fp32)                                                 |
| --use_ggml                  | Enable ggml for quantization and inference                                                                    |
| -p / --prompt               | Prompt to start generation with: String (default: empty)                                                      |
| -f / --file                 | Path to a text file containing the prompt (for large prompts)                                                  |
| -n / --n_predict            | Number of tokens to predict: Int (default: -1, -1 = infinity)                                                 |
| -t / --threads              | Number of threads to use during computation: Int (default: 56)                                                |
| -b / --batch_size_truncate  | Batch size for prompt processing: Int (default: 512)                                                          |
| -c / --ctx_size             | Size of the prompt context: Int (default: 512, can not be larger than specific model's context window length) |
| -s / --seed                 | NG seed: Int (default: -1, use random seed for < 0)                                                           |
| --repeat_penalty            | Penalize repeat sequence of tokens: Float (default: 1.1, 1.0 = disabled)                                      |
| --color                     | Colorise output to distinguish prompt and user input from generations                                         |
| --keep                      | Number of tokens to keep from the initial prompt: Int (default: 0, -1 = all)                                  |
| --shift-roped-k             | Use [ring-buffer](./docs/infinite_inference.md#shift-rope-k-and-ring-buffer) and thus do not re-computing after reaching ctx_size (default: False) |
| --token                     | Access token ID for models that require it (e.g: LLaMa2, etc..)                                                |


### 1. Conversion and Quantization
Neural Speed assumes the compatible model format as [llama.cpp](https://github.com/ggerganov/llama.cpp) and [ggml](https://github.com/ggerganov/ggml). You can also convert the model by following the below steps:

```bash

# convert the model directly use model id in Hugging Face. (recommended)
python scripts/convert.py --outtype f32 --outfile ne-f32.bin EleutherAI/gpt-j-6b

# or you can download fp32 model (e.g., LLAMA2) from Hugging Face at first, then convert the pytorch model to ggml format.
git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
python scripts/convert.py --outtype f32 --outfile ne-f32.bin model_path

# To convert model with PEFT(Parameter-Efficient Fine-Tuning) adapter, you need to merge the PEFT adapter into the model first, use below command to merge the PEFT adapter and save the merged model, afterwards you can use 'scripts/convert.py' just like above mentioned.
python scripts/load_peft_and_merge.py --model_name_or_path meta-llama/Llama-2-7b-hf --peft_name_or_path dfurman/llama-2-7b-instruct-peft --save_path ./Llama-2-7b-hf-instruct-peft

# quantize weights of fp32 ggml bin
# model_name: llama, llama2, mpt, falcon, gptj, starcoder, dolly
# optimized INT4 model with group size 128 (recommended)
python scripts/quantize.py --model_name llama2 --model_file ne-f32.bin --out_file ne-q4_j.bin --weight_dtype int4 --group_size 128 --compute_dtype int8

# Alternatively you could run ggml q4_0 format like following
python scripts/quantize.py --model_name llama2 --model_file ne-f32.bin --out_file ne-q4_0.bin --weight_dtype int4 --use_ggml
# optimized INT4 model with group size 32
python scripts/quantize.py --model_name llama2 --model_file ne-f32.bin --out_file ne-q4_j.bin --weight_dtype int4 --group_size 32 --compute_dtype int8

```
Argument description of quantize.py ([supported MatMul combinations](#supported-matrix-multiplication-data-types-combinations)):
| Argument        | Description                                                  |
| --------------  | -----------------------------------------------------------  |
| --model_file    | Path to the fp32 model: String                               |
| --out_file      | Path to the quantized model: String                          |
| --build_dir     | Path to the build file: String                               |
| --config        | Path to the configuration file: String (default: "")         |
| --nthread       | Number of threads to use: Int (default: 1)                   |
| --weight_dtype  | Data type of quantized weight: int4/int8/fp8(=fp8_e4m3)/fp8_e5m2/fp4(=fp4_e2m1)/nf4 (default: int4)     |
| --alg           | Quantization algorithm to use: sym/asym (default: sym)       |
| --group_size    | Group size: Int 32/128/-1 (per channel) (default: 32)                                |
| --scale_dtype   | Data type of scales: bf16/fp32/fp8 (default: fp32)               |
| --compute_dtype | Data type of Gemm computation: int8/bf16/fp16/fp32 (default: fp32)|
| --use_ggml      | Enable ggml for quantization and inference                   |

#### Supported Matrix Multiplication Data Types Combinations

Our Neural Speed supports  INT4 / INT8 / FP8 (E4M3, E5M2) / FP4 (E2M1) / NF4 weight-only quantization and FP32 / FP16 / BF16 / INT8 computation forward matmul on the Intel platforms. Here are the all supported data types combinations for matmul operations (quantization and forward).
> This table will be updated frequently due to active development. For details you can refer to [BesTLA](../bestla#weight-only)

| Weight dtype | Compute dtype (default value) | Scale dtype (default value) | Quantization scheme (default value) |
|---|:---:|:---:|:---:|
| FP32 | FP32 | NA | NA |
| INT8 | INT8 / BF16 / FP16 / FP32 (FP32) | BF16 / FP32 (FP32) | sym / asym (sym) |
| INT4 | INT8 / BF16 / FP16 / FP32 (FP32) | BF16 / FP32 (FP32) | sym / asym (sym) |
| FP8 (E4M3, E5M2) | BF16 / FP16 / FP32 (FP32) | FP8 (FP8) | sym (sym) |
| FP4 (E2M1) | BF16 / FP16 / FP32 (FP32) | BF16 / FP32 (FP32) | sym (sym) |
| NF4 | BF16 / FP16 / FP32 (FP32) | BF16 / FP32 (FP32) | sym (sym) |


### 2. Inference

```bash
# recommend to use numactl to bind cores in Intel cpus for better performance
# if you use different core numbers, please also  change -t arg value
# please type prompt about codes when run `StarCoder`, for example, -p "def fibonnaci(".

#Linux and WSL
numactl -m 0 -C 0-<physic_cores-1> python scripts/inference.py --model_name llama -m ne-q4_j.bin -c 512 -b 1024 -n 256 -t <physic_cores> --color -p "She opened the door and see"

# if you want to generate fixed outputs, please set --seed arg, for example:
numactl -m 0 -C 0-<physic_cores-1> python scripts/inference.py --model_name llama -m ne-q4_j.bin -c 512 -b 1024 -n 256 -t <physic_cores> --color -p "She opened the door and see" --seed 12

# if you want to reduce repeated generated texts, please set --repeat_penalty (value > 1.0, default = 1.0), for example:
numactl -m 0 -C 0-<physic_cores-1> python scripts/inference.py --model_name llama -m ne-q4_j.bin -c 512 -b 1024 -n 256 -t <physic_cores> --color -p "She opened the door and see" --repeat_penalty 1.2

#Windows
#Recommend to build and run our project in WSL to get a better and stable performance
python scripts/inference.py --model_name llama -m ne-q4_j.bin -c 512 -b 1024 -n 256 -t <physic_cores|P-cores> --color -p "She opened the door and see"
```

Argument description of inference.py:
| Argument                                          | Description                                                                                                                                                                             |
| --------------                                    | -----------------------------------------------------------------------                                                                                                                 |
| --model_name                                      | Model name: String                                                                                                                                                                      |
| -m / --model                                      | Path to the executed model: String                                                                                                                                                      |
| --build_dir                                       | Path to the build file: String                                                                                                                                                          |
| -p / --prompt                                     | Prompt to start generation with: String (default: empty)                                                                                                                                |
| -f / --file                                       | Path to a text file containing the prompt (for large prompts)                                                                                                                            |
| -n / --n_predict                                  | Number of tokens to predict: Int (default: -1, -1 = infinity)                                                                                                                           |
| -t / --threads                                    | Number of threads to use during computation: Int (default: 56)                                                                                                                          |
| -b / --batch_size                                 | Batch size for prompt processing: Int (default: 512)                                                                                                                                    |
| -c / --ctx_size                                   | Size of the prompt context: Int (default: 512, can not be larger than specific model's context window length)                                                                           |
| -s / --seed                                       | NG seed: Int (default: -1, use random seed for < 0)                                                                                                                                     |
| --repeat_penalty                                  | Penalize repeat sequence of tokens: Float (default: 1.1, 1.0 = disabled)                                                                                                                |
| --color                                           | Colorise output to distinguish prompt and user input from generations                                                                                                                   |
| --keep                                            | Number of tokens to keep from the initial prompt: Int (default: 0, -1 = all)                                                                                                            |
| --shift-roped-k                                   | Use [ring-buffer](./docs/infinite_inference.md#shift-rope-k-and-ring-buffer) and thus do not re-computing after reaching ctx_size (default: False)                                      |
| --glm_tokenizer                                   | The path of the chatglm tokenizer: String (default: THUDM/chatglm-6b)                                                                                                                   |
| --memory-f32 <br> --memory-f16 <br> --memory-auto | Data type of kv memory (default to auto);<br>If set to auto, the runtime will try with bestla flash attn managed format (currently requires GCC11+ & AMX) and fall back to fp16 if failed |
