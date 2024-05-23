# Neural Speed

Neural Speed is an innovative library designed to support the efficient inference of large language models (LLMs) on Intel platforms through the state-of-the-art (SOTA) low-bit quantization powered by [Intel Neural Compressor](https://github.com/intel/neural-compressor). The work is inspired by [llama.cpp](https://github.com/ggerganov/llama.cpp) and further optimized for Intel platforms with our innovations in [NeurIPS' 2023](https://arxiv.org/abs/2311.00502)

## Key Features
- Highly optimized kernels on CPUs with ISAs (AMX, VNNI, AVX512F, AVX_VNNI and AVX2) for N-bit weight (int1, int2, int3, int4, int5, int6, int7 and int8). See [details](neural_speed/core/README.md)
- Up to 40x performance speedup on popular LLMs compared with llama.cpp. See [details](https://medium.com/@NeuralCompressor/llm-performance-of-intel-extension-for-transformers-f7d061556176) 
- Tensor parallelism across sockets/nodes on CPUs. See [details](./docs/tensor_parallelism.md)

> Neural Speed is under active development so APIs are subject to change.

## Supported Hardware
| Hardware | Supported |
|-------------|:-------------:|
|Intel Xeon Scalable Processors | ✔ |
|Intel Xeon CPU Max Series | ✔ |
|Intel Core Processors | ✔ |

## Supported Models
Support almost all the LLMs in PyTorch format from Hugging Face such as Llama2, ChatGLM2, Baichuan2, Qwen, Mistral, Whisper, etc. File an [issue](https://github.com/intel/neural-speed/issues) if your favorite LLM does not work.

Support typical LLMs in GGUF format such as Llama2, Falcon, MPT, Bloom etc. More are coming. Check out the [details](./docs/supported_models.md).

## Installation

### Install from binary
```shell
pip install -r requirements.txt
pip install neural-speed
```

### Build from Source
```shell
pip install .
```

>**Note**: GCC requires version 10+


## Quick Start (Transformer-like usage)

Install [Intel Extension for Transformers](https://github.com/intel/intel-extension-for-transformers/blob/main/docs/installation.md) to use Transformer-like APIs.


### PyTorch Model from Hugging Face

```python
from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM
model_name = "Intel/neural-chat-7b-v3-1"     # Hugging Face model_id or local model
prompt = "Once upon a time, there existed a little girl,"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)

model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True)
outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)
```

### GGUF Model from Hugging Face

```python
from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM

# Specify the GGUF repo on the Hugginface
model_name = "TheBloke/Llama-2-7B-Chat-GGUF"
# Download the the specific gguf model file from the above repo
gguf_file = "llama-2-7b-chat.Q4_0.gguf"
# make sure you are granted to access this model on the Huggingface.
tokenizer_name = "meta-llama/Llama-2-7b-chat-hf"

prompt = "Once upon a time"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)
model = AutoModelForCausalLM.from_pretrained(model_name, gguf_file = gguf_file)
outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)
```
### PyTorch Model from Modelscope
```python
from transformers import TextStreamer
from modelscope import AutoTokenizer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM
model_name = "qwen/Qwen-7B"     # Modelscope model_id or local model
prompt = "Once upon a time, there existed a little girl,"

model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, model_hub="modelscope")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)
outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)
```

### As an Inference Backend in Neural Chat Server
`Neural Speed` can be used in [Neural Chat Server](https://github.com/intel/intel-extension-for-transformers/tree/main/intel_extension_for_transformers/neural_chat/server) of `Intel Extension for Transformers`. You can choose to enable it by adding `use_neural_speed: true` in `config.yaml`.

- add `optimization` key section to use `Neural Speed` and its RTN quantization ([example](https://github.com/intel/intel-extension-for-transformers/blob/main/intel_extension_for_transformers/neural_chat/examples/deployment/codegen/backend/pc/woq/codegen.yaml)).
```yaml
device: "cpu"

# itrex int4 llm runtime optimization
optimization:
    use_neural_speed: true
    optimization_type: "weight_only"
    compute_dtype: "fp32"
    weight_dtype: "int4"
```
- add key `use_neural_speed` and key `use_gptq` to use `Neural Speed` and load `GPT-Q` model ([example](https://github.com/intel/intel-extension-for-transformers/blob/main/intel_extension_for_transformers/neural_chat/examples/deployment/codegen/backend/pc/gptq/codegen.yaml)).

```yaml
device: "cpu"
use_neural_speed: true
use_gptq: true
```

More details please refer to [Neural Chat](https://github.com/intel/intel-extension-for-transformers/tree/main/intel_extension_for_transformers/neural_chat).


## Quick Start (llama.cpp-like usage)

### Single (One-click) Step

```
python scripts/run.py model-path --weight_dtype int4 -p "She opened the door and see"
```

### Multiple Steps

#### Convert and Quantize

```bash
# skip the step if GGUF model is from Hugging Face or generated by llama.cpp
python scripts/convert.py --outtype f32 --outfile ne-f32.bin EleutherAI/gpt-j-6b
```

#### Inference

```bash
# Linux and WSL
OMP_NUM_THREADS=<physic_cores> numactl -m 0 -C 0-<physic_cores-1> python scripts/inference.py --model_name llama -m ne-q4_j.bin -c 512 -b 1024 -n 256 -t <physic_cores> --color -p "She opened the door and see"
```

```bash
# Windows
python scripts/inference.py --model_name llama -m ne-q4_j.bin -c 512 -b 1024 -n 256 -t <physic_cores|P-cores> --color -p "She opened the door and see"
```

> Please refer to [Advanced Usage](./docs/advanced_usage.md) for more details.

## Advanced Topics

### New model enabling
You can consider adding your own models, please follow the document: [graph developer document](./developer_document.md).

### Performance profiling
Enable `NEURAL_SPEED_VERBOSE` environment variable for performance profiling.

Available modes:
- 0: Print full information: evaluation time and operator profiling. Need to set `NS_PROFILING` to ON and recompile.
- 1: Print evaluation time. Time taken for each evaluation.
- 2: Profile individual operator. Identify performance bottleneck within the model. Need to set `NS_PROFILING` to ON and recompile.
