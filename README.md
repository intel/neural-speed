# Neural Speed

Neural Speed is an innovation library designed to provide the efficient inference of large language models (LLMs) on Intel platforms through the state-of-the-art (SOTA) low-bit quantization and sparsity powered by [Intel Neural Compressor](https://github.com/intel/neural-compressor) and [llama.cpp](https://github.com/ggerganov/llama.cpp). We provide the experimental features as below:

- Modular design to support new models
- [Highly optimized low precision kernels](neural_speed/core/README.md)
- Utilize AMX, VNNI, AVX512F and AVX2 instruction set
- Support CPU (x86 platforms only) and Intel GPU (WIP)
- Support 4bits and 8bits quantization

> Neural Speed is under active development so APIs are subject to change.

## Quick Start
There are two methods for utilizing the Neural Speed:
You can refer to [ITREX(intel extension for transformers)](https://github.com/intel/intel-extension-for-transformers) for [Transformer-based API](https://github.com/intel/intel-extension-for-transformers/blob/main/docs/weightonlyquant.md#llm-runtime-example-code). 

Or you can use Neural Speed straight forward.

> For details please refer to [Advanced Usage](#Advanced-usage).

#### One-click Python scripts
Run LLM with one-click python script including conversion, quantization and inference.
```
python scripts/run.py model-path --weight_dtype int4 -p "She opened the door and see"
```

#### Quantize and Inference Step By Step
Besides the one-click script, Neural Speed also offers the detailed script: 1) convert and quantize, and 2) inference.

##### 1. Convert and Quantize LLM
Neural Speed assumes the compatible model format as [llama.cpp](https://github.com/ggerganov/llama.cpp) and [ggml](https://github.com/ggerganov/ggml). You can also convert the model by following the below steps:

```bash
# convert the model directly use model id in Hugging Face. (recommended)
python scripts/convert.py --outtype f32 --outfile ne-f32.bin EleutherAI/gpt-j-6b
```

##### 2. Inference

We provide LLM inference script to run the quantized model. Please reach [us](mailto:itrex.maintainers@intel.com) if you want to run using C++ API directly.
```bash
#Linux and WSL
OMP_NUM_THREADS=<physic_cores> numactl -m 0 -C 0-<physic_cores-1> python scripts/inference.py --model_name llama -m ne-q4_j.bin -c 512 -b 1024 -n 256 -t <physic_cores> --color -p "She opened the door and see"
```

## Supported Hardware
| Hardware | Optimization |
|-------------|:-------------:|
|Intel Xeon Scalable Processors | ✔ |
|Intel Xeon CPU Max Series | ✔ |
|Intel Core Processors | ✔ |
|Intel Arc GPU Series | WIP |
|Intel Data Center GPU Max Series | WIP |
|Intel Gaudi2 | Not yet |

## Supported Models

Neural Speed supports the following models:
### Text Generation

<table>
<thead>
  <tr>
    <th rowspan="2">Model Name</th>
    <th colspan="2">INT8</th>
    <th colspan="2">INT4</th>
    <th rowspan="2">Transformer Version</th>
  </tr>
  <tr>
    <th>RTN</th>
    <th>GPTQ</th>
    <th>RTN</th>
    <th>GPTQ</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><a href="https://huggingface.co/meta-llama/Llama-2-7b-chat-hf" target="_blank" rel="noopener noreferrer">LLaMA2-7B</a>,
    <a href="https://huggingface.co/meta-llama/Llama-2-13b-chat-hf" target="_blank" rel="noopener noreferrer">LLaMA2-13B</a>,
    <a href="https://huggingface.co/meta-llama/Llama-2-70b-chat-hf" target="_blank" rel="noopener noreferrer">LLaMA2-70B</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>Latest</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/decapoda-research/llama-7b-hf" target="_blank" rel="noopener noreferrer">LLaMA-7B</a>,
    <a href="https://huggingface.co/decapoda-research/llama-13b-hf" target="_blank" rel="noopener noreferrer">LLaMA-13B</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>Latest</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/EleutherAI/gpt-j-6b" target="_blank" rel="noopener noreferrer">GPT-J-6B</a></td>
    <td>✅</td>
    <td>✅</td>
    <td> </td>
    <td> </td>
    <td>Latest</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/EleutherAI/gpt-neox-20b" target="_blank" rel="noopener noreferrer">GPT-NeoX-20B</a></td>
    <td>✅</td>
    <td>✅</td>
    <td> </td>
    <td> </td>
    <td>Latest</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/databricks/dolly-v2-3b" target="_blank" rel="noopener noreferrer">Dolly-v2-3B</a></td>
    <td>✅</td>
    <td>✅</td>
    <td> </td>
    <td> </td>
    <td>4.28.1 or newer</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/mosaicml/mpt-7b" target="_blank" rel="noopener noreferrer">MPT-7B</a>,
    <a href="https://huggingface.co/mosaicml/mpt-30b" target="_blank" rel="noopener noreferrer">MPT-30B</a></td>
    <td>✅</td>
    <td>✅</td>
    <td> </td>
    <td> </td>
    <td>Latest</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/tiiuae/falcon-7b" target="_blank" rel="noopener noreferrer">Falcon-7B</a>,
    <a href="https://huggingface.co/tiiuae/falcon-40b" target="_blank" rel="noopener noreferrer">Falcon-40B</a></td>
    <td>✅</td>
    <td>✅</td>
    <td> </td>
    <td> </td>
    <td>Latest</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/bigscience/bloomz-7b1" target="_blank" rel="noopener noreferrer">BLOOM-7B</a></td>
    <td>✅</td>
    <td>✅</td>
    <td> </td>
    <td> </td>
    <td>Latest</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/facebook/opt-125m" target="_blank" rel="noopener noreferrer">OPT-125m</a>,
    <a href="https://huggingface.co/facebook/opt-1.3b" target="_blank" rel="noopener noreferrer">OPT-1.3B</a>,
    <a href="https://huggingface.co/facebook/opt-13b" target="_blank" rel="noopener noreferrer">OPT-13B</a></td>
    <td>✅</td>
    <td>✅</td>
    <td> </td>
    <td> </td>
    <td>Latest</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/THUDM/chatglm-6b" target="_blank" rel="noopener noreferrer">ChatGLM-6B</a>,
    <a href="https://huggingface.co/THUDM/chatglm2-6b" target="_blank" rel="noopener noreferrer">ChatGLM2-6B</a></td>
    <td>✅</td>
    <td>✅</td>
    <td> </td>
    <td> </td>
    <td>4.33.1</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/baichuan-inc/Baichuan-13B-Chat" target="_blank" rel="noopener noreferrer">Baichuan-13B-Chat</a>,
    <a href="https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat" target="_blank" rel="noopener noreferrer">Baichuan2-13B-Chat</a></td>
    <td>✅</td>
    <td>✅</td>
    <td> </td>
    <td> </td>
    <td>4.33.1</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/mistralai/Mistral-7B-v0.1" target="_blank" rel="noopener noreferrer">Mistral-7B</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>4.34.0 or newer</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/Qwen/Qwen-7B-Chat" target="_blank" rel="noopener noreferrer">Qwen-7B</a>,
    <a href="https://huggingface.co/Qwen/Qwen-14B-Chat" target="_blank" rel="noopener noreferrer">Qwen-14B</a></td>
    <td>✅</td>
    <td>✅</td>
    <td> </td>
    <td> </td>
    <td>Latest</td>
  </tr>
</tbody>
</table>

### Code Generation

<table>
<thead>
  <tr>
    <th rowspan="2">Model Name</th>
    <th colspan="2">INT8</th>
    <th colspan="2">INT4</th>
    <th rowspan="2">Transformer Version</th>
  </tr>
  <tr>
    <th>RTN</th>
    <th>GPTQ</th>
    <th>RTN</th>
    <th>GPTQ</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><a href="https://huggingface.co/codellama/CodeLlama-7b-hf" target="_blank" rel="noopener noreferrer">Code-LLaMA-7B</a>,
    <a href="https://huggingface.co/codellama/CodeLlama-13b-hf" target="_blank" rel="noopener noreferrer">Code-LLaMA-13B</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>Latest</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/bigcode/starcoderbase-1b" target="_blank" rel="noopener noreferrer">StarCoder-1B</a>,
    <a href="https://huggingface.co/bigcode/starcoderbase-3b" target="_blank" rel="noopener noreferrer">StarCoder-3B</a>,
    <a href="https://huggingface.co/bigcode/starcoder" target="_blank" rel="noopener noreferrer">StarCoder-15.5B</a></td>
    <td>✅</td>
    <td>✅</td>
    <td> </td>
    <td> </td>
    <td>Latest</td>
  </tr>
</tbody>
</table>


## Install

### Build Python package
```shell
pip install .
```

### Build executable only

```shell
# Linux and WSL
git submodule update --init --recursive
mkdir build
cd build
cmake .. -G Ninja
ninja
```

```powershell
# Windows
# Install VisualStudio 2022 and open 'Developer PowerShell for VS 2022'
mkdir build
cd build
cmake ..
cmake --build . -j --config Release
```


## Advanced Usage
You can do quantization and inference without Transformer-API: [advanced_usage](./docs/advanced_usage.md).

### 3. Tensor Parallelism cross nodes/sockets

We support tensor parallelism strategy for distributed inference/training on multi-node and multi-socket. You can refer to [tensor_parallelism.md](./docs/tensor_parallelism.md) to enable this feature.


### 4. Contribution

You can consider adding your own models via [graph developer document](./developer_document.md).

### 5. Custom Stopping Criteria

You can customize the stopping criteria according to your own needs by processing the input_ids to determine if text generation needs to be stopped.
Here is a simple example, which requires a minimum generation length of 80 tokens. Once the `min_length` is met, encountering a terminator `eos_token_id` will end the generation.

```python
import torch
from typing import List
from transformers import StoppingCriteria, StoppingCriteriaList

class StopOnTokens(StoppingCriteria):
    def __init__(self, min_length: int, start_length: int, stop_token_id: List[int]):
        self.min_length = min_length
        self.start_length = start_length
        self.stop_token_id = stop_token_id

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        if input_ids.shape[-1] - self.start_length > self.min_length:
            for stop_id in self.stop_token_id:
                if input_ids[0][input_ids.shape[-1] - 1] == stop_id:
                    return True
        return False

stopping_criteria = StoppingCriteriaList(
    [
        StopOnTokens(
            min_length=80,
            start_length=inputs.shape[1],
            stop_token_id=[tokenizer.eos_token_id],
        )
    ]
)

outputs = model.generate(inputs, streamer=streamer, stopping_criteria=stopping_criteria)
```

### 6. Verbose Mode

Enable verbose mode and control tracing information using the `NEURAL_SPEED_VERBOSE` environment variable.

Available modes:
- 0: Print all tracing information. Comprehensive output, including: evaluation time and operator profiling.
- 1: Print evaluation time. Time taken for each evaluation.
- 2: Profile individual operator. Identify performance bottleneck within the model.
