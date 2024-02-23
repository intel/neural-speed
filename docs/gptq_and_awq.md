GPTQ & AWQ
=======

Neural Speed supports multiple weight-only quantization algorithms, such as GPTQ and AWQ.

More algorithm details please check [GPTQ](https://arxiv.org/abs/2210.17323) and [AWQ](https://arxiv.org/abs/2306.00978).

Validated GPTQ & AWQ models directly from the HuggingFace:
* [Llama-2-7B-Chat-GPT](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GPTQ) & [Llama-2-13B-Chat-GPT](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GPTQ)
* [CodeLlama-7B-Instruct-GPTQ](https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GPTQ) & [CodeLlama-13B-Instruct-GPTQ](https://huggingface.co/TheBloke/CodeLlama-13B-Instruct-GPTQ)
* [SOLAR-10.7B-v1.0-GPTQ](https://huggingface.co/TheBloke/SOLAR-10.7B-v1.0-GPTQ)
* [Llama-2-7B-AWQ](https://huggingface.co/TheBloke/Llama-2-7B-AWQ) & [Llama-2-13B-chat-AWQ](https://huggingface.co/TheBloke/Llama-2-13B-chat-AWQ)
* [CodeLlama-7B-AWQ](https://huggingface.co/TheBloke/CodeLlama-7B-AWQ) & [CodeLlama-13B-AWQ](https://huggingface.co/TheBloke/CodeLlama-13B-AWQ)

Please check more validated GPTQ & AWQ models in the list of [supported_models](./docs/supported_models.md).

## Examples

How to run GPTQ or AWQ models in Neural Speed:
```python
import sys
from transformers import AutoTokenizer, TextStreamer
from neural_speed import Model

if len(sys.argv) != 2:
    print("Usage: python python_api_example.py model_path")
model_name = sys.argv[1]

prompt = "Once upon a time, a little girl"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)

model = Model()
# Inference GPTQ models.
model.init(model_name, weight_dtype="int4", compute_dtype="int8", use_gptq=True)
# Inference AWQ models.
# model.init(model_name, weight_dtype="int4", compute_dtype="int8", use_awq=True)

outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300, do_sample=True)
```

Note: we have provided the [script](../scripts/python_api_example.py) to run these models.
