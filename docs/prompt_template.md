# Prompt template

This document will show some examples to introduce how to correctly use prompt templates in Neural Speed and [ITREX](https://github.com/intel/intel-extension-for-transformers).

For the base model (without SFT for pre-training), prompt can be directly encoded into token ids without adding any special prefix or suffix token. But for the chat model, we need some prompt templates to generate correct and human understandable words. The reason is that these models are usually trained with specific prompt templates.

## Chat with ChatGLM3:
```python
from transformers import AutoTokenizer, TextStreamer
from neural_speed import Model

prompt = "你好"
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
inputs = tokenizer.build_chat_input(prompt)['input_ids']
model = Model()
model.init_from_bin(args.model_name, gguf_path)
outputs = model.generate(inputs, max_new_tokens=300, do_sample=True)
words = tokenizer.decode(outputs[0])
```

## Chat with LLaMA2:

```python
from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM, WeightOnlyQuantConfig

# Please change to local path to model, llama2 does not support online conversion, currently.
model_name = "meta-llama/Llama-2-7b-chat-hf"
woq_config = WeightOnlyQuantConfig(compute_dtype="int8", weight_dtype="int4")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
streamer = TextStreamer(tokenizer)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=woq_config, trust_remote_code=True)

while True:
    prompt = input("> ").strip()
    if prompt == "quit":
        break
    b_prompt = "[INST]{}[/INST]".format(prompt)  # prompt template for llama2
    inputs = tokenizer(b_prompt, return_tensors="pt").input_ids
    outputs = model.generate(inputs, streamer=streamer, interactive=True, ignore_prompt=True, do_sample=True)
```

## Chat with ChatGLM2:
```python
from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM, WeightOnlyQuantConfig

model_name = "THUDM/chatglm2-6b"  # or local path to model
woq_config = WeightOnlyQuantConfig(compute_dtype="int8", weight_dtype="int4")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
streamer = TextStreamer(tokenizer)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=woq_config, trust_remote_code=True)

while True:
    prompt = input("> ").strip()
    if prompt == "quit":
        break
    prompt = tokenizer.build_prompt(prompt)  # prompt template for chatglm2
    inputs = tokenizer([prompt], return_tensors="pt").input_ids
    outputs = model.generate(inputs, streamer=streamer, interactive=True, ignore_prompt=True, do_sample=True, n_keep=2)
```

## Chat with Qwen:
```python
from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM, WeightOnlyQuantConfig

model_name = "Qwen/Qwen-7B-Chat"  # or local path to model
woq_config = WeightOnlyQuantConfig(compute_dtype="int8", weight_dtype="int4")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
streamer = TextStreamer(tokenizer)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=woq_config, trust_remote_code=True)

while True:
    prompt = input("> ").strip()
    if prompt == "quit":
        break
    prompt = "\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n".format(prompt)  # prompt template for qwen
    inputs = tokenizer([prompt], return_tensors="pt").input_ids
    outputs = model.generate(inputs, streamer=streamer, interactive=True, ignore_prompt=True, do_sample=True)
```
