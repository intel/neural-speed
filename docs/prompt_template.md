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

## Chat with Baichuan2-7B:
```python
import argparse
from pathlib import Path
from typing import List, Optional
from transformers import AutoTokenizer, TextStreamer, AutoModelForCausalLM
from neural_speed import Model
import torch

def build_chat_input(model, tokenizer, messages: List[dict], max_new_tokens: int=0):
    def _parse_messages(messages, split_role="user"):
        system, rounds = "", []
        round = []
        for i, message in enumerate(messages):
            if message["role"] == "system":
                assert i == 0
                system = message["content"]
                continue
            if message["role"] == split_role and round:
                rounds.append(round)
                round = []
            round.append(message)
        if round:
            rounds.append(round)
        return system, rounds

    max_new_tokens = max_new_tokens or model.generation_config.max_new_tokens
    max_input_tokens = model.config.model_max_length - max_new_tokens
    system, rounds = _parse_messages(messages, split_role="user")
    system_tokens = tokenizer.encode(system)
    max_history_tokens = max_input_tokens - len(system_tokens)

    history_tokens = []
    for round in rounds[::-1]:
        round_tokens = []
        for message in round:
            if message["role"] == "user":
                round_tokens.append(model.generation_config.user_token_id)
            else:
                round_tokens.append(model.generation_config.assistant_token_id)
            round_tokens.extend(tokenizer.encode(message["content"]))
        if len(history_tokens) == 0 or len(history_tokens) + len(round_tokens) <= max_history_tokens:
            history_tokens = round_tokens + history_tokens  # concat left
            if len(history_tokens) < max_history_tokens:
                continue
        break

    input_tokens = system_tokens + history_tokens
    if messages[-1]["role"] != "assistant":
        input_tokens.append(model.generation_config.assistant_token_id)
    input_tokens = input_tokens[-max_input_tokens:]  # truncate left
    return torch.LongTensor([input_tokens]).to(model.device)


def main(args_in: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="main program llm running")
    parser.add_argument("--model_name", type=str, help="Model name: String", required=True)
    parser.add_argument("--model_path", type=Path, help="Path to the model: String", required=True)
    parser.add_argument("-m", "--model", type=Path, help="Path to the executed model: String", required=True)
    parser.add_argument("--format",
                        type=str,
                        default="GGUF",
                        choices=["NE", "GGUF"],
                        help="convert to the GGUF or NE format")
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="Prompt to start generation with: String (default: empty)",
        default="Once upon a time",
    )

    args = parser.parse_args(args_in)
    print(args)

    gguf_path = args.model.as_posix()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    transformer_model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    messages = []
    prompt = args.prompt
    messages.append({"role": "user", "content": prompt})
    inputs_ids = build_chat_input(transformer_model, tokenizer, messages, max_new_tokens = 300)
    print(inputs_ids)
    model = Model()
    model.init_from_bin(args.model_name, gguf_path)
    outputs = model.generate(inputs_ids, max_new_tokens=300, do_sample=True)
    words = tokenizer.decode(outputs[0])
    print(words)

if __name__ == "__main__":
    main()
```
