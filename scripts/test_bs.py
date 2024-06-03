"""usage: 
    numactl -l -C 0-31 python test_bs.py 4
    4 is the batch_size
"""

from transformers import AutoTokenizer
from neural_speed import Model
import sys
import time

model_name = "meta-llama/Llama-2-7b-hf"
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
            "What is the best way to learn how to play the guitar?",
            ]

# expand for dataset 
prompts = prompts * 100
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
# if the tokenizer has no pad_token, you can specify it.
tokenizer.pad_token = tokenizer.eos_token
pad_token_id = tokenizer.pad_token_id

bs = min(int(sys.argv[1]), len(prompts))
print("batch_size is {}.".format(bs))
inputs = tokenizer(prompts[:bs], padding=True, return_tensors='pt').input_ids

model = Model()
model.init(model_name, use_quant=True, weight_dtype="int4", compute_dtype="int8", group_size=128)
# greedy search example, top_k_top_p sampling and beam_search also supported
# do not forget to pass pad_token_id
# warmup
outputs = model.generate(inputs,
                         max_new_tokens=4,
                         do_sample=False,
                         pad_token=pad_token_id,
                         ignore_prompt=True,
                         max_request_num=bs)
t0 = time.time()
outputs = model.generate(inputs,
                         max_new_tokens=128,
                         do_sample=False,
                         pad_token=pad_token_id,
                         ignore_prompt=True,
                         max_request_num=bs)
duration = time.time() - t0
total_tokens = sum([len(a) for a in outputs])
print("throughput is {} token/s.".format(total_tokens / duration))
# ans = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
# for a in ans:
#     print(a)
#     print("===========================")