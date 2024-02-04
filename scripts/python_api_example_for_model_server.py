import time
import neural_speed.gptj_cpp as cpp
from transformers import AutoTokenizer

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

model_name = "EleutherAI/gpt-j-6b"  # model_name from huggingface or local model path
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

res_collect = []
def f_response(res, working):
    ret_token_ids = [r.token_ids for r in res]
    res_collect.extend(ret_token_ids)
    ans = tokenizer.batch_decode(ret_token_ids, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)
    print(f"working_size: {working}, ans:", flush=True)
    for a in ans:
        print(a)
        print("=====================================")

model_path = "gptj-q4.bin"  # please set your corresponding local neural_speed low-bits model file
added_count = 0
s = cpp.ModelServer(f_response,                      # response function (deliver generation results and current remain working size in server)
                    model_path,                      # model_path
                    max_new_tokens=128,              # global query max generation token length
                    num_beams=4,                     # global beam search related generation parameters
                    min_new_tokens=30,               # global beam search related generation parameters (default: 0)
                    early_stopping=True,             # global beam search related generation parameters (default: False)
                    continuous_batching=True,        # turn on continuous batching mechanism (default: True)
                    return_prompt=True,              # also return prompt token ids in generation results (default: False)
                    threads=56,                      # number of threads in model evaluate process (please bind cores if need)
                    max_request_num=8,               # maximum number of running requests (or queries, default: 8)
                    print_log=True,                  # print server running logs (default: False)
                    model_scratch_enlarge_scale = 1, # model memory scratch enlarge scale (default: 1)
                )
for i in range(len(prompts)):
    p_token_ids = tokenizer(prompts[i], return_tensors='pt').input_ids.tolist()
    s.issueQuery([cpp.Query(i, p_token_ids)])
    added_count += 1
    time.sleep(2)  # adjust query sending time interval

# recommend to use time.sleep in while loop to exit program
# let cpp server owns more resources
while (added_count != len(prompts) or not s.Empty()):
    time.sleep(1)
del s
print("should finished")
