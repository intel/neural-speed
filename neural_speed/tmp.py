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

model_name = "/home/zhentao/gpt-j-6b"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

res_collect = []
def f(res, working):
    ans = tokenizer.batch_decode([r.token_ids for r in res], skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)
    print(f"working_size: {working}, ans:", flush=True)
    for a in ans:
        res_collect.append(a)
        print(a)
        print("================")

model_path = "/home/zhentao/ils/runtime_outs/ne_gptj_q_int4_bestla_cint8_g32.bin"
added_count = 0
s = cpp.ModelServer(f, model_path, max_new_tokens=128, max_request_num=8, threads=56, num_beams=4,
                    min_new_tokens=30, early_stopping=True, continuous_batching=True,
                    return_prompt=True)
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
