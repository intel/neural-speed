import time
import argparse
from pathlib import Path
from typing import List, Optional
from neural_speed import ModelServer
from transformers import AutoTokenizer


def main(args_in: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="example program llm model server")
    parser.add_argument("--model_name", type=str,
                        help="model_name from huggingface or local model path: String",
                        required=True)
    parser.add_argument("--model_path", type=Path,
                        help="Path to the local neural_speed low-bits model file: String",
                        required=True)
    parser.add_argument("--max_new_tokens", type=int,
                        help="global query max generation token length: Int", required=False,
                        default=128)
    parser.add_argument("--min_new_tokens", type=int,
                        help="global min new tokens for generation (only works in beam search): Int",
                        required=False, default=30)
    parser.add_argument("--num_beams", type=int,
                        help="global num beams for beam search generation: Int", required=False,
                        default=4)
    parser.add_argument("--do_sample", action="store_true", help="do sample for generation")
    parser.add_argument("--early_stopping", action="store_true",
                        help="do early_stopping for beam search generation")
    parser.add_argument("--return_prompt", action="store_true",
                        help="add prompt token ids in generation results")
    parser.add_argument("--threads", type=int, help="num threads for model inference: Int",
                        required=False, default=8)
    parser.add_argument("--max_request_num", type=int,
                        help="maximum number of running requests (or queries) for model inference: Int",
                        required=False, default=8)
    parser.add_argument("--print_log", action="store_true", help="print server running logs")
    parser.add_argument("--scratch_size_ratio", type=float,
                        help="scale for enlarge memory for model inference: Float",
                        required=False, default=1.0)
    parser.add_argument("--memory_dtype", type=str, help="KV cache memory dtype: String",
                        required=False, default="auto", choices=["f32", "f16", "auto"])
    parser.add_argument("--ctx_size", type=int, help="Size of the prompt context: "\
                        "Int (default: 512, can not be larger than specific model's context window"\
                        " length)", required=False, default=512)
    parser.add_argument("--seed", type=int,
                        help="NG seed: Int (default: -1, use random seed for < 0)",
                        required=False, default=-1)
    parser.add_argument("--repeat_penalty", type=float,
        help="Penalize repeat sequence of tokens: Float (default: 1.1, 1.0 = disabled)",
        required=False, default=1.1)
    parser.add_argument("--top_k", type=int,
        help="top_k in generated token sampling: Int (default: 40, <= 0 to use vocab size)",
        required=False, default=40)
    parser.add_argument("--top_p", type=float,
        help="top_p in generated token sampling: Float (default: 0.95, 1.0 = disabled)",
        required=False, default=0.95)
    parser.add_argument("--temperature", type=float,
        help="temperature in generated token sampling: Float (default: 0.8, 1.0 = disabled)",
        required=False, default=0.8)
    args = parser.parse_args(args_in)
    print(args)

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
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    res_collect = []
    # response function (deliver generation results and current remain working size in server)
    def f_response(res, working):
        ret_token_ids = [r.token_ids for r in res]
        res_collect.extend(ret_token_ids)
        ans = tokenizer.batch_decode(ret_token_ids, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
        print(f"working_size: {working}, ans:", flush=True)
        for a in ans:
            print(a)
            print("=====================================")

    added_count = 0
    s = ModelServer(args.model_name,
                    f_response,
                    str(args.model_path),
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                    min_new_tokens=args.min_new_tokens,
                    early_stopping=args.early_stopping,
                    do_sample=args.do_sample,
                    continuous_batching=True,
                    return_prompt=args.return_prompt,
                    threads=args.threads,
                    max_request_num=args.max_request_num,
                    print_log=args.print_log,
                    scratch_size_ratio = args.scratch_size_ratio,
                    memory_dtype= args.memory_dtype,
                    ctx_size=args.ctx_size,
                    seed=args.seed,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repetition_penalty=args.repeat_penalty,
                    temperature=args.temperature,
                    )
    for i in range(len(prompts)):
        p_token_ids = tokenizer(prompts[i], return_tensors='pt').input_ids.tolist()
        s.issueQuery(i, p_token_ids)
        added_count += 1
        time.sleep(2)  # adjust query sending time interval

    # recommend to use time.sleep in while loop to exit program
    # let cpp server owns more resources
    while (added_count != len(prompts) or not s.Empty()):
        time.sleep(1)
    del s
    print("should finished")

if __name__ == "__main__":
    main()
