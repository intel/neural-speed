import time
import neural_speed.gptj_cpp as cpp
def f(res, working):
    print(f"res: {res}, working: {working}")

model_path = "/home/zhentao/ils/runtime_outs/ne_gptj_q_int4_bestla_cint8_g32.bin"
s = cpp.ModelServer(f, model_path, max_new_tokens=128, max_request_num=8)
s.issueQuery([cpp.Query(i, [i+.1, i+.2]) for i in range(5)])
time.sleep(3)
del s
print("should finished")
