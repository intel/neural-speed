import time
import neural_speed.llama_cpp as cpp
def f(res, working):
    print(f"res: {res}, working: {working}")


s = cpp.ModelServer(f)
s.issueQuery([cpp.Query(i, [i+.1, i+.2]) for i in range(5)])
time.sleep(3)
del s
print("should finished")
