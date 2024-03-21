#  Copyright (c) 2023 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

file(GLOB MODEL_UTILS_SOURCE "model_utils/*.cpp")

function(add_model target)
  add_library_w_warning(${target} ${ARGN})  # no (gpt) model utils needed
  target_compile_features(${target} PUBLIC cxx_std_11) # don't bump
  set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_link_libraries(${target} PUBLIC ne_layers bestla)
endfunction()

add_model(llama llama/llama.cpp llama/llama_utils.cpp ${MODEL_UTILS_SOURCE})
add_model(gptj gptj/gptj.cpp gptj/gptj_utils.cpp ${MODEL_UTILS_SOURCE})
add_model(mpt mpt/mpt.cpp mpt/mpt_utils.cpp ${MODEL_UTILS_SOURCE})
add_model(gptneox gptneox/gptneox.cpp gptneox/gptneox_utils.cpp ${MODEL_UTILS_SOURCE})
add_model(starcoder starcoder/starcoder.cpp starcoder/starcoder_utils.cpp ${MODEL_UTILS_SOURCE})
add_model(falcon falcon/falcon.cpp falcon/falcon_utils.cpp ${MODEL_UTILS_SOURCE})
add_model(opt opt/opt.cpp opt/opt_utils.cpp ${MODEL_UTILS_SOURCE})
add_model(bloom bloom/bloom.cpp bloom/bloom_utils.cpp ${MODEL_UTILS_SOURCE})
add_model(baichuan baichuan/baichuan.cpp baichuan/baichuan_utils.cpp ${MODEL_UTILS_SOURCE})
add_model(qwen qwen/qwen.cpp qwen/qwen_utils.cpp ${MODEL_UTILS_SOURCE})
add_model(whisper whisper/whisper.cpp whisper/whisper_utils.cpp ${MODEL_UTILS_SOURCE})
add_model(chatglm chatglm/chatglm.cpp chatglm/chatglm_utils.cpp ${MODEL_UTILS_SOURCE})
add_model(chatglm2 chatglm/chatglm2.cpp chatglm/chatglm2_utils.cpp ${MODEL_UTILS_SOURCE})
add_model(gemma gemma/gemma.cpp gemma/gemma_utils.cpp ${MODEL_UTILS_SOURCE})
add_model(phi phi/phi.cpp phi/phi_utils.cpp ${MODEL_UTILS_SOURCE})
add_model(stablelm stablelm/stablelm.cpp stablelm/stablelm_utils.cpp ${MODEL_UTILS_SOURCE})
add_model(chatglm3 chatglm/chatglm2.cpp chatglm/chatglm2_utils.cpp ${MODEL_UTILS_SOURCE})
