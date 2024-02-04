#!/bin/bash
# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -eo pipefail
set -x

# IMPORTANT! we use half of one socket cores to simulate TP functionality
cores_list=(24)
# model_list=("llama2-7b" "llama2-13b" "llama2-70b" "gptj-6b" "baichuan2-13b")
model_list=("llama2-7b")
input_list=(32 1024 2016)
output=32
beam_list=(1)
precision_list=("q4_j")
export NEURAL_SPEED_VERBOSE=1

# parse named arguments
while [ $# -gt 0 ]; do
    case "$1" in
    --local_models=*)
        # A json file which map HF model name to local path
        local_models="${1#*=}"
        ;;
    *)
        break
        ;;
    esac
    shift
done

declare -A model_name_map
model_name_map["llama2-7b"]="meta-llama/Llama-2-7b-hf"
model_name_map["llama2-13b"]="meta-llama/Llama-2-13b-hf"
model_name_map["llama2-70b"]="meta-llama/Llama-2-70b-hf"
model_name_map["gptj-6b"]="EleutherAI/gpt-j-6b"
model_name_map["baichuan2-13b"]="baichuan-inc/Baichuan2-13B-Chat"


function main {
    conda_env="$1"
    working_dir="$2"

    script_dir="${working_dir}/tests/model-test"
    PROMPTS_PATH="$script_dir/cpp_graph_prompts.json"
    convert_dir="${working_dir}/neural_speed/convert"

    # init conda
    . $(dirname ${CONDA_EXE})/../etc/profile.d/conda.sh
    conda activate $conda_env || source activate $conda_env
    pip install cmake psutil
    # get the compiler version
    gcc_version_info=$(gcc --version)
    compiler_version=$(echo "$gcc_version_info" | grep -oP 'gcc \(.*\) \K[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    if [[ "${compiler_version}" != "12.3.0" ]]; then
        conda install --update-deps -c conda-forge gxx==${compiler_version} gcc==${compiler_version} gxx_linux-64==${compiler_version} libstdcxx-ng sysroot_linux-64 -y
    fi
    # check oneCCL from oneapi or build from source
    if [ -z ${ONEAPI_ROOT} ]; then
      if [ -d /opt/intel/oneapi ]; then
        ONEAPI_ROOT=/opt/intel/oneapi
      elif [ -d ~/intel/oneapi ]; then
        ONEAPI_ROOT=~/intel/oneapi
      fi
    fi
    
    if [ ! -z ${ONEAPI_ROOT} ]; then
      source ${ONEAPI_ROOT}/setvars.sh
    fi
    # check oneCCL existence 
    if ! [ -x "$(command -v mpirun)" ]; then
      # check oneCCL and build
      ccl_dir=${working_dir}/oneCCL/build/_install
      if [ ! -d "$ccl_dir" ]; then
          cd ${working_dir}
          git clone https://github.com/intel/oneCCL.git
          cd ${working_dir}/oneCCL
          git checkout 2021.9
          sed -i 's/cpu_gpu_dpcpp/./g' cmake/templates/oneCCLConfig.cmake.in
          mkdir build && cd build
          cmake ..
          make -j install
      fi
      source ${ccl_dir}/env/setvars.sh
    fi

    # compile binary
    cd ${working_dir}
    if [ ! -d "${working_dir}/build" ]; then
        mkdir build
    fi
    cd ${working_dir}/build
    cmake -DNS_TP=ON .. 
    make -j
    cd ..

    ## prepare example requirement
    pip install -r requirements.txt

    # launch benchmark
    for model in ${model_list[@]}; do
        model_name="${model_name_map["$model"]}"
        if [[ -n "$local_models" ]]; then
            model_path=$(python -c "import sys, json; print(json.load(sys.stdin).get('$model_name', '$model_name'))" <"$local_models")
        else
            model_path=$model_name
        fi
        if [[ "${model}" == "llama2-7b" ]]; then
            convert_script="${convert_dir}/convert_llama.py"
            quant_script="./build/bin/quant_llama"
            infer_cmd="./build/bin/run_llama"
        elif [[ "${model}" == "llama2-13b" ]]; then
            convert_script="${convert_dir}/convert_llama.py"
            quant_script="./build/bin/quant_llama"
            infer_cmd="./build/bin/run_llama"
        elif [[ "${model}" == "llama2-70b" ]]; then
            convert_script="${convert_dir}/convert_llama.py"
            quant_script="./build/bin/quant_llama"
            infer_cmd="./build/bin/run_llama"
        elif [[ "${model}" == "gptj-6b" ]]; then
            convert_script="${convert_dir}/convert_gptj.py"
            quant_script="./build/bin/quant_gptj"
            infer_cmd="./build/bin/run_gptj"
        elif [[ "${model}" == "baichuan-13b" ]]; then
            convert_script="${convert_dir}/convert_baichuan.py"
            quant_script="./build/bin/quant_baichuan"
            infer_cmd="python ./scripts/inference.py --model_name baichuan --tokenizer baichuan-inc/Baichuan2-13B-Chat"
        fi
        ## prepare fp32 bin if not exists
        f32_model=${working_dir}/${model}-f32.bin
        if [ ! -f "$f32_model" ]; then
          python ${convert_script} --outtype f32 --outfile ${f32_model} ${model_path}
        fi
        for cores_per_instance in ${cores_list[@]}; do
            for input in ${input_list[@]}; do
                for precision in ${precision_list[@]}; do
                    # [[ "${input}" == "32" ]] && output=32 ||
                    prompt=$(python -c "import sys, json; i = json.load(sys.stdin)['$input']; print(i['prompts'][i['map'].get('$model', 'default')])" <$PROMPTS_PATH)
                    if [[ -z $prompt ]]; then
                        echo "Error: Unexpedted input: $input" 1>&2
                        continue
                    fi
                    ctx=$(($output + $input + 10))
                    logs_file="${model}-${precision}-${cores_per_instance}-${input}-${output}.log"
                    ## prepare model.bin
                    quantized_model="${model}-${precision}.bin"
                    if [[ ! -f ${quantized_model} ]]; then
                        if [[ ${precision} == "q4_j" ]]; then
                            ${quant_script} --model_file ${f32_model} --out_file ${quantized_model} --nthread $cores_per_instance --weight_dtype int4 --group_size 128 --scale_dtype fp32 --compute_dtype fp32 --alg sym
                        else
                            echo "Not supported precision on TP"
                            exit 1
                        fi
                    fi
                    ## run inference
                    export LANG=en_US.UTF-8
                    export LC_ALL=en_US.UTF-8
                    if [[ "${model}" == "baichuan"* ]]; then
                        echo $infer_cmd -t $cores_per_instance -c ${ctx} -n ${output} -m ${model}-${precision}.bin --ids \"$ids\"  >  run_${model}.sh
                    else
                        echo $infer_cmd --seed 1234 -t $cores_per_instance -b 2047 -c ${ctx} -n ${output} -m ${model}-${precision}.bin -p \"$prompt\"  >  run_${model}.sh
                    fi
                    taskset -c 0-$(($cores_per_instance * 1 - 1)) bash run_${model}.sh 2>&1 | tee ${WORKING_DIR}/"origin_"${logs_file}
                    TP_LOCAL_SIZE=2 mpirun -n 1 taskset -c 0-$(($cores_per_instance * 1 - 1)) bash run_${model}.sh  : -n 1 taskset -c $(($cores_per_instance))-$(($cores_per_instance * 2- 1)) bash run_${model}.sh  2>&1 | tee ${WORKING_DIR}/"tp_"${logs_file}
                    collect_perf "origin_"${logs_file} 1 ${precision} ${input}
                    collect_perf "tp_"${logs_file} 2 ${precision} ${input}
                    check_accuracy "origin_"${logs_file} "tp_"${logs_file} "${prompt}" 2
                    exit 1
                done
            done
        done
    done
    conda deactivate >/dev/null 2>&1
}

function check_accuracy {
    origin_output=$1
    tp_output=$2
    prompt=$3
    rank=$4
    last_words=$(echo "$prompt" | awk '{for(i=NF-2; i<NF; i++) printf $i " ";printf $NF}')
    origin_first_token=$(grep -oP "${last_words} \K\S+" $origin_output)
    tp_first_token=$(grep -oP "${last_words} \K\S+" $tp_output | awk -v rank="$rank" 'NR==rank {print $1}')
    echo "origin first token is $origin_first_token and tp first token is ${tp_first_token}\n"
    if [[ "$origin_first_token" == "$tp_first_token" ]]; then
        echo "accuracy is good"
    else
        echo "accuracy not good!"
        exit 1
    fi
}

function collect_perf {
    # latency
    log_dir="${WORKING_DIR}/$1"
    rank=$2
    precision=$3
    input_tokens=$4
    eval_time=($(grep -i 'eval time' ${log_dir} | grep -v "prompt" | grep -oP '\(\s*\K[0-9]+\.[0-9]+(?= ms per token)')) 
    first_token_time=($(grep -i 'eval time' ${log_dir} | grep "prompt" | grep -o -E '=\s*[0-9]+\.[0-9]+ ms' | awk '{print $2}'))
    printf "${model},${precision},${rank},${input_tokens},${first_token_time},${eval_time},${log_dir}\n" | tee -a ${WORKING_DIR}/tp_summary.log
    set +x
    echo -e "\n\n-------- Summary --------"
    sed -n '1p;$p' ${WORKING_DIR}/tp_summary.log | column -t -s ','
}


main $@ 2>&1 | tee ${WORKING_DIR}/launch.log
