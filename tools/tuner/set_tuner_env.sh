script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd ${script_dir}/../../
xetla_src_dir=`pwd`
cd - &> /dev/null
#xetla_src_dir=~/00_code/xetla_opensource/libraries.gpu.xetla
echo "XeTLA is: ${xetla_src_dir}."
echo "If you use other XeTLA, please set the XETLA_SRC_DIR by yourself!"
export XETLA_SRC_DIR=${xetla_src_dir}
export XETLA_TUNER_CMD_TIME_OUT=50
# export ZE_AFFINITY_MASK=0.0
