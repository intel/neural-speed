#!/bin/bash

source /neural-speed/.github/workflows/scripts/change_color.sh
cd /neural-speed
$BOLD_YELLOW && echo "---------------- git submodule update --init --recursive -------------" && $RESET
git config --global --add safe.directory "*"
git submodule update --init --recursive

$BOLD_YELLOW && echo "---------------- install NeuralSpeed -------------" && $RESET
export PYTHONPATH=`pwd`
pip list


cd /neural-speed
log_dir=/neural-speed/.github/workflows/scripts/formatScan
if [ -f "requirements.txt" ]; then
    python -m pip install --default-timeout=100 -r requirements.txt
    pip list
else
    echo "Not found requirements.txt file."
fi

echo "[DEBUG] list pipdeptree..."
pip install pipdeptree
pipdeptree

python -m pylint -f json --disable=R,C,W,E1129 \
    --enable=line-too-long \
    --max-line-length=120 \
    --disable=no-name-in-module,import-error,no-member,undefined-variable,no-value-for-parameter,unexpected-keyword-arg,not-callable,no-self-argument,too-many-format-args,invalid-unary-operand-type,too-many-function-args \
    --extension-pkg-whitelist=numpy,nltk \
    --ignored-classes=TensorProto,NodeProto \
    --ignored-modules=tensorflow,torch,torch.quantization,torch.tensor,torchvision,mxnet,onnx,onnxruntime,neural_compressor,neural_compressor.benchmark,intel_extension_for_transformers.neural_engine_py,cv2,PIL.Image \
    /neural-speed/neural_speed >${log_dir}/pylint.json
exit_code=$?

$BOLD_YELLOW && echo " -----------------  Current log file output start --------------------------" && $RESET
cat ${log_dir}/pylint.json
$BOLD_YELLOW && echo " -----------------  Current log file output end --------------------------" && $RESET

if [ ${exit_code} -ne 0 ]; then
    $BOLD_RED && echo "Error!! Please Click on the artifact button to download and view Pylint error details." && $RESET
    exit 1
fi
$BOLD_PURPLE && echo "Congratulations, Pylint check passed!" && $LIGHT_PURPLE && echo " You can click on the artifact button to see the log details." && $RESET
exit 0
