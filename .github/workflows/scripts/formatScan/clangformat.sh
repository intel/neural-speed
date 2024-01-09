#!/bin/bash
source /neural-speed/.github/workflows/scripts/change_color.sh

pip install clang-format==14.0.0
log_dir=/neural-speed/.github/workflows/scripts/formatScan
log_path=${log_dir}/clangformat.log

cd /neural-speed
git config --global --add safe.directory "*"

cd /neural-speed
python clang-format.py

echo "run git diff"
git diff 2>&1 | tee -a ${log_path}

if [[ ! -f ${log_path} ]] || [[ $(grep -c "diff" ${log_path}) != 0 ]]; then
    exit 1
fi
$BOLD_PURPLE && echo "Congratulations, check passed!" && $LIGHT_PURPLE && echo "You can click on the artifact button to see the log details." && $RESET
exit 0
