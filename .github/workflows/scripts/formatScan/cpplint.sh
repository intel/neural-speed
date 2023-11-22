#!/bin/bash

source /neural-speed/.github/workflows/scripts/change_color.sh

pip install cpplint
REPO_DIR=/neural-speed
log_dir=/neural-speed/.github/workflows/scripts/formatScan
log_path=${log_dir}/cpplint.log
cpplint --filter=-build/include_subdir,-build/header_guard --recursive --quiet --linelength=120 ${REPO_DIR}/neural_speed 2>&1 | tee ${log_path}
cpplint --filter=-build/include_subdir,-build/header_guard --recursive --quiet --linelength=120 ${REPO_DIR}/bestla 2>&1 | tee -a ${log_path}
if [[ ! -f ${log_path} ]] || [[ $(grep -c "Total errors found:" ${log_path}) != 0 ]]; then
    exit 1
fi
$BOLD_PURPLE && echo "Congratulations, check passed!" && $LIGHT_PURPLE && echo "You can click on the artifact button to see the log details." && $RESET
exit 0
