#!/bin/bash

source /neural-speed/.github/workflows/scripts/change_color.sh

pip install cmake ninja clang-tidy==16.0.4
REPO_DIR=/neural-speed
log_dir=/neural-speed/.github/workflows/scripts/formatScan
log_path=${log_dir}/clangtidy.log

# compile binary
cd ${REPO_DIR}
mkdir build
cd build
cmake .. -G Ninja -DNS_USE_CLANG_TIDY=CHECK -DBTLA_ENABLE_OPENMP=OFF -DNS_USE_OMP=OFF
ninja 2>&1 | tee ${log_path}

if [[ ! -f ${log_path} ]] || [[ $(grep -c "warning:" ${log_path}) != 0 ]] || [[ $(grep -c "error" ${log_path}) != 0 ]]; then
    exit 1
fi
$BOLD_PURPLE && echo "Congratulations, check passed!" && $LIGHT_PURPLE && echo "You can click on the artifact button to see the log details." && $RESET
exit 0
