#!/bin/bash

source /neural-speed/.github/workflows/scripts/change_color.sh

REPO_DIR=/neural-speed
log_dir=/neural-speed/.github/workflows/scripts/formatScan
pydocstyle --convention=google ${REPO_DIR} >${log_dir}/pydocstyle.log
exit_code=$?

$BOLD_YELLOW && echo " -----------------  Current pydocstyle cmd start --------------------------" && $RESET
echo "pydocstyle --convention=google ${REPO_DIR} >${log_dir}/pydocstyle.log"
$BOLD_YELLOW && echo " -----------------  Current pydocstyle cmd end --------------------------" && $RESET

$BOLD_YELLOW && echo " -----------------  Current log file output start --------------------------"
cat $log_dir/pydocstyle.log
$BOLD_YELLOW && echo " -----------------  Current log file output end --------------------------" && $RESET

if [ ${exit_code} -ne 0 ]; then
    $BOLD_RED && echo "Error!! Please Click on the artifact button to download and view error details." && $RESET
    exit 1
fi

$BOLD_PURPLE && echo "Congratulations, check passed!" && $LIGHT_PURPLE && echo " You can click on the artifact button to see the log details." && $RESET
exit 0
