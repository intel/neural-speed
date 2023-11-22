#!/bin/bash
source /neural-speed/.github/workflows/scripts/change_color.sh
pip install bandit==1.7.4
log_dir=/neural-speed/.github/workflows/scripts/formatScan
python -m bandit -r -lll -iii /neural-speed >${log_dir}/bandit.log
exit_code=$?

$BOLD_YELLOW && echo " -----------------  Current log file output start --------------------------"
cat ${log_dir}/bandit.log
$BOLD_YELLOW && echo " -----------------  Current log file output end --------------------------" && $RESET

if [ ${exit_code} -ne 0 ]; then
    $BOLD_RED && echo "Error!! Please Click on the artifact button to download and view Bandit error details." && $RESET
    exit 1
fi

$BOLD_PURPLE && echo "Congratulations, Bandit check passed!" && $LIGHT_PURPLE && echo " You can click on the artifact button to see the log details." && $RESET
exit 0
