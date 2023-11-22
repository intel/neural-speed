#!/bin/bash

source /neural-speed/.github/workflows/script/change_color.sh
log_dir=/neural-speed/.github/workflows/script/formatScan
cloc --include-lang=Python --csv --out=${log_dir}/cloc.csv /neural-speed
