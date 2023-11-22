#!/bin/bash

source /neural-speed/.github/workflows/scripts/change_color.sh
log_dir=/neural-speed/.github/workflows/scripts/formatScan
cloc --include-lang=Python --csv --out=${log_dir}/cloc.csv /neural-speed
