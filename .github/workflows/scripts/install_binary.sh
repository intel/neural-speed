#!/bin/bash
source /neural-speed/.github/workflows/scripts/change_color.sh

cd /neural-speed
$BOLD_YELLOW && echo "---------------- git submodule update --init --recursive -------------" && $RESET
git config --global --add safe.directory "*"
git submodule update --init --recursive


$BOLD_YELLOW && echo "---------------- run python setup.py sdist bdist_wheel -------------" && $RESET
python setup.py bdist_wheel

$BOLD_YELLOW && echo "---------------- pip install binary -------------" && $RESET
pip install dist/neural_speed*.whl
pip list
