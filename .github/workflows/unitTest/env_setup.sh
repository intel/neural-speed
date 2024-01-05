pip list

# Install test requirements
echo "Install Tests Requirements"
cd $1 || exit 1
pwd
if [ -f "requirements.txt" ]; then
    python -m pip install --default-timeout=100 -r requirements.txt
    pip list
else
    echo "Not found requirements.txt file."
fi

pip install coverage
pip install pytest
