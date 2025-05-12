# First deactivate any active environment
deactivate || exit

# Ensure pyenv has correct Python version
pyenv install 3.11.8
pyenv local 3.11.8

# Remove existing Poetry environment
rm -rf .venv/

# Clear Poetry cache
rm -rf ~/.cache/pypoetry

# Create new environment, activate it and install requirements.txt

python3 -m venv .venv

source .venv/bin/activate

python3 main.py