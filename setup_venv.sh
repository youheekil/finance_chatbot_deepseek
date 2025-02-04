#!/bin/bash

# Create virtual environment
python -m venv .venv

# Activate virtual environment (Unix/MacOS)
source .venv/bin/activate

# Install requirements
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
else
    echo "requirements.txt not found. Skipping package installation."
fi

echo "Virtual environment setup complete. Activated and ready to use."
