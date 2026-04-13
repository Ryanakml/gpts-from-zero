#!/bin/bash

# Exit on error
set -e

echo "Setting up Python environment for gpts-from-zero..."

# Check if venv already exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

# Activate venv
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
if [ -f requirements.txt ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found, skipping dependency install"
    exit 1
fi

# Install jupyterlab if not present
if ! pip show jupyterlab > /dev/null 2>&1; then
    echo "Installing JupyterLab..."
    pip install jupyterlab
fi

echo "Setup complete."
echo "----------------------------------------"
echo "To activate this environment next time, run:"
echo "source venv/bin/activate"
echo "Then start JupyterLab with:"
echo "jupyter lab"
echo "----------------------------------------"
