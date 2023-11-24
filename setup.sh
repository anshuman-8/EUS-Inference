#!/bin/bash

# Navigate to the project directory
cd "$(dirname "$0")"

# Create desktop file
echo "[Desktop Entry]
Name=EUSML
Exec=./venv/bin/python ./app.py
Icon=./icon.ico
Type=Application
Categories=Development;" > EUSML.desktop

# Create a virtual environment (if not exists)
if [ ! -d "venv" ]; then
    python3 -m venv venv
    # Install required Python packages from requirements.txt (if not exists)
    if [ ! -f "requirements.txt" ]; then
        echo "requirements.txt not found!"
        exit 1
    fi
    pip install -r requirements.txt
fi

# Activate the virtual environment
source venv/bin/activate


# # Run your PyQt app
# python ./your_app.py
