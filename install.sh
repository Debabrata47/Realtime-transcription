#!/bin/bash

# Install Python and pip
sudo apt update
sudo apt install -y python3 python3-pip

# Install necessary audio processing tools
sudo apt install -y sox ffmpeg libsndfile1

# Install uvicorn for ASGI server
sudo apt install uvicorn

# Python Installation
pip3 install -r requirements.txt


echo "Installation complete. You can now proceed with the setup."
