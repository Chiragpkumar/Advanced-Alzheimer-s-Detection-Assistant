#!/bin/bash

# Update and install libGL.so.1
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx

# Install Python dependencies
pip install -r requirements.txt
