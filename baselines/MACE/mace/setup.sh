#!/bin/bash

# Ensure conda is initialized
source /usr/local/etc/profile.d/conda.sh
conda activate mace

# Make sure correct CUDA path is set
export CUDA_HOME=/usr/local/cuda-11.7
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

# Check CUDA and torch
# nvcc --version
# python -c "import torch; print(torch.version.cuda, torch.__version__)"

# Install Grounded-Segment-Anything components
cd /content/MACE
cd Grounded-Segment-Anything

# Install each submodule cleanly
# pip install --no-build-isolation -e GroundingDINO
pip install  ./segment_anything

git submodule update --init --recursive
cd grounded-sam-osx && bash install.sh && cd ..

git clone https://github.com/xinyu1205/recognize-anything.git
pip install -r recognize-anything/requirements.txt
pip install ./recognize-anything/

echo "✅ Setup complete!"
