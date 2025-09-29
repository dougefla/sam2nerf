#!/bin/bash
conda create -n sam2nerf_test python=3.10 -y
conda activate sam2nerf_test
cd sam2
pip install -e .
cd checkpoints
bash download_ckpts.sh
cd ../..
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
cd workspace