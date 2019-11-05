#! /bin/bash
# Build AWS deep learning machine image.
# Called by Packer to produce an AWS AMI
set -e

# Handle some magical apt sources race condition bullshit.
echo "Waiting up to 180 seconds for cloud-init to update /etc/apt/sources.list"
timeout 180 /bin/bash -c 'until stat /var/lib/cloud/instance/boot-finished 2>/dev/null; do echo waiting ...; sleep 1; done'

# Install basic requirements
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3.6-dev \
    virtualenv \
    sox

# Install NVIDIA's CUDA
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda

# Check graphics card
nvidia-smi

# Get source code
git clone https://github.com/MattSegal/speech-enhancement.git code

# Install Python requirements
cd ~/code
virtualenv -p python3 env
. ./env/bin/activate
pip3 install -r requirements.txt

# Verify PyTorch configuration
python3 << EOF
import torch
print('CUDA enabled:', torch.cuda.is_available())
print('Current device:', torch.cuda.get_device_name(0))
EOF