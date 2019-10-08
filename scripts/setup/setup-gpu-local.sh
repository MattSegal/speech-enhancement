#! /bin/bash
# Config for AWS Ubuntu 18.04 AMI to install CUDA
set -e
apt update
apt install -y \
    build-essential \
    software-properties-common \
    ubuntu-drivers-common \
    dkms \
    freeglut3 \
    freeglut3-dev \
    libxi-dev \
    libxmu-dev


ubuntu-drivers autoinstall

add-apt-repository -y ppa:ubuntu-toolchain-r/test

apt update
apt install -y \
    gcc-6 \
    g++-6

update-alternatives \
    --install /usr/bin/gcc gcc /usr/bin/gcc-6 60 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-6

# Check GCC version
gcc -v

# Install cuda
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda


# Check graphics card
nvidia-smi

# Install sox for graphics
sudo apt install sox

# Verify torch config.
. enb/bin/activate
python3 << EOF
import torch
print('CUDA enabled:', torch.cuda.is_available())
print('Current device:', torch.cuda.get_device_name(0))
EOF