#! /bin/bash
# Config for AWS Ubuntu 18.04 AMI to install CUDA
set -e
apt update
apt install -y \
    build-essential \
    software-properties-common \
    ubuntu-drivers-common

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

# Install nvidia + cuda
apt install nvidia-cuda-toolkit

# Check graphics card
nvidia-smi

# Check CUDA version
nvcc --version
