#! /bin/bash
set -e
apt install -y \
    python3-pip \
    virtualenv \
    sox

virtualenv -p python3 env

. ./env/bin/activate

pip3 install -r requirements.txt

# Verify torch config.
. enb/bin/activate
python3 << EOF
import torch
print('CUDA enabled:', torch.cuda.is_available())
print('Current device:', torch.cuda.get_device_name(0))
EOF