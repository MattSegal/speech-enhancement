#! /bin/bash
set -e

add-apt-repository ppa:jonathonf/python-3.6
apt-get update
apt install -y \
    python3.6 \
    python3-pip \
    python3.6-dev \
    virtualenv \
    sox

update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.5 1
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 2
update-alternatives --config python3

# Manually select v3.6
virtualenv -p python3 env

. ./env/bin/activate

pip3 install -r requirements.txt

# Verify torch config.
. env/bin/activate
python3 << EOF
import torch
print('CUDA enabled:', torch.cuda.is_available())
print('Current device:', torch.cuda.get_device_name(0))
EOF