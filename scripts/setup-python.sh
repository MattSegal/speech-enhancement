#! /bin/bash
apt install -y \
    python3-pip \
    virtualenv

virtualenv -p python3 env

. ./env/bin/activate

pip3 install -r requirments.txt
