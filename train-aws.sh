#!/bin/bash
# Training script for AWS EC2 instances
cd code
git pull
. ./env/bin/activate
pip3 install -r requirements.txt
. ./scripts/wandb-login.sh
python3 -W ignore -m src dev
