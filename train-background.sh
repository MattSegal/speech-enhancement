#! /bin/bash
git pull
. ./env/bin/activate
nohup ./train-local.py &
