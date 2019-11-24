#!/bin/bash
if [[ -z "$1" ]]; then
    echo "Training type parameter missing"
    exit 1
fi


if [[ "$1" == "aws" ]]; then
    if [[ -z "$2" ]]; then
        echo "Git branch missing"
        exit 1
    fi
    # Navigate to codebase
    cd code

    # Fetch the latest code
    git fetch
    git checkout $2
    git pull

    # Ensure requirements are up to date
    . ./env/bin/activate
    pip3 install -r requirements.txt

    # Allocate 32GB of swap memory
    sudo fallocate -l 32G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile

    # Ensure we can talk to Weights and Biases
    . ./scripts/wandb-login.sh

    python3.6 -W ignore -m src --env $1 --branch $2
else
    if [[ -d "env" ]]; then
        . ./env/bin/activate
    fi
    python3.6 -W ignore -m src --env $1
fi
