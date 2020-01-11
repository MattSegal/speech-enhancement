#!/bin/bash
if [[ -z "$1" ]]; then
    echo "ERROR: Training type parameter missing"
    exit 1
fi


if [[ "$1" == "aws" ]]; then
    if [[ -z "$2" ]]; then
        echo "ERROR: Git branch missing"
        exit 1
    fi
    # Navigate to codebase
    cd code

    # Fetch the latest code
    echo -e "\n>>> Pulling latest code\n"
    git fetch
    git checkout $2
    git pull
    echo -e "\n>>> Done pulling latest code\n"

    # Ensure requirements are up to date
    echo -e "\n>>> Installing Python requirements\n"
    . ./env/bin/activate
    pip3 install -r requirements.txt
    echo -e "\n>>> Done installing Python requirements\n"

    # Allocate 32GB of swap memory
    echo -e "\n>>> Allocating swap memory\n"
    sudo fallocate -l 32G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo -e "\n>>> Done allocating swap memory\n"

    # Ensure we can talk to Weights and Biases
    echo -e "\n>>> Logging in to W&B\n"
    . ./scripts/wandb-login.sh
    echo -e "\n>>> Done logging in to W&B\n"

    echo -e "\n>>> Running training script with env $1 and branch $2\n"
    python3.6 -u -W ignore -m src --env $1 --branch $2
    echo -e "\n>>> Done training script with env $1 and branch $2\n"
else
    if [[ -d "env" ]]; then
        . ./env/bin/activate
    fi
    python3.6 -u -W ignore -m src --env $1
fi
