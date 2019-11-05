#!/bin/bash
git pull
. /app/scripts/wandb-login.sh
/app/train-dev.sh
