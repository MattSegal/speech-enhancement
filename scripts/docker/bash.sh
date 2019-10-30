#!/bin/bash
. ./scripts/docker/env.sh
docker run --gpus all -it $IMAGE_URI bash