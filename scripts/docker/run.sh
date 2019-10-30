#!/bin/bash
. ./scripts/docker/env.sh
docker run \
    --gpus all \
    -v "$(pwd)":/app \
    $IMAGE_URI \
    /app/scripts/docker/entrypoint.sh

