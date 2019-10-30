#!/bin/bash
. ./scripts/docker/env.sh
docker build -f scripts/docker/Dockerfile -t $IMAGE_URI .
