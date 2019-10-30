#!/bin/bash
. ./scripts/docker/env.sh
exit 1
gcloud ai-platform jobs submit training $JOB_NAME \
  --scale-tier BASIC_GPU \
  --region $REGION \
  --master-image-uri $IMAGE_URI \
  --