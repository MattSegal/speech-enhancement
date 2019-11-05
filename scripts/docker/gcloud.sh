#!/bin/bash
. ./scripts/docker/env.sh
gcloud ai-platform jobs submit training $1 \
    --master-image-uri $IMAGE_URI \
    --region $REGION \
    --scale-tier BASIC_GPU \
    --stream-logs
    # -- \ ## USer args
    # --model-dir=gs://$BUCKET_NAME/$MODEL_DIR \
    # --epochs=10
