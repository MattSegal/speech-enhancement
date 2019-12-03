#!/bin/bash
S3_BUCKET=matt-segal-datasets
aws s3 cp --recursive $1 s3://$S3_BUCKET/$2
