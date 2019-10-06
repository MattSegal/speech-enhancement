#!/bin/bash
. "$(dirname $0)/settings.sh"
echo "EC2 Status"
aws ec2 describe-instance-status --instance-ids $INSTANCE_ID
