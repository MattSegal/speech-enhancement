#!/bin/bash
. "$(dirname $0)/settings.sh"
echo "Starting EC2"
aws ec2 start-instances --instance-ids $INSTANCE_ID
