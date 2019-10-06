#!/bin/bash
. "$(dirname $0)/settings.sh"
echo "Stopping EC2"
aws ec2 stop-instances --instance-ids $INSTANCE_ID
