#!/bin/bash
. "$(dirname $0)/settings.sh"
aws ec2 describe-instances --instance-ids $INSTANCE_ID > "$(dirname $0)/.ec2-info.json"
python3 "$(dirname $0)/read_ip.py"
