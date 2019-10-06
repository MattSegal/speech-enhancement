#/bin/bash
set -e
. ./scripts/infra/.ec2-ip.sh
ssh root@$EC2_IP
