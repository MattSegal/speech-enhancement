#! /bin/bash
# Use Packer to produce an AWS AMI for deep dearning
set -e
packer validate ami.json
packer build ami.json