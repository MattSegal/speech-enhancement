# Packer build

Install packer with

```bash
sudo apt install packer
```

Build an AMI with

```bash
./pack.sh
```

# Cleaning up old AMIs

- Deregister [here](https://console.aws.amazon.com/ec2/v2/home?region=ap-southeast-2#Images:sort=name)
- Delete snapshot [here](https://ap-southeast-2.console.aws.amazon.com/ec2/v2/home?region=ap-southeast-2#Snapshots:sort=snapshotId)
