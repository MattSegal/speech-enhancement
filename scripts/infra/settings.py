import secrets


class EC2InstanceType:
    g3s = "g3s.xlarge"
    g4dn = "g4dn.2xlarge"  # Bump up for 32GB of memory
    # g4dn = "g4dn.xlarge"
    p2 = "p2.xlarge"


EC2_INSTANCE_TYPE_LIMIT = 2
EC2_AMI = "ami-01457c9c7fd48b0e2"
EC2_SPOT_MAX_PRICE = "0.75"
EC2_SECURITY_GROUP = "sg-3d0ecf44"
EC2_LAUNCH_PREFERENCE = [EC2InstanceType.g4dn, EC2InstanceType.g3s, EC2InstanceType.p2]
AWS_REGION = "ap-southeast-2"
BUILDKITE_ACCESS_TOKEN = getattr(secrets, "BUILDKITE_ACCESS_TOKEN", "")
