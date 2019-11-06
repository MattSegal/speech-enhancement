class EC2InstanceType:
    g3s = "g3s.xlarge"
    g4dn = "g4dn.xlarge"
    p2 = "p2.xlarge"


EC2_INSTANCE_TYPE_LIMIT = 2
EC2_AMI = "ami-05fbded987de0bcb0"
EC2_SPOT_MAX_PRICE = "0.75"
EC2_SECURITY_GROUP = "sg-3d0ecf44"
EC2_LAUNCH_PREFERENCE = [EC2InstanceType.g3s, EC2InstanceType.g4dn, EC2InstanceType.p2]
