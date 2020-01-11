import time
from datetime import datetime
from dateutil.tz import tzutc

import boto3
import timeago
from tabulate import tabulate

import settings

client = boto3.client("ec2", region_name=settings.AWS_REGION)

DESCRIBE_KEYS = ["InstanceId", "InstanceType", "LaunchTime", "State"]


class NoInstanceAvailable(Exception):
    pass


def run_job(job_id: str):
    instance_type = settings.EC2InstanceType.g4dn
    run_instance(job_id, instance_type)


def stop_job(job_id: str):
    print(f"Stopping EC2 instances running job {job_id}... ", end="")
    instance_ids = [i["InstanceId"] for i in describe_instances() if i["name"] == job_id]
    client.terminate_instances(InstanceIds=instance_ids)
    print("request sent.")


def cleanup_volumes():
    volumes = client.describe_volumes()
    volume_ids = [v["VolumeId"] for v in volumes["Volumes"] if v["State"] == "available"]
    for v_id in volume_ids:
        print(f"Deleting orphaned volume {v_id}")
        client.delete_volume(VolumeId=v_id)


def run_instance(job_id: str, instance_type: str):
    print(f"Creating EC2 instance {instance_type} for job {job_id}... ", end="")
    client.run_instances(
        MaxCount=1,
        MinCount=1,
        ImageId=settings.EC2_AMI,
        InstanceType=instance_type,
        SecurityGroupIds=[settings.EC2_SECURITY_GROUP],
        IamInstanceProfile={"Name": "deeplearning"},
        KeyName="wizard",
        InstanceInitiatedShutdownBehavior="terminate",
        InstanceMarketOptions={
            "MarketType": "spot",
            "SpotOptions": {
                "MaxPrice": settings.EC2_SPOT_MAX_PRICE,
                "SpotInstanceType": "one-time",
            },
        },
        TagSpecifications=[
            {"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": job_id}]}
        ],
    )
    print("request sent.")
    instance = find_instance(job_id)
    instance_id = instance["InstanceId"]
    print(f"Waiting for server {instance_id} to boot... ", flush=True)
    instance_ready = False
    while not instance_ready:
        time.sleep(10)
        print(f"\tChecking status of server {instance_id}...")
        response = client.describe_instance_status(InstanceIds=[instance_id])
        if not response["InstanceStatuses"]:
            continue

        status = response["InstanceStatuses"][0]
        instance_ready = (
            status["InstanceState"]["Name"] == "running"
            and status["InstanceStatus"]["Status"] == "ok"
            and status["SystemStatus"]["Status"] == "ok"
        )

    print(f"Server ready to run job {job_id}.")


def find_instance(name):
    instances = describe_instances()
    for instance in instances:
        if instance["name"] == name:
            return instance


def start_instance(instance):
    name = instance["name"]
    print(f"Starting EC2 instance {name}")
    response = client.start_instances(InstanceIds=[instance["InstanceId"]])


def stop_instance(instance):
    name = instance["name"]
    print(f"Stopping EC2 instance {name}")
    response = client.stop_instances(InstanceIds=[instance["InstanceId"]])
    print(response)


def is_running(instance):
    status = instance["State"]["Name"]
    return status == "running"


def print_status(instances):
    now = datetime.utcnow().replace(tzinfo=tzutc())
    print("\nEC2 instance statuses\n")
    table_data = [
        [
            i["name"],
            i["InstanceType"],
            i["State"]["Name"],
            i["ip"],
            timeago.format(i["LaunchTime"], now),
        ]
        for i in instances
    ]
    table_str = tabulate(table_data, headers=["Name", "Type", "Status", "IP", "Launched"])
    print(table_str, "\n")


def describe_instances():
    response = client.describe_instances()
    aws_instances = []
    for reservation in response["Reservations"]:
        for aws_instance in reservation["Instances"]:
            aws_instances.append(aws_instance)

    instances = []
    for aws_instance in aws_instances:
        if aws_instance["State"]["Name"] == "terminated":
            continue

        name = ""
        for tag in aws_instance["Tags"]:
            if tag["Key"] == "Name":
                name = tag["Value"]

        instance = {}
        instance["name"] = name
        instances.append(instance)
        for k, v in aws_instance.items():
            if k in DESCRIBE_KEYS:
                instance[k] = v

        # Read IP address
        network_interface = aws_instance["NetworkInterfaces"][0]
        try:
            instance["ip"] = network_interface["Association"]["PublicIp"]
        except KeyError:
            instance["ip"] = None

    return instances
