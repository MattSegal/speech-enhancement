from datetime import datetime
from dateutil.tz import tzutc

import boto3
import timeago
from tabulate import tabulate

client = boto3.client("ec2")

DESCRIBE_KEYS = ["InstanceId", "InstanceType", "LaunchTime", "State"]


def find_instance(name):
    instances = describe_instances()
    for instance in instances:
        if instance["name"] == name:
            return instance

    return None


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
        [i["name"], i["State"]["Name"], i["ip"], timeago.format(i["LaunchTime"], now)]
        for i in instances
    ]
    table_str = tabulate(table_data, headers=["Name", "Status", "IP", "Launched"])
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
