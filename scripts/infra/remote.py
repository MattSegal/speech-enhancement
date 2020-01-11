import os
import subprocess

SSH_ARGS = "-o StrictHostKeyChecking=no -i ~/.ssh/wizard.pem"
BRANCH = os.environ.get("BUILDKITE_BRANCH", "master")


def ssh_interactive(instance):
    ip = instance["ip"]
    name = instance["name"]
    print(f"Starting SSH session with instance {name}.")
    subprocess.call(f"ssh {SSH_ARGS} ubuntu@{ip}", shell=True)


def ssh_run_job(instance):
    ip = instance["ip"]
    name = instance["name"]
    print(f"Starting SSH session with instance {name}.")
    subprocess.call(
        f"scp {SSH_ARGS} ./scripts/train.sh ubuntu@{ip}:/home/ubuntu/train.sh", shell=True
    )
    subprocess.call(
        f"ssh {SSH_ARGS} ubuntu@{ip} 'bash ~/train.sh aws {BRANCH}'", shell=True
    )
