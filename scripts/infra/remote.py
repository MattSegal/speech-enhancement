import subprocess


def ssh_interactive(instance):
    ip = instance["ip"]
    name = instance["name"]
    print(f"Starting SSH session with instance {name}.")
    subprocess.call(f"ssh root@{ip}", shell=True)
