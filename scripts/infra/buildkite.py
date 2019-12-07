import subprocess as sp
import requests

import settings

ORG = "deep-learning"
PIPELINE = "speech-enhancement"


def swarm(num_workers: int):
    """
    Kicks off a swarm of builds in Buildkite
    """
    commit_hash = get_stdout(["git rev-parse HEAD"])
    branch = get_stdout(["git rev-parse --abbrev-ref HEAD"])
    assert "train" in branch, "Must use a training branch"
    for i in range(1, num_workers + 1):
        msg = f"Training swarm with worker {i} / {num_workers}"
        create_build(commit_hash, branch, msg)


def create_build(commit_hash, branch, message):
    data = {"commit": commit_hash, "branch": branch, "message": message}
    headers = {"Authorization": f"Bearer {settings.BUILDKITE_ACCESS_TOKEN}"}
    url = f"https://api.buildkite.com/v2/organizations/{ORG}/pipelines/{PIPELINE}/builds"
    resp = requests.post(url, json=data, headers=headers)
    try:
        resp.raise_for_status()
    except requests.HTTPError:
        raise Exception(resp.text)


def get_stdout(cmd):
    proc = sp.run(cmd, shell=True, check=True, stdout=sp.PIPE, encoding="utf-8")
    return proc.stdout.strip()
