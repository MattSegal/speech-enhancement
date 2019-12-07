import time
import subprocess as sp
import requests

import settings
import git

ORG = "deep-learning"
PIPELINE = "speech-enhancement"


def swarm(num_workers: int):
    """
    Kicks off a swarm of builds in Buildkite
    """
    branch = git.get_branch()
    time_str = str(int(time.time()))
    assert "train" in branch, "Must use a training branch"
    for i in range(1, num_workers + 1):
        worker_branch = f"{branch}-{i}-{time_str}"
        print(f"Launching worker {worker_branch}")
        git.set_new_branch(worker_branch)
        git.push_new_branch(worker_branch)

    git.set_branch(branch)


# Not used
def create_build(commit_hash, branch, message):
    """
    Create a new build in Buildkite
    FIXME: This fails when pulling from S3 for some reason
    """
    data = {"commit": commit_hash, "branch": branch, "message": message}
    headers = {"Authorization": f"Bearer {settings.BUILDKITE_ACCESS_TOKEN}"}
    url = f"https://api.buildkite.com/v2/organizations/{ORG}/pipelines/{PIPELINE}/builds"
    resp = requests.post(url, json=data, headers=headers)
    try:
        resp.raise_for_status()
    except requests.HTTPError:
        raise Exception(resp.text)
