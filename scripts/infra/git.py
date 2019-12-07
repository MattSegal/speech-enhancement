import subprocess as sp


def get_commit_hash():
    return get_stdout(["git rev-parse HEAD"])


def get_branch():
    return get_stdout(["git rev-parse --abbrev-ref HEAD"])


def set_new_branch(branch_name):
    get_stdout([f"git checkout -b {branch_name}"])


def set_branch(branch_name):
    get_stdout([f"git checkout {branch_name}"])


def push_new_branch(branch_name):
    get_stdout([f"git push -u origin {branch_name}"])


def get_stdout(cmd):
    proc = sp.run(cmd, shell=True, check=True, stdout=sp.PIPE, encoding="utf-8")
    return proc.stdout.strip()
