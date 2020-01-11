import os
import yaml
from pprint import pprint

import click
from cerberus import Validator

from src.tasks.tasks import TASKS

ENV_CHOICES = ("aws", "desktop", "laptop")
CONFIG_SCHEMA = {
    "runtime": {
        "type": "dict",
        "schema": {"cuda": {"type": "boolean", "required": True, "nullable": False},},
    },
    "logging": {
        "type": "dict",
        "schema": {
            "wandb": {
                "type": "dict",
                "schema": {
                    "run_name": {"type": "string", "required": True, "nullable": True},
                    "project_name": {
                        "type": "string",
                        "required": True,
                        "nullable": True,
                    },
                },
            },
            "checkpoint": {
                "type": "dict",
                "schema": {
                    "save_name": {"type": "string", "required": True, "nullable": True},
                    "save_epochs": {
                        "type": "integer",
                        "required": True,
                        "nullable": True,
                    },
                },
            },
        },
    },
    "training": {
        "type": "dict",
        "schema": {
            "epochs": {"type": "integer", "required": True, "nullable": False},
            "batch_size": {"type": "integer", "required": True, "nullable": False},
            "subsample": {"type": "integer", "required": True, "nullable": True},
        },
    },
}
validator = Validator(CONFIG_SCHEMA)


@click.command()
@click.option("--env", type=click.Choice(ENV_CHOICES, case_sensitive=False))
@click.option("--branch", default="")
def train_cli(env, branch):
    """
    Run model training
    """
    with open(f"config.yaml", "r") as f:
        configs = yaml.load(f)

    task_name = configs["task"]
    train_func = TASKS[task_name]
    # Read default config, merge in default env settings.
    print(f"Loading config for env {env}...")
    config = configs["default"]
    env_config = config["envs"][env]
    del config["envs"]
    config = merge_dicts(env_config, config)

    # Read task config, merge in default task env settings.
    task_config = configs["tasks"][task_name]
    task_env_config = task_config["envs"][env]
    del task_config["envs"]
    task_config = merge_dicts(task_env_config, task_config)

    # Merge task config into default config
    config = merge_dicts(task_config, config)

    # Finally, read W&B task name from git branch
    config["logging"]["wandb"]["run_name"] = None
    if branch.startswith("train/"):
        wandb_name = branch.replace("train/", "")
        config["logging"]["wandb"]["run_name"] = wandb_name

    print("Found task config:")
    pprint(config)
    is_valid = validator.validate(config)
    assert is_valid, validator.errors

    print(f"\n==== Running task {task_name} using {env} config =====\n")
    train_func(
        runtime=config["runtime"], logging=config["logging"], training=config["training"],
    )


def merge_dicts(src, dest):
    for key, value in src.items():
        if isinstance(value, dict):
            # get node or create one
            node = dest.setdefault(key, {})
            merge_dicts(value, node)
        else:
            dest[key] = value

    return dest


train_cli()
