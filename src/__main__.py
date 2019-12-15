import os
import yaml

import click
from cerberus import Validator

from src import tasks

ENV_CHOICES = [f.split(".")[0] for f in os.listdir("config")]
CONFIG_SCHEMA = {
    "task": {"type": "string", "required": True, "nullable": False},
    "cuda": {"type": "boolean", "required": True, "nullable": False},
    "epochs": {"type": "integer", "required": True, "nullable": False},
    "batch_size": {"type": "integer", "required": True, "nullable": False},
    "subsample": {"type": "integer", "required": True, "nullable": True},
    "checkpoint_epochs": {"type": "integer", "required": True, "nullable": True},
    "wandb_name": {"type": "string", "required": True, "nullable": True},
}
validator = Validator(CONFIG_SCHEMA)


@click.command()
@click.option("--env", type=click.Choice(ENV_CHOICES, case_sensitive=False))
@click.option("--branch", default="")
def train_cli(env, branch):
    """
    Run model training
    """
    print(f"Running training job using {env} config.")
    with open(f"config/{env}.yaml", "r") as f:
        config = yaml.load(f)

    is_valid = validator.validate(config)
    assert is_valid, validator.errors

    job_name = branch.replace("train/", "")
    if job_name:
        old_name = config.get("wandb_name")
        print(f'Overriding W&B name "{old_name}" with "{job_name}"')
        config["wandb_name"] = job_name

    train = getattr(tasks, config["task"])
    train(
        num_epochs=config["epochs"],
        use_cuda=config["cuda"],
        batch_size=config["batch_size"],
        wandb_name=config["wandb_name"],
        subsample=config["subsample"],
        checkpoint_epochs=config["checkpoint_epochs"],
    )


train_cli()
