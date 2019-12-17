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

    wandb_name = None
    if branch.startswith("train/"):
        wandb_name = branch.replace("train/", "")
        print(f'Using W&B name "{wandb_name}"')

    train = getattr(tasks, config["task"])
    train(
        num_epochs=config["epochs"],
        use_cuda=config["cuda"],
        batch_size=config["batch_size"],
        wandb_name=wandb_name,
        subsample=config["subsample"],
        checkpoint_epochs=config["checkpoint_epochs"],
    )


train_cli()
