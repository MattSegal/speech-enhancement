import os
import yaml

import click

from src import tasks

ENV_CHOICES = [f.split(".")[0] for f in os.listdir("config")]


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
        wandb_name=config.get("wandb_name"),
        subsample=config.get("subsample"),
        checkpoint_epochs=config.get("checkpoint_epochs", 6),
    )


train_cli()
