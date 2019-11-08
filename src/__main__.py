import json

import click

from .training.train_wave_u import train


@click.group()
def cli():
    """
    Run model training
    """
    pass


@click.command()
def dev():
    print("Running training job using dev config.")
    with open("config.dev.json", "r") as f:
        config = json.load(f)
    train(
        num_epochs=config["epochs"],
        use_cuda=config["cuda"],
        wandb_name=config.get("wandb_name"),
        subsample=config.get("subsample"),
        checkpoint_epochs=config.get("checkpoint_epochs", 4),
    )


@click.command()
@click.option("--branch", default="")
def prod(branch):
    print("Running training job using prod config.")
    with open("config.prod.json", "r") as f:
        config = json.load(f)

    job_name = branch.replace("train/", "")
    if job_name:
        old_name = config.get("wandb_name")
        print(f'Overriding W&B name "{old_name}" with "{job_name}"')
        config["wandb_name"] = job_name

    train(
        num_epochs=config["epochs"],
        use_cuda=config["cuda"],
        wandb_name=config.get("wandb_name"),
        subsample=config.get("subsample"),
        checkpoint_epochs=config.get("checkpoint_epochs", 4),
    )


cli.add_command(prod)
cli.add_command(dev)
cli()
