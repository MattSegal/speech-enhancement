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
def prod():
    with open("config.prod.json", "r") as f:
        config = json.load(f)
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
