import yaml

import click

from src import tasks


@click.group()
def cli():
    """
    Run model training
    """
    pass


@click.command()
def dev():
    print("Running training job using dev config.")
    with open("config/dev.yaml", "r") as f:
        config = yaml.load(f)

    run_training(config)


@click.command()
@click.option("--branch", default="")
def prod(branch):
    print("Running training job using prod config.")
    with open("config/prod.yaml", "r") as f:
        config = yaml.load(f)

    run_training(config, branch)


@click.command()
@click.option("--branch", default="")
def aws(branch):
    print("Running training job using AWS config.")
    with open("config/aws.yaml", "r") as f:
        config = yaml.load(f)

    run_training(config, branch)


def run_training(config, branch=""):
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


cli.add_command(aws)
cli.add_command(prod)
cli.add_command(dev)
cli()
