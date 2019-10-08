import click

import aws
import remote


@click.group()
def cli():
    """
    Simple AWS EC2 instance manager
    """
    pass


@click.command()
def status():
    """Print EC2 instance status"""
    instances = aws.describe_instances()
    aws.print_status(instances)


@click.command()
@click.argument("name")
def stop(name):
    """Stop an EC2 instance"""
    instance = aws.find_instance(name)
    if instance:
        aws.stop_instance(instance)


@click.command()
@click.argument("name")
def start(name):
    """Start an EC2 instance"""
    instance = aws.find_instance(name)
    if instance:
        aws.start_instance(instance)


@click.command()
@click.argument("name")
def ssh(name):
    """SSH into an EC2 instance"""
    instance = aws.find_instance(name)
    if instance and aws.is_running(instance):
        remote.ssh_interactive(instance)


cli.add_command(status)
cli.add_command(ssh)
cli.add_command(start)
cli.add_command(stop)
cli()
