import time

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
    else:
        click.echo(f"Instance {name} not found")


@click.command()
@click.argument("name")
def start(name):
    """Start an EC2 instance"""
    instance = aws.find_instance(name)
    if instance:
        aws.start_instance(instance)
    else:
        click.echo(f"Instance {name} not found")


@click.command()
@click.argument("name")
def ssh(name):
    """SSH into an EC2 instance"""
    instance = aws.find_instance(name)
    if instance and aws.is_running(instance):
        remote.ssh_interactive(instance)
    elif instance:
        click.echo(f"Instance {name} not running")
    else:
        click.echo(f"Instance {name} not found")


@click.command()
@click.argument("job_id")
def start(job_id):
    """
    Start a job
    """
    try:
        aws.run_job(job_id)
    except aws.NoInstanceAvailable:
        click.echo("Could not run job - no instances available")
        return

    print("Waiting 60s for server to boot... ", end="")
    time.sleep(60)
    print("done.")
    instance = aws.find_instance(job_id)
    start_time = time.time()
    while time.time() - start_time < 20:
        print("Attempting to run job")
        remote.ssh_run_job(instance)
        time.sleep(3)

    aws.stop_job(job_id)


@click.command()
@click.argument("job_id")
def stop(job_id):
    """
    Stop a job
    """
    aws.stop_job(job_id)


@click.command()
def cleanup():
    """
    Cleanup dangling AWS bits.
    TODO: Kill any GPU types older thank X hours.
    """
    aws.cleanup_volumes()


cli.add_command(cleanup)
cli.add_command(status)
cli.add_command(ssh)
cli.add_command(start)
cli.add_command(stop)
cli()
