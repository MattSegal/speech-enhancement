import subprocess

DATA_S3_BUCKET = "matt-segal-datasets"


def fetch_data(bucket_dir, target_dir):
    """
    Recursively copy files from bucket dir to target dir
    """
    subprocess.run(
        args=[f"aws s3 cp --recursive --quiet s3://{DATA_S3_BUCKET}/{bucket_dir} {target_dir}"],
        shell=True,
        check=True,
    )

