import subprocess

DATA_S3_BUCKET = "matt-segal-datasets"


def fetch_data(bucket_dir, target_dir):
    """
    Recursively copy files from bucket dir to target dir
    """
    s3_key = f"s3://{DATA_S3_BUCKET}/{bucket_dir}"
    cmd = f"aws s3 cp --recursive --quiet {s3_key} {target_dir}"
    subprocess.run(
        args=[cmd], shell=True, check=True,
    )


def upload_file(bucket_dir, file_path):
    s3_key = f"s3://{DATA_S3_BUCKET}/{bucket_dir}"
    cmd = f"aws s3 cp {file_path} s3://{DATA_S3_BUCKET}/{bucket_dir}"
    subprocess.run(
        args=[cmd], shell=True, check=True,
    )
