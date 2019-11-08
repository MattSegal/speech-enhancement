import os
import subprocess

DATA_S3_BUCKET = "matt-segal-datasets"


def fetch_data(bucket_dir, target_dir):
    """
    Recursively copy files from bucket dir to target dir
    """
    s3_key = f"s3://{DATA_S3_BUCKET}/{bucket_dir}"
    cmd = f"aws s3 cp --recursive --quiet {s3_key} {target_dir}"
    subprocess.run(args=[cmd], shell=True, check=True)


def upload_file(bucket_dir, file_path):
    """
    Upload a file from file_path to the bucket_dir
    """
    filename = os.path.basename(file_path)
    s3_key = f"s3://{DATA_S3_BUCKET}/{bucket_dir}/{filename}"
    cmd = f"aws s3 cp {file_path} {s3_key}"
    subprocess.run(args=[cmd], shell=True, check=True)


def download_file(s3_path, file_path):
    """
    Download a file from s3_path to file_path
    """
    s3_key = "s3://" + os.path.join(DATA_S3_BUCKET, s3_path)
    cmd = f"aws s3 cp {s3_key} {file_path}"
    subprocess.run(args=[cmd], shell=True, check=True)
