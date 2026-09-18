import os
import boto3
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

load_dotenv(PROJECT_ROOT / ".env")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_RAW_PREFIX = os.getenv("S3_RAW_PREFIX", "raw/cmapss")


required_variables = {
    "AWS_ACCESS_KEY_ID": AWS_ACCESS_KEY_ID,
    "AWS_SECRET_ACCESS_KEY": AWS_SECRET_ACCESS_KEY,
    "AWS_REGION": AWS_REGION,
    "S3_BUCKET_NAME": S3_BUCKET_NAME,
}

missing = [
    name for name, value in required_variables.items()
    if not value
]

if missing:
    raise ValueError(
        f"Missing environment variables: {', '.join(missing)}"
    )


RAW_DIR = PROJECT_ROOT / "data" / "raw"

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

def upload_raw_data():

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Raw directory does not exist: {RAW_DIR}"
        )

    files = [
        file
        for file in RAW_DIR.rglob("*")
        if file.is_file()
    ]

    if not files:
        print("No files found in data/raw/")
        return

    print(f"Found {len(files)} files.")
    print()

    for file in files:

        relative_path = file.relative_to(RAW_DIR)

        s3_key = (
            f"{S3_RAW_PREFIX}/{relative_path.as_posix()}"
        )

        print(f"Uploading: {file.name}")
        print(f"  → s3://{S3_BUCKET_NAME}/{s3_key}")

        s3.upload_file(
            str(file),
            S3_BUCKET_NAME,
            s3_key
        )

        print("\nUploaded successfully\n")

if __name__ == "__main__":

    print("=" * 60)
    print("FORESIGHT - C-MAPSS S3 UPLOAD")
    print("=" * 60)

    print(f"Bucket : {S3_BUCKET_NAME}")
    print(f"Region : {AWS_REGION}")
    print(f"Source : {RAW_DIR}")
    print(f"S3 Path: {S3_RAW_PREFIX}")
    print()

    upload_raw_data()

    print("=" * 60)
    print("UPLOAD COMPLETED")
    print("=" * 60)