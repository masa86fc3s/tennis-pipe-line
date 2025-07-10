import os
import boto3


def upload_directory(local_directory, bucket_name):
    s3 = boto3.client("s3", region_name="ap-southeast-2")

    for root, dirs, files in os.walk(local_directory):
        for file in files:
            # フォルダ名によるスキップ
            skip_folders = [
                ".git",
                "__pycache__",
                ".mypy_cache",
                ".pytest_cache",
                "sagemaker-env",
            ]
            if any(skip in root for skip in skip_folders):
                continue

            # ファイル拡張子によるスキップ
            skip_extensions = [".pkl", ".csv", ".md"]
            if any(file.endswith(ext) for ext in skip_extensions):
                continue

            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_directory)
            s3_path = os.path.join(relative_path).replace("\\", "/")

            print(f"Uploading {local_path} to s3://{bucket_name}/{s3_path}")
            s3.upload_file(local_path, bucket_name, s3_path)


if __name__ == "__main__":
    local_directory = "./"  # カレントディレクトリ
    bucket_name = "tennis-sagemaker"  # ←自分のバケット名
    # = "tennis-pipe-line/"   # ←S3内での保存先フォルダ名的なもの

    upload_directory(local_directory, bucket_name)
