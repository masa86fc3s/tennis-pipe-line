import boto3
from sagemaker import Session, image_uris
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput

# 必要情報
region = "ap-southeast-2"
role = "arn:aws:iam::216989098479:role/service-role/AmazonSageMaker-ExecutionRole-20250630T164486"
bucket = "tennis-sagemaker"

boto_session = boto3.session.Session(region_name=region)
sagemaker_session = Session(boto_session=boto_session)

# Dockerイメージ (scikit-learn)
image_uri = image_uris.retrieve(
    framework='sklearn',
    region=region,
    version='1.0-1',
    py_version='py3'
)
print(image_uri)

# ScriptProcessor 定義
script_processor = ScriptProcessor(
    image_uri=image_uri,
    command=["python3"],
    instance_type="ml.t3.medium",
    instance_count=1,
    role=role,
    sagemaker_session=sagemaker_session,
)

# ジョブ実行
script_processor.run(
    code="tennis-pipe-line/sagemaker-pipeline.py",  # ローカル相対パス
    inputs=[
        ProcessingInput(
            source=f"s3://{bucket}/tennis-pipe-line/",
            destination="/opt/ml/processing/input/tennis-pipe-line",
        ),
        ProcessingInput(
            source=f"s3://{bucket}/requirements.txt",
            destination="/opt/ml/processing/input/",
        ),
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/output/",
            destination=f"s3://{bucket}/output/",
        ),
    ],
)


