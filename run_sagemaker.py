import sagemaker
from sagemaker.processing import ScriptProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
# もしくは明示的に
import boto3
from sagemaker import Session

boto_session = boto3.session.Session(region_name='ap-southeast-2')  # ←東京ならこれ
sagemaker_session = Session(boto_session=boto_session)


role = "arn:aws:iam::216989098479:user/masa86fc3s"  # 自分のIAMロールARNにする
bucket = "tennis-sagemaker"
script_path_in_s3 = "s3://tennis-sagemaker/tennis-pipe-line/pipeline.py"

from sagemaker import image_uris

image_uri = image_uris.retrieve(
    framework='scikit-learn',
    region='ap-southeast-2',
    version='1.0-1',
    py_version='py3',
    instance_type="ml.t3.medium",
)
print(image_uri)

script_processor = ScriptProcessor(
    image_uri=image_uri,  # ここでimage_uris.retrieveの結果を使う
    command=["python3"],
    instance_type="ml.t3.medium",
    instance_count=1,
    role=role,
    sagemaker_session=sagemaker_session,
)

script_processor.run(
    code=script_path_in_s3,
    inputs=[
        ProcessingInput(
            source=f"s3://{bucket}/tennis-pipe-line/",
            destination="/opt/ml/processing/input",
        )
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/output",
            destination=f"s3://{bucket}/output/",
        )
    ],
)
