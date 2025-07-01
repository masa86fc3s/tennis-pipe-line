import sagemaker
from sagemaker.processing import ScriptProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

role = "arn:aws:iam::xxxxxxxxxxxx:role/YourSageMakerRole"  # 自分のIAMロールARNにする
bucket = "tennis-sagemaker"
script_path_in_s3 = "s3://tennis-sagemaker/tennis-pipe-line/train_model.py"

# 処理用インスタンス指定（無料枠なら ml.t3.mediumとか ml.m5.largeあたりが安い）
script_processor = ScriptProcessor(
    image_uri="683313688378.dkr.ecr.ap-northeast-1.amazonaws.com/sagemaker-scikit-learn:1.0-1-cpu-py3",  # scikit-learn用公式コンテナ
    command=["python3"],
    instance_type="ml.m5.large",
    instance_count=1,
    role=role,
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
