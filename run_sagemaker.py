import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.estimator import Estimator
from sagemaker.workflow.parameters import ParameterString
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.image_uris import retrieve
from sagemaker.inputs import TrainingInput

# --- リージョン指定とセッション初期化 ---
region = "ap-southeast-2"
boto_session = boto3.Session(region_name=region)
session = sagemaker.Session(boto_session=boto_session)

# --- IAMロール設定 ---
role = "arn:aws:iam::216989098479:role/service-role/AmazonSageMaker-ExecutionRole-20250630T164486"

# --- パイプラインパラメータ ---
bucket_param = ParameterString(name="InputBucket", default_value="tennis-pipe-line")
model_output_path = f"s3://{bucket_param.default_value}/models/"
from sagemaker.processing import ProcessingInput

preprocess_inputs = [
    ProcessingInput(  # ← ★ s3_data.yml を使うための input
        source="./tennis-pipe-line/yml",  # ローカルの `yml/` ディレクトリ（s3_data.yml が入っている）
        destination="/opt/ml/processing/input/tennis-pipe-line/yml",
    )
]

# --- 1. 前処理ステップ ---
preprocess_processor = ScriptProcessor(
    image_uri=retrieve(
        framework="sklearn",
        region=region,
        version="0.23-1",
        instance_type="ml.t3.medium",
    ),
    command=["python3"],
    instance_count=1,
    instance_type="ml.t3.medium",
    base_job_name="tennis-preprocess",
    role=role,
    sagemaker_session=session,
)

preprocess_step = ProcessingStep(
    name="PreprocessData",
    processor=preprocess_processor,
    inputs=preprocess_inputs,  # ← ★ ここを追加
    outputs=[
        ProcessingOutput(
            output_name="preprocessed_train",
            source="/opt/ml/processing/output/train_preprocessed.tsv",
            destination=f"s3://{bucket_param.default_value}/data/"
        ),
        ProcessingOutput(
            output_name="preprocessed_test",
            source="/opt/ml/processing/output/test_preprocessed.tsv",
            destination=f"s3://{bucket_param.default_value}/data/"
        ),
    ],
    code="tennis-pipe-line/src/preprocess.py",
)

# --- 2. 学習ステップ ---
# ★汎用Python3.8コンテナを指定
image_uri = sagemaker.image_uris.retrieve(
    framework="pytorch",  # 軽い汎用コンテナとしてPyTorchを利用
    region=region,
    version="1.12.0",     # Python3.8が動くバージョン
    py_version="py38",
    instance_type="ml.m5.large",
    image_scope="training",
)

estimator = Estimator(
    entry_point="train_model.py",
    source_dir="tennis-pipe-line/src",
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    image_uri=image_uri,
    sagemaker_session=session,
    base_job_name="tennis-lgbm-train",
    output_path=model_output_path,
    dependencies=["requirements.txt"],  # ★requirements.txtを指定
)

training_step = TrainingStep(
    name="TrainLightGBMModel",
    estimator=estimator,
    inputs={
        "train": TrainingInput(
            s3_data=preprocess_step.properties.ProcessingOutputConfig.Outputs["preprocessed_train"].S3Output.S3Uri,
            content_type="text/tab-separated-values"
        ),
        "config": TrainingInput(
            s3_data="s3://tennis-sagemaker/tennis-pipe-line/yml",
            content_type="application/x-yaml"
        )
    },
)
# --- 3. パイプライン定義 ---
pipeline = Pipeline(
    name="TennisModelPipeline",
    parameters=[bucket_param],
    steps=[preprocess_step, training_step],
    sagemaker_session=session,
)

# --- パイプライン登録と実行 ---
pipeline.upsert(role_arn=role)
execution = pipeline.start()

print(f"Pipeline {pipeline.name} が実行開始されました。")
