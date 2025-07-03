import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.processing import ScriptProcessor
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.parameters import ParameterString
from sagemaker.processing import ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.image_uris import retrieve

# --- リージョン指定とセッション初期化 ---
region = "ap-southeast-2"
boto_session = boto3.Session(region_name=region)
session = sagemaker.Session(boto_session=boto_session)

# --- IAMロール設定 ---
role = "arn:aws:iam::216989098479:role/service-role/AmazonSageMaker-ExecutionRole-20250630T164486"

# --- パイプラインパラメータ ---
bucket_param = ParameterString(name="InputBucket", default_value="tennis-pipe-line")
model_output_path = f"s3://{bucket_param.default_value}/models/"

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
sklearn_estimator = SKLearn(
    entry_point="tennis-pipe-line/src/train_model.py",
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",  # ← 修正ポイント！！
    framework_version="0.23-1",
    sagemaker_session=session,
    base_job_name="tennis-lgbm-train",
    output_path=model_output_path,
)

training_step = TrainingStep(
    name="TrainLightGBMModel",
    estimator=sklearn_estimator,
    inputs={
        "train": sagemaker.inputs.TrainingInput(
            s3_data=preprocess_step.properties.ProcessingOutputConfig.Outputs["preprocessed_train"].S3Output.S3Uri,
            content_type="text/csv",  # 実際のデータ形式に合わせて。例えばcsv、parquet、など
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
