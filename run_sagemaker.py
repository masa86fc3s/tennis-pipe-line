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
            destination=f"s3://{bucket_param.default_value}/data/",
        ),
        ProcessingOutput(
            output_name="preprocessed_test",
            source="/opt/ml/processing/output/test_preprocessed.tsv",
            destination=f"s3://{bucket_param.default_value}/data/",
        ),
    ],
    code="tennis-pipe-line/src/preprocess.py",
)

# --- 2. 学習ステップ ---
# ★汎用Python3.8コンテナを指定
image_uri = "216989098479.dkr.ecr.ap-southeast-2.amazonaws.com/tennis-pipeline:latest"

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
            s3_data=preprocess_step.properties.ProcessingOutputConfig.Outputs[
                "preprocessed_train"
            ].S3Output.S3Uri,
            content_type="text/tab-separated-values",
        ),
        "config": TrainingInput(
            s3_data="s3://tennis-sagemaker/tennis-pipe-line/yml",
            content_type="application/x-yaml",
        ),
    },
)
# --- 3. 推論ステップ (ProcessingStepに変更) ---
predict_processor = ScriptProcessor(
    image_uri=image_uri,  # 学習時と同じコンテナ
    command=["python3"],
    instance_count=1,
    instance_type="ml.t3.medium",  # ✅ 推論だけ軽量化,
    base_job_name="tennis-submission",
    role=role,
    sagemaker_session=session,
)

predict_step = ProcessingStep(
    name="RunSubmission",
    processor=predict_processor,
    depends_on=[training_step],  # ★ 学習が終わってから実行する
    code="tennis-pipe-line/src/submission.py",
    inputs=[
        ProcessingInput(
            source="./tennis-pipe-line/yml",  # s3_data.yml, features.yml を含む
            destination="/opt/ml/processing/input/yml",
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="submission_csv",
            source="/opt/ml/processing/output/submission.csv",
            destination=f"s3://{bucket_param.default_value}/tennis-pipe-line/output/",
        )
    ],
)


# --- パイプライン定義（3ステップに拡張） ---
pipeline = Pipeline(
    name="TennisModelPipeline",
    parameters=[bucket_param],
    steps=[
        preprocess_step,  # 前処理
        training_step,  # 学習
        predict_step,  # ← ★ここが追加
    ],
    sagemaker_session=session,
)

# --- パイプライン登録と実行 ---
pipeline.upsert(role_arn=role)
execution = pipeline.start()

print(f"Pipeline {pipeline.name} が実行開始されました。")
