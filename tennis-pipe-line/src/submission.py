import pandas as pd
import boto3
import io
import joblib
import os
import yaml


class ModelPredictor:
    def __init__(self, yaml_path: str):
        # YAML読み込み
        with open(os.path.abspath(yaml_path), "r") as f:
            config = yaml.safe_load(f)

        self.bucket_name = config["s3"]["bucket_name"]
        self.test_key = config["s3"]["test_key"]
        self.model_key = config["s3"]["model_key"]
        self.region = config["s3"]["region"]
        self.features = config["features"]["columns"]

        self.s3 = boto3.client("s3", region_name=self.region)
        self.model = None
        self.df_test = None
        self.predictions = None
        self.submission = None

    def load_model_from_s3(self):
        response = self.s3.get_object(Bucket=self.bucket_name, Key=self.model_key)
        self.model = joblib.load(io.BytesIO(response["Body"].read()))
        print(f"モデルをロードしました: {type(self.model)}")

    def load_test_data_from_s3(self):
        response = self.s3.get_object(Bucket=self.bucket_name, Key=self.test_key)
        self.df_test = pd.read_csv(io.BytesIO(response["Body"].read()), sep="\t")
        print(f"テストデータを読み込みました: {self.df_test.shape}")

    def predict(self):
        if self.model is None or self.df_test is None:
            raise ValueError("モデルまたはテストデータが未ロードです。")

        X_test = self.df_test[self.features]
        y_pred = self.model.predict(X_test)

        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            y_pred_label = (y_pred[:, 1] > 0.5).astype(int)
        else:
            y_pred_label = (y_pred > 0.5).astype(int)

        self.predictions = y_pred_label

        self.submission = pd.DataFrame(
            {"id": self.df_test["id"], "Result": self.predictions}
        )
        print("予測が完了しました。")

    def save_submission(
        self,
        local_path: str = "submission.csv",
        upload_to_s3: bool = False,
        s3_key: str = "data/submission.csv",
    ):
        # ローカル保存
        self.submission.to_csv(local_path, index=False, header=False)
        print(f"submission.csv をローカルに保存しました: {local_path}")

        # S3保存（必要な場合）
        if upload_to_s3:
            csv_buffer = io.StringIO()
            self.submission.to_csv(csv_buffer, sep="\t", index=False)
            self.s3.put_object(
                Bucket=self.bucket_name, Key=s3_key, Body=csv_buffer.getvalue()
            )
            print(
                f"submission.csv を S3 に保存しました: s3://{self.bucket_name}/{s3_key}"
            )


if __name__ == "__main__":
    yaml_path = os.path.join(os.path.dirname(__file__), "../yml/s3_data.yml")
    predictor = ModelPredictor(yaml_path)
    predictor.load_model_from_s3()
    predictor.load_test_data_from_s3()
    predictor.predict()
    predictor.save_submission(local_path="submission.csv", upload_to_s3=False)
