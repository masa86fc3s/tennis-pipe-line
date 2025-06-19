import pandas as pd
import boto3
import io
import yaml
import os
from sklearn.preprocessing import LabelEncoder


class ConfigLoader:  # （YAML読み込みクラス）
    @staticmethod
    def load_config(path: str) -> dict:
        with open(os.path.abspath(path), "r") as f:
            return yaml.safe_load(f)


class S3Client:  # （S3との入出力操作をまとめるクラス）
    def __init__(self, region_name: str):
        self.client = boto3.client("s3", region_name=region_name)

    def load_tsv(self, bucket: str, key: str) -> pd.DataFrame:
        response = self.client.get_object(Bucket=bucket, Key=key)
        body = response["Body"].read()
        return pd.read_csv(io.BytesIO(body), sep="\t")

    def upload_df_as_tsv(self, df: pd.DataFrame, bucket: str, key: str):
        buffer = io.StringIO()
        df.to_csv(buffer, sep="\t", index=False)
        self.client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())


class DataPreprocessor:  # （前処理・特徴量生成など）
    def __init__(self):
        self.label_encoders = {}

    def encode_label(self, df_train: pd.DataFrame, df_test: pd.DataFrame, col: str):
        encoder = LabelEncoder()
        encoder.fit(pd.concat([df_train[col], df_test[col]]))
        df_train[col] = encoder.transform(df_train[col])
        df_test[col] = encoder.transform(df_test[col])
        self.label_encoders[col] = encoder
        return df_train, df_test

    def drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        drop_cols = [
            "Player1",
            "Player2",
            "Year",
            "ST2.1",
            "ST3.1",
            "ST4.1",
            "ST5.1",
            "ST2.2",
            "ST3.2",
            "ST4.2",
            "ST5.2",
            "FNL.1",
            "FNL.2",
            "TPW.1",
            "TPW.2",
            "BPW.1",
            "BPW.2",
            "BPC.1",
            "BPC.2",
        ]
        return df.drop(drop_cols, axis=1)

    def impute_missing(
        self, df: pd.DataFrame, cols_to_impute, cols_to_zero
    ) -> pd.DataFrame:
        for col in cols_to_impute:
            df[col] = df[col].fillna(df[col].median())
        for col in cols_to_zero:
            df[col] = df[col].fillna(0)
        return df

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["long_rally_success_1"] = df["SSW.1"] / df["SSP.1"]
        df["aggressiveness_1"] = (
            df["ACE.1"] * 1.0
            + df["WNR.1"] * 0.8
            - df["UFE.1"] * 0.7
            - df["DBF.1"] * 0.5
        )
        return df


def main():  # メイン処理（main()関数）
    # 設定読み込み
    yaml_path = os.path.join(os.path.dirname(__file__), "../yml/s3_data.yml")
    config = ConfigLoader.load_config(yaml_path)

    bucket = config["s3"]["bucket_name"]
    region = config["s3"]["region"]
    train_key = config["s3"]["train1_key"]
    test_key = config["s3"]["test1_key"]

    s3_client = S3Client(region)
    df_train = s3_client.load_tsv(bucket, train_key)
    df_test = s3_client.load_tsv(bucket, test_key)

    # 前処理
    processor = DataPreprocessor()
    for col in ["Tournament", "Sex"]:
        df_train, df_test = processor.encode_label(df_train, df_test, col)

    df_train = processor.drop_columns(df_train)
    df_test = processor.drop_columns(df_test)

    cols_to_impute = [
        "WNR.1",
        "UFE.1",
        "WNR.2",
        "UFE.2",
        "NPW.1",
        "NPW.2",
        "NPA.1",
        "NPA.2",
    ]
    cols_to_zero = ["ACE.1", "DBF.1", "ACE.2", "DBF.2"]

    df_train = processor.impute_missing(df_train, cols_to_impute, cols_to_zero)
    df_test = processor.impute_missing(df_test, cols_to_impute, cols_to_zero)

    df_train = processor.create_features(df_train)
    df_test = processor.create_features(df_test)

    # アップロード
    s3_client.upload_df_as_tsv(df_train, bucket, "data/train_preprocessed.tsv")
    s3_client.upload_df_as_tsv(df_test, bucket, "data/test_preprocessed.tsv")

    print("前処理済みのtrainおよびtestデータをS3に保存しました。")


if __name__ == "__main__":
    main()
