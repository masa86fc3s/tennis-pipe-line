# scripts/predict_model.py（またはevaluate_model.pyを置き換えてもOK）
import pandas as pd
import boto3
import io
import joblib
import os
import yaml

# YAMLファイルの絶対パスを取得して読み込み
yaml_path = os.path.join(os.path.dirname(__file__), "../yaml/s3_data.yaml")
with open(os.path.abspath(yaml_path), "r") as f:
    config = yaml.safe_load(f)

bucket_name = config["s3"]["bucket_name"]
test_key = config["s3"]["test_key"]
model_key = config["s3"]["model_key"]
region = config["s3"]["region"]

features = config["features"]["columns"]

s3 = boto3.client("s3", region_name=region)

# モデル読み込み
response = s3.get_object(Bucket=bucket_name, Key=model_key)
model = joblib.load(io.BytesIO(response["Body"].read()))
print(f"モデルの型: {type(model)}")  # 出力例: <class 'lightgbm.sklearn.LGBMClassifier'>

# テストデータ読み込み
response = s3.get_object(Bucket=bucket_name, Key=test_key)
df_test = pd.read_csv(io.BytesIO(response["Body"].read()), sep="\t")

X_test = df_test[features]

# 予測（LightGBMモデルの出力に合わせて二値分類の閾値0.5で変換）
y_pred = model.predict(X_test)
if y_pred.ndim > 1 and y_pred.shape[1] > 1:  # 確率配列なら二値に変換
    y_pred_label = (y_pred[:, 1] > 0.5).astype(int)
else:
    y_pred_label = (y_pred > 0.5).astype(int)

# 提出用DataFrame作成
submission = pd.DataFrame(
    {"id": df_test["id"], "Result": y_pred_label}  # カラム名は適宜確認
)

# CSV保存（ローカルに保存したい場合）
submission.to_csv("submission.csv", index=False, header=False)

print("予測完了、submission.csvを出力しました。")


# s3に保存したいときは以下のコードを実行する
""""
# ===============================
# 提出用ファイルもS3にアップロード
# ===============================
csv_buffer_test = io.StringIO()
submission.to_csv(csv_buffer_test, sep='\t', index=False)

s3.put_object(Bucket=bucket_name, Key="data/submission.csv", Body=csv_buffer_test.getvalue())

print("予測完了、submission.csvをS3に保存しました。")
"""
