# scripts/evaluate_model.py
import pandas as pd
import numpy as np
import boto3
import io
import joblib
from sklearn.metrics import accuracy_score, classification_report

bucket_name = 'tennis-pipe-line'
test_key = 'data/test.tsv'
model_key = 'model/lgb_model.pkl'

# S3クライアント
s3 = boto3.client('s3', region_name='ap-southeast-2')

# モデル読み込み
response = s3.get_object(Bucket=bucket_name, Key=model_key)
model = joblib.load(io.BytesIO(response['Body'].read()))

# テストデータ読み込み
response = s3.get_object(Bucket=bucket_name, Key=test_key)
df_test = pd.read_csv(io.BytesIO(response['Body'].read()), sep='\t')

features = ['FSW.1', 'WNR.1', 'NPW.1', 'UFE.1', 'ST1.1',
            'FSW.2', 'NPW.2', 'UFE.2', 'SSW.2', 'WNR.2',
            'long_rally_success_1', 'aggressiveness_1']
target = "Result"

X_test = df_test[features]
y_test = df_test[target]

# 予測と評価
y_pred = model.predict(X_test)
y_pred_label = (y_pred > 0.5).astype(int)

print(f"Accuracy: {accuracy_score(y_test, y_pred_label):.4f}")
print(classification_report(y_test, y_pred_label))
