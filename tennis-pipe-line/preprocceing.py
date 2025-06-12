import pandas as pd
import boto3
import io

# S3設定
bucket_name = 'tennis-pipe-line'
train_key = 'data/train.tsv'
test_key = 'data/test.tsv'

# ===============================
# S3からTSVを読み込む
# ===============================
s3 = boto3.client('s3', region_name='ap-southeast-2')

# Trainデータの読み込み
response1 = s3.get_object(Bucket=bucket_name, Key=train_key)
csv_body1 = response1['Body'].read()
train_df = pd.read_csv(io.BytesIO(csv_body1), sep='\t')

# Testデータの読み込み
response2 = s3.get_object(Bucket=bucket_name, Key=test_key)
csv_body2 = response2['Body'].read()
test_df = pd.read_csv(io.BytesIO(csv_body2), sep='\t')

# 確認
print(train_df.head())
