import pandas as pd
import boto3
import io
from sklearn.preprocessing import LabelEncoder
import yaml

with open("s3_data.yaml", "r") as f:
    config = yaml.safe_load(f)

bucket_name = config["s3"]["bucket_name"]
train_key = config["s3"]["train1_key"]
test_key = config["s3"]["test1_key"]
region = config["s3"]["region"]

s3 = boto3.client("s3", region_name=region)

def load_tsv_from_s3(bucket: str, key: str) -> pd.DataFrame:
    response = s3.get_object(Bucket=bucket, Key=key)
    body = response['Body'].read()
    return pd.read_csv(io.BytesIO(body), sep='\t')

df_train = load_tsv_from_s3(bucket_name, train_key)
df_test = load_tsv_from_s3(bucket_name, test_key)

# ===============================
# 前処理
# ===============================

# Tournamentのラベルエンコード
encoder = LabelEncoder()
encoder.fit(pd.concat([df_train["Tournament"], df_test["Tournament"]]))
df_train["Tournament"] = encoder.transform(df_train["Tournament"])
df_test["Tournament"] = encoder.transform(df_test["Tournament"])

# Sexのラベルエンコード
encoder = LabelEncoder()
encoder.fit(pd.concat([df_train["Sex"], df_test["Sex"]]))
df_train["Sex"] = encoder.transform(df_train["Sex"])
df_test["Sex"] = encoder.transform(df_test["Sex"])



#print("変換結果 : ", encoder.classes_)    # 左から順に0, 1

#原因変数を削除
df_train = df_train.drop(["Player1","Player2","Year","ST2.1","ST3.1","ST4.1","ST5.1","ST2.2","ST3.2","ST4.2","ST5.2","FNL.1","FNL.2","TPW.1","TPW.2","BPW.1", "BPW.2", "BPC.1", "BPC.2"],axis=1)
df_test = df_test.drop(["Player1","Player2","Year","ST2.1","ST3.1","ST4.1","ST5.1","ST2.2","ST3.2","ST4.2","ST5.2","FNL.1","FNL.2","TPW.1","TPW.2","BPW.1", "BPW.2", "BPC.1", "BPC.2"],axis=1)


# ===============================
# 欠損値補完
# ===============================

# 対象カラムリスト
cols_to_impute = ['WNR.1', 'UFE.1', 'WNR.2', 'UFE.2', 'NPW.1', 'NPW.2', 'NPA.1', 'NPA.2']
cols_to_zero = ['ACE.1','DBF.1','ACE.2','DBF.2']
# 各列に対して中央値で補完
for col in cols_to_impute:
    median_val = df_train[col].median()
    df_train[col] = df_train[col].fillna(median_val)
for cols in cols_to_zero:
    df_train[cols] = df_train[cols].fillna(0)
# testデータも同様に補完
for col in cols_to_impute:
    median_val = df_test[col].median()
    df_test[col] = df_test[col].fillna(median_val)
for col in cols_to_zero:
    df_test[col] = df_test[col].fillna(0)

# ===============================
# 特徴量生成
# ===============================
df_train['long_rally_success_1'] = df_train['SSW.1'] / df_train['SSP.1']
df_test['long_rally_success_1'] = df_test['SSW.1'] / df_test['SSP.1']
df_train["aggressiveness_1"] = (
    df_train["ACE.1"] * 1.0 +
    df_train["WNR.1"] * 0.8 -
    df_train["UFE.1"] * 0.7 -
    df_train["DBF.1"] * 0.5
)
df_test["aggressiveness_1"] = (
    df_test["ACE.1"] * 1.0 +
    df_test["WNR.1"] * 0.8 -
    df_test["UFE.1"] * 0.7 -
    df_test["DBF.1"] * 0.5
) 


# S3クライアントの作成
# 共通のS3クライアント
s3 = boto3.client('s3', region_name='ap-southeast-2')


# データフレームをメモリ上にCSV（TSV）形式で保存
csv_buffer = io.StringIO()
df_train.to_csv(csv_buffer, sep='\t', index=False)

# S3にアップロード
bucket_name = "tennis-pipe-line"
region_name = 'ap-southeast-2'  # 東京リージョンの場合（実際のリージョンを確認してください）
s3_key = "data/train_preprocessed.tsv"
url = f"https://{bucket_name}.s3.{region_name}.amazonaws.com/{s3_key}"


s3.put_object(Bucket=bucket_name, Key=s3_key, Body=csv_buffer.getvalue())
# ===============================
# TestデータもS3にアップロード
# ===============================
csv_buffer_test = io.StringIO()
df_test.to_csv(csv_buffer_test, sep='\t', index=False)

s3.put_object(Bucket=bucket_name, Key="data/test_preprocessed.tsv", Body=csv_buffer_test.getvalue())

print("前処理済みのtrainおよびtestデータをS3に保存しました。")
