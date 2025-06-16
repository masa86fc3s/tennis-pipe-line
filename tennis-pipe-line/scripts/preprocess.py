import pandas as pd
import boto3
import io
from sklearn.preprocessing import LabelEncoder


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
df_train = pd.read_csv(io.BytesIO(csv_body1), sep='\t')

# Testデータの読み込み
response2 = s3.get_object(Bucket=bucket_name, Key=test_key)
csv_body2 = response2['Body'].read()
df_test = pd.read_csv(io.BytesIO(csv_body2), sep='\t')

# 確認
#print(df_train.head())


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

