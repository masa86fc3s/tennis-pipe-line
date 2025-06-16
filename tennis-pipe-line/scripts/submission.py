# scripts/predict_model.py（またはevaluate_model.pyを置き換えてもOK）
import pandas as pd
import boto3
import io
import joblib

bucket_name = 'tennis-pipe-line'
test_key = 'data/test_preprocessed.tsv'
model_key = 'model/lgb_model.pkl'

# S3クライアント
s3 = boto3.client('s3', region_name='ap-southeast-2')

# モデル読み込み
response = s3.get_object(Bucket=bucket_name, Key=model_key)
model = joblib.load(io.BytesIO(response['Body'].read()))

# テストデータ読み込み
response = s3.get_object(Bucket=bucket_name, Key=test_key)
df_test = pd.read_csv(io.BytesIO(response['Body'].read()), sep='\t')

# 予測に使う特徴量リスト
features = ['FSW.1', 'WNR.1', 'NPW.1', 'UFE.1', 'ST1.1',
            'FSW.2', 'NPW.2', 'UFE.2', 'SSW.2', 'WNR.2',
            'long_rally_success_1', 'aggressiveness_1']

X_test = df_test[features]

# 予測（LightGBMモデルの出力に合わせて二値分類の閾値0.5で変換）
y_pred = model.predict(X_test)
if y_pred.ndim > 1 and y_pred.shape[1] > 1:  # 確率配列なら二値に変換
    y_pred_label = (y_pred[:, 1] > 0.5).astype(int)
else:
    y_pred_label = (y_pred > 0.5).astype(int)

# 提出用DataFrame作成
submission = pd.DataFrame({
    'id': df_test['id'],  # カラム名は適宜確認
    'Result': y_pred_label
})

# CSV保存（ローカルに保存したい場合）
submission.to_csv("submission.csv", index=False)

print("予測完了、submission.csvを出力しました。")



#s3に保存したいときは以下のコードを実行する
""""
# ===============================
# 提出用ファイルもS3にアップロード
# ===============================
csv_buffer_test = io.StringIO()
submission.to_csv(csv_buffer_test, sep='\t', index=False)

s3.put_object(Bucket=bucket_name, Key="data/submission.csv", Body=csv_buffer_test.getvalue())

print("予測完了、submission.csvをS3に保存しました。")
"""