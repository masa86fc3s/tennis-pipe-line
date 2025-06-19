import boto3
import io
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# S3クライアント
s3 = boto3.client("s3", region_name="ap-southeast-2")


# S3からモデル読み込み
def load_model_from_s3(bucket, key):
    response = s3.get_object(Bucket=bucket, Key=key)
    body = response["Body"].read()
    return pickle.load(io.BytesIO(body))


# S3からデータ読み込み
def load_data_from_s3(bucket, key):
    response = s3.get_object(Bucket=bucket, Key=key)
    body = response["Body"].read()
    return pd.read_csv(io.BytesIO(body), sep="\t")


# 設定
BUCKET_NAME = "tennis-pipe-line"
MODEL_KEY = "model/lgb_model.pkl"
DATA_KEY = "data/train_preprocessed.tsv"

# モデル・データ読み込み
model = load_model_from_s3(BUCKET_NAME, MODEL_KEY)
df = load_data_from_s3(BUCKET_NAME, DATA_KEY)

# 特徴量・ターゲット
FEATURES = [
    "FSW.1",
    "WNR.1",
    "NPW.1",
    "UFE.1",
    "ST1.1",
    "FSW.2",
    "NPW.2",
    "UFE.2",
    "SSW.2",
    "WNR.2",
    "long_rally_success_1",
    "aggressiveness_1",
]
TARGET = "Result"

X = df[FEATURES]
y = df[TARGET]


def evaluate_model(X, y, params, folds, X_va2=None, y_va2=None):
    val_accuracies = []
    base_accuracies = []
    models = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        dtrain = lgb.Dataset(X_train, y_train)
        dvalid = lgb.Dataset(X_val, y_val, reference=dtrain)

        model_fold = lgb.train(
            params,
            dtrain,
            valid_sets=[dvalid],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )

        y_val_pred = model_fold.predict(
            X_val, num_iteration=model_fold.best_iteration)
        y_val_pred_label = (y_val_pred > 0.5).astype(int)
        acc = accuracy_score(y_val, y_val_pred_label)
        val_accuracies.append(acc)

        baseline_acc = max(y_val.mean(), 1 - y_val.mean())
        base_accuracies.append(baseline_acc)

        models.append(model_fold)

    print("=== クロスバリデーション平均結果 ===")
    print(f"検証データの平均 Accuracy: {np.mean(val_accuracies):.4f}")
    print(f"ベースライン検証データの平均 Accuracy: {np.mean(base_accuracies):.4f}")

    if X_va2 is not None and y_va2 is not None:
        print("=== 別検証データ X_va2 の評価 ===")
        y_va_pred = models[-1].predict(X_va2,
                                       num_iteration=models[-1].best_iteration)
        y_va_pred_label = (y_va_pred > 0.5).astype(int)
        va_acc = accuracy_score(y_va2, y_va_pred_label)
        print(f"X_va2のAccuracy: {va_acc:.4f}")

    return val_accuracies, base_accuracies, models[-1], models


# 予測ラベルの作成
y_pred = model.predict(X, num_iteration=model.best_iteration)
y_pred_label = (y_pred > 0.5).astype(int)

# 精度計算と表示
accuracy = accuracy_score(y, y_pred_label)
print(f"Loaded model Accuracy on the entire dataset: {accuracy:.4f}")

# 混同行列表示
cm = confusion_matrix(y, y_pred_label)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
