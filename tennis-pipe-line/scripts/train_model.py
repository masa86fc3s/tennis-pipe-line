import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import joblib
import boto3
import io
import pickle
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score

# ================================
# S3設定
# ================================
bucket_name = 'tennis-pipe-line'
train_key = 'data/train_preprocessed.tsv'
model_output_key = 'model/lgb_model.pkl'

s3 = boto3.client('s3', region_name='ap-southeast-2')

# ================================
# データ読み込み（S3から）
# ================================
response = s3.get_object(Bucket=bucket_name, Key=train_key)
csv_body = response['Body'].read()
df_train = pd.read_csv(io.BytesIO(csv_body), sep='\t')

# ================================
# 特徴量・目的変数
# ================================
FEATURES = ['FSW.1', 'WNR.1', 'NPW.1', 'UFE.1', 'ST1.1',
            'FSW.2', 'NPW.2', 'UFE.2', 'SSW.2', 'WNR.2',
            'long_rally_success_1', 'aggressiveness_1']
TARGET = 'Result'

X_train, y_train = df_train[FEATURES].copy(), df_train[TARGET].copy()

# データ分割
X_tr, X_va2, y_tr, y_va2 = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train, shuffle=True
)
X_tr1, X_va1, y_tr1, y_va1 = train_test_split(
    X_tr, y_tr, test_size=0.2, random_state=42, stratify=y_tr, shuffle=True
)

folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_tr, y_tr))

# ================================
# Optunaによるハイパーパラメータチューニング
# ================================
def optimize_params_with_optuna(X, y, folds, X_val2=None, y_val2=None, n_trials=30):
    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'random_state': 42,
            'feature_pre_filter': False,
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 4, 64),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        }

        scores = []
        val2_scores = []

        for train_idx, val_idx in folds:
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            dtrain = lgb.Dataset(X_train_fold, y_train_fold)
            dvalid = lgb.Dataset(X_val_fold, y_val_fold, reference=dtrain)

            model = lgb.train(
                params,
                dtrain,
                valid_sets=[dvalid],
                num_boost_round=1000,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=0)
                ]
            )

            y_pred_val = model.predict(X_val_fold, num_iteration=model.best_iteration)
            acc = accuracy_score(y_val_fold, (y_pred_val > 0.5).astype(int))
            scores.append(acc)

            if X_val2 is not None and y_val2 is not None:
                y_pred_val2 = model.predict(X_val2, num_iteration=model.best_iteration)
                acc_val2 = accuracy_score(y_val2, (y_pred_val2 > 0.5).astype(int))
                val2_scores.append(acc_val2)

        score_cv = np.mean(scores)
        score_val2 = np.mean(val2_scores) if val2_scores else 0
        final_score = 1 - ((score_cv + score_val2) / 2)

        return final_score

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    study.optimize(objective, n_trials=n_trials)

    return study.best_params, study

# ハイパーパラメータ最適化
print("🔍 Optunaでハイパーパラメータ探索中...")
best_params, study = optimize_params_with_optuna(X_tr, y_tr, folds, X_va2, y_va2, n_trials=30)
print("✅ 最適パラメータ:", best_params)

# ================================
# モデル評価（CV）
# ================================
def evaluate_model_cv(X, y, params, folds, X_va2=None, y_va2=None):
    val_accuracies = []
    base_accuracies = []
    models = []

    for fold, (train_idx, val_idx) in enumerate(folds):
        print(f"📂 Fold {fold + 1}/{len(folds)}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        dtrain = lgb.Dataset(X_train, y_train)
        dvalid = lgb.Dataset(X_val, y_val, reference=dtrain)

        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dvalid],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=10)
            ]
        )

        y_pred = model.predict(X_val, num_iteration=model.best_iteration)
        y_pred_label = (y_pred > 0.5).astype(int)

        acc = accuracy_score(y_val, y_pred_label)
        val_accuracies.append(acc)

        y_base_pred = [y_val.mode().values[0]] * len(y_val)
        base_acc = accuracy_score(y_val, y_base_pred)
        base_accuracies.append(base_acc)

        models.append(model)

    print("=== 📊 クロスバリデーション結果 ===")
    print(f"✅ 平均 Accuracy: {np.mean(val_accuracies):.4f}")
    print(f"📉 ベースライン Accuracy: {np.mean(base_accuracies):.4f}")

    if X_va2 is not None and y_va2 is not None:
        print("=== 別検証データ X_va2 の評価 ===")
        y_va_pred = models[-1].predict(X_va2, num_iteration=models[-1].best_iteration)
        y_va_pred_label = (y_va_pred > 0.5).astype(int)
        va_acc = accuracy_score(y_va2, y_va_pred_label)
        print(f"X_va2のAccuracy: {va_acc:.4f}")

    return val_accuracies, base_accuracies, models[-1], models

# モデル学習・評価
_, _, final_model, _ = evaluate_model_cv(X_tr, y_tr, best_params, folds, X_va2, y_va2)

# ================================
# S3にモデル保存
# ================================
model_bytes = io.BytesIO()
pickle.dump(final_model, model_bytes)
model_bytes.seek(0)

s3.upload_fileobj(model_bytes, bucket_name, model_output_key)
print("💾 モデルをS3に保存しました:", model_output_key)
