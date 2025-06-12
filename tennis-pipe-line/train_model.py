import pandas as pd
import numpy as np
import boto3
import io
import lightgbm as lgb
import optuna
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score

# ================================
# S3設定
# ================================
bucket_name = 'tennis-pipe-line'
train_key = 'data/train.tsv'
model_output_key = 'model/lgb_model.pkl'

# S3からtrain.tsvを読み込み
s3 = boto3.client('s3', region_name='ap-southeast-2')
response = s3.get_object(Bucket=bucket_name, Key=train_key)
csv_body = response['Body'].read()
df_train = pd.read_csv(io.BytesIO(csv_body), sep='\t')

# ================================
# 特徴量
# ================================
features = ['FSW.1', 'WNR.1', 'NPW.1', 'UFE.1', 'ST1.1',
            'FSW.2', 'NPW.2', 'UFE.2', 'SSW.2', 'WNR.2',
            'long_rally_success_1', 'aggressiveness_1']
target = "Result"

# ================================
# データ分割
# ================================
X = df_train[features]
y = df_train[target]

X_tr, X_va2, y_tr, y_va2 = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_tr, y_tr))

# ================================
# Optuna最適化
# ================================
def optimize_params_with_optuna(X, y, folds, n_trials=30):
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
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(0)
                ]
            )

            y_pred = model.predict(X_val_fold)
            y_pred_label = (y_pred > 0.5).astype(int)
            acc = accuracy_score(y_val_fold, y_pred_label)
            scores.append(acc)

        return 1 - np.mean(scores)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study

# ================================
# モデル評価
# ================================
def evaluate_model_cv(X_tr, y_tr, params, folds, X_va2=None, y_va2=None):
    models = []
    val_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(folds):
        print(f"Fold {fold + 1}")
        X_train_fold, X_val_fold = X_tr.iloc[train_idx], X_tr.iloc[val_idx]
        y_train_fold, y_val_fold = y_tr.iloc[train_idx], y_tr.iloc[val_idx]

        dtrain = lgb.Dataset(X_train_fold, y_train_fold)
        dvalid = lgb.Dataset(X_val_fold, y_val_fold)

        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dvalid],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(10)
            ]
        )

        y_pred = model.predict(X_val_fold)
        y_pred_label = (y_pred > 0.5).astype(int)
        acc = accuracy_score(y_val_fold, y_pred_label)
        val_accuracies.append(acc)
        models.append(model)

    print(f"平均CV Accuracy: {np.mean(val_accuracies):.4f}")

    if X_va2 is not None and y_va2 is not None:
        y_pred = models[-1].predict(X_va2)
        y_pred_label = (y_pred > 0.5).astype(int)
        acc = accuracy_score(y_va2, y_pred_label)
        print(f"ホールドアウト検証Accuracy: {acc:.4f}")

    return models[-1]

# ================================
# 実行部分
# ================================
print("Optunaによるハイパーパラメータ最適化中...")
best_params, _ = optimize_params_with_optuna(X_tr, y_tr, folds, n_trials=30)

print("クロスバリデーションでモデル評価中...")
best_model = evaluate_model_cv(X_tr, y_tr, best_params, folds, X_va2, y_va2)

# ================================
# モデル保存（S3へ）
# ================================
joblib.dump(best_model, "lgb_model.pkl")
with open("lgb_model.pkl", "rb") as f:
    s3.upload_fileobj(f, bucket_name, model_output_key)

print("最良モデルをS3に保存しました。")
