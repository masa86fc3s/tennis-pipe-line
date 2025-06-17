import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import joblib
import boto3
import io
import pickle
import yaml
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score

from lightgbm.basic import Booster
from pandas import DataFrame, Series

# ================================
# S3設定
# ================================
bucket_name = 'tennis-pipe-line'
train_key = 'data/train_preprocessed.tsv'
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


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


# 関数定義のあとで呼び出す！
config = load_config()
optuna_cfg = config['optuna_params']
# 数値に変換（必要ならループ化してもOK）
for param in ['lambda_l1', 'lambda_l2']:
    optuna_cfg[param]['low'] = float(optuna_cfg[param]['low'])
    optuna_cfg[param]['high'] = float(optuna_cfg[param]['high'])

# booleanの変換（必要に応じて）
for param in ['lambda_l1', 'lambda_l2']:
    if isinstance(optuna_cfg[param]['log'], str):
        optuna_cfg[param]['log'] = optuna_cfg[param]['log'].lower() == 'true'



def optimize_params_with_optuna(
    X: DataFrame,
    y: Series,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    X_val2: Optional[DataFrame] = None,
    y_val2: Optional[Series] = None,
    n_trials: int = 30
) -> Tuple[dict, optuna.study.Study]:

    optuna_cfg = config['optuna_params']
    base_params = config['model_params']

    def objective(trial: optuna.trial.Trial) -> float:
        params = base_params.copy()

        params['lambda_l1'] = trial.suggest_float(
            'lambda_l1', optuna_cfg['lambda_l1']['low'], optuna_cfg['lambda_l1']['high'], log=optuna_cfg['lambda_l1']['log']
        )
        params['lambda_l2'] = trial.suggest_float(
            'lambda_l2', optuna_cfg['lambda_l2']['low'], optuna_cfg['lambda_l2']['high'], log=optuna_cfg['lambda_l2']['log']
        )
        params['num_leaves'] = trial.suggest_int(
            'num_leaves', optuna_cfg['num_leaves']['low'], optuna_cfg['num_leaves']['high']
        )
        params['feature_fraction'] = trial.suggest_float(
            'feature_fraction', optuna_cfg['feature_fraction']['low'], optuna_cfg['feature_fraction']['high']
        )
        params['bagging_fraction'] = trial.suggest_float(
            'bagging_fraction', optuna_cfg['bagging_fraction']['low'], optuna_cfg['bagging_fraction']['high']
        )
        params['bagging_freq'] = trial.suggest_int(
            'bagging_freq', optuna_cfg['bagging_freq']['low'], optuna_cfg['bagging_freq']['high']
        )
        params['min_child_samples'] = trial.suggest_int(
            'min_child_samples', optuna_cfg['min_child_samples']['low'], optuna_cfg['min_child_samples']['high']
        )

        scores: List[float] = []
        val2_scores: List[float] = []

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

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    return study.best_params, study


# ================================
# モデル評価（CV）
# ================================

def evaluate_model_cv(
    X: DataFrame,                                               #特徴量データ（トレーニング全体）
    y: Series,                                                  #目的変数（ラベル）
    params: dict,                                               #LightGBMモデルのハイパーパラメータ
    folds: List[Tuple[np.ndarray, np.ndarray]],                 #KFoldで生成した学習・検証インデックスのリスト（タプル）
    X_va2: Optional[DataFrame] = None,                          #別検証用の特徴量データ（任意）
    y_va2: Optional[Series] = None                              #別検証用のラベルデータ（任意）
) -> Tuple[List[float], List[float], Booster, List[Booster]]:
    val_accuracies: List[float] = []                            #各foldごとのLightGBMモデルの検証データでの精度（accuracy）のリスト
    base_accuracies: List[float] = []                           #各foldごとの「ベースライン精度」（= 予測せずに一番多いクラスで全部予測）のリスト
    models: List[Booster] = []                                  #各foldごとに学習されたLightGBMモデル（Boosterオブジェクト）のリスト

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
        print(f"交差検証Accuracy: {acc:.4f}")

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
joblib.dump(best_model, "../models/lgb_model.pkl")
with open("../models/lgb_model.pkl", "rb") as f:
    s3.upload_fileobj(f, bucket_name, model_output_key)

print("最良モデルをS3に保存しました。")






