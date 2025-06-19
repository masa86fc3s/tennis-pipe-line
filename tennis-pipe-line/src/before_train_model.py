# ==============================
# 標準ライブラリ
# ==============================
import os
import io

# ==============================
# サードパーティライブラリ
# ==============================
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from lightgbm.basic import Booster
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
import joblib
import yaml
import boto3

# ==============================
# 型ヒント・補助
# ==============================
from typing import Tuple, List, Optional
from pandas import DataFrame, Series
from numpy.typing import NDArray
from typing import BinaryIO  # 追加する

# YAMLファイルの絶対パスを取得して読み込み
yaml_path = os.path.join(os.path.dirname(__file__), "../yaml/s3_data.yaml")
with open(os.path.abspath(yaml_path), "r") as f:
    config = yaml.safe_load(f)

bucket_name = config["s3"]["bucket_name"]
train_key = config["s3"]["train_key"]
model_output_key = config["s3"]["model_output_key"]
region = config["s3"]["region"]

features = config["features"]["columns"]
target = config["target"]

s3 = boto3.client("s3", region_name=region)

response = s3.get_object(Bucket=bucket_name, Key=train_key)
csv_body = response["Body"].read()
df_train = pd.read_csv(io.BytesIO(csv_body), sep="\t")

X = df_train[features]
y = df_train[target]

X_tr, X_va2, y_tr, y_va2 = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

folds = list(
    StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_tr, y_tr)
)


def load_config(config_path: str = "../yaml/config.yaml") -> dict:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, config_path)
    with open(full_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


config = load_config()
optuna_cfg = config["optuna_params"]

for param in ["lambda_l1", "lambda_l2"]:
    optuna_cfg[param]["low"] = float(optuna_cfg[param]["low"])
    optuna_cfg[param]["high"] = float(optuna_cfg[param]["high"])

for param in ["lambda_l1", "lambda_l2"]:
    if isinstance(optuna_cfg[param]["log"], str):
        optuna_cfg[param]["log"] = optuna_cfg[param]["log"].lower() == "true"


def optimize_params_with_optuna(
    X: DataFrame,
    y: Series,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    X_val2: Optional[DataFrame] = None,
    y_val2: Optional[Series] = None,
    n_trials: int = 30,
) -> Tuple[dict, optuna.study.Study]:

    optuna_cfg = config["optuna_params"]
    base_params = config["model_params"]

    def objective(trial: optuna.trial.Trial) -> float:
        params = base_params.copy()

        params["lambda_l1"] = trial.suggest_float(
            "lambda_l1",
            optuna_cfg["lambda_l1"]["low"],
            optuna_cfg["lambda_l1"]["high"],
            log=optuna_cfg["lambda_l1"]["log"],
        )
        params["lambda_l2"] = trial.suggest_float(
            "lambda_l2",
            optuna_cfg["lambda_l2"]["low"],
            optuna_cfg["lambda_l2"]["high"],
            log=optuna_cfg["lambda_l2"]["log"],
        )
        params["num_leaves"] = trial.suggest_int(
            "num_leaves",
            optuna_cfg["num_leaves"]["low"],
            optuna_cfg["num_leaves"]["high"],
        )
        params["feature_fraction"] = trial.suggest_float(
            "feature_fraction",
            optuna_cfg["feature_fraction"]["low"],
            optuna_cfg["feature_fraction"]["high"],
        )
        params["bagging_fraction"] = trial.suggest_float(
            "bagging_fraction",
            optuna_cfg["bagging_fraction"]["low"],
            optuna_cfg["bagging_fraction"]["high"],
        )
        params["bagging_freq"] = trial.suggest_int(
            "bagging_freq",
            optuna_cfg["bagging_freq"]["low"],
            optuna_cfg["bagging_freq"]["high"],
        )
        params["min_child_samples"] = trial.suggest_int(
            "min_child_samples",
            optuna_cfg["min_child_samples"]["low"],
            optuna_cfg["min_child_samples"]["high"],
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
                    lgb.log_evaluation(period=0),
                ],
            )

            y_pred_val: NDArray[np.float64] = np.array(
                model.predict(X_val_fold, num_iteration=model.best_iteration)
            )
            acc = accuracy_score(y_val_fold, (y_pred_val > 0.5).astype(int))
            scores.append(acc)

            if X_val2 is not None and y_val2 is not None:
                y_pred_val2: NDArray[np.float64] = np.array(
                    model.predict(X_val2, num_iteration=model.best_iteration)
                )
                acc_val2 = accuracy_score(y_val2, (y_pred_val2 > 0.5).astype(int))
                val2_scores.append(acc_val2)

        score_cv: float = float(np.mean(scores))
        score_val2: float = float(np.mean(val2_scores)) if val2_scores else 0.0
        final_score: float = 1 - ((score_cv + score_val2) / 2)
        return final_score

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    return study.best_params, study


def evaluate_model_cv(
    X: DataFrame,
    y: Series,
    params: dict,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    X_va2: Optional[DataFrame] = None,
    y_va2: Optional[Series] = None,
) -> Tuple[List[float], List[float], Booster, List[Booster]]:
    val_accuracies: List[float] = []
    base_accuracies: List[float] = []
    models: List[Booster] = []

    for fold, (train_idx, val_idx) in enumerate(folds):
        print(f"Fold {fold + 1}")
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        dtrain = lgb.Dataset(X_train_fold, y_train_fold)
        dvalid = lgb.Dataset(X_val_fold, y_val_fold)

        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dvalid],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(10)],
        )

        y_pred: NDArray[np.float64] = np.array(model.predict(X_val_fold))
        y_pred_label = (y_pred > 0.5).astype(int)
        acc = accuracy_score(y_val_fold, y_pred_label)

        val_accuracies.append(acc)
        models.append(model)

    print(f"平均CV Accuracy: {np.mean(val_accuracies):.4f}")

    if X_va2 is not None and y_va2 is not None:
        y_preds = np.mean([model.predict(X_va2) for model in models], axis=0)
        y_pred_label = (y_preds > 0.5).astype(int)
        acc = accuracy_score(y_va2, y_pred_label)
        print(f"アンサンブルモデルの検証Accuracy: {acc:.4f}")

    return val_accuracies, base_accuracies, models[0], models


print("Optunaによるハイパーパラメータ最適化中...")
best_params, _ = optimize_params_with_optuna(X_tr, y_tr, folds, n_trials=30)

print("クロスバリデーションでモデル評価中...")
val_accuracies, base_accuracies, best_model, models = evaluate_model_cv(
    X_tr, y_tr, best_params, folds, X_va2, y_va2
)

best_index = int(np.argmax(val_accuracies))
best_model = models[best_index]

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "../models/lgb_model.pkl")

joblib.dump(best_model, model_path)


model_file: BinaryIO
with open(model_path, "rb") as model_file:
    s3.upload_fileobj(model_file, bucket_name, model_output_key)

print("最良モデルをS3に保存しました。")
