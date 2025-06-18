# ==============================
# 標準ライブラリ
# ==============================
import os
import io
import pickle

# ==============================
# サードパーティライブラリ
# ==============================
# データ処理
import pandas as pd
import numpy as np
# 機械学習・モデル
import lightgbm as lgb
import optuna
from lightgbm.basic import Booster
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
# ファイル保存・読み込み
import joblib
import yaml
# AWS（S3操作）
import boto3
# ==============================
# 型ヒント・補助
# ==============================
from typing import Tuple, List, Optional
from pandas import DataFrame, Series


class LightGBMPipeline:
    def __init__(self, config_path: str = "../yaml/s3_data.yaml"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(base_dir, config_path)
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)

        self.bucket_name = config["s3"]["bucket_name"]
        self.train_key = config["s3"]["train_key"]
        self.model_output_key = config["s3"]["model_output_key"]
        self.region = config["s3"]["region"]

        self.features = config["features"]["columns"]
        self.target = config["target"]
        self.s3 = boto3.client("s3", region_name=self.region)

        self.df_train = self._load_data_from_s3()
        self.X = self.df_train[self.features]
        self.y = self.df_train[self.target]
        self.X_tr, self.X_va2, self.y_tr, self.y_va2 = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y)
        self.folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(self.X_tr, self.y_tr))

        self.config = self._load_local_config()
        self._convert_optuna_config()

    def _load_data_from_s3(self) -> DataFrame:
        response = self.s3.get_object(Bucket=self.bucket_name, Key=self.train_key)
        csv_body = response["Body"].read()
        return pd.read_csv(io.BytesIO(csv_body), sep="\t")

    def _load_local_config(self, config_path: str = "../yaml/config.yaml") -> dict:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_dir, config_path)
        with open(full_path, "r") as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    def _convert_optuna_config(self):
        optuna_cfg = self.config['optuna_params']
        for param in ['lambda_l1', 'lambda_l2']:
            optuna_cfg[param]['low'] = float(optuna_cfg[param]['low'])
            optuna_cfg[param]['high'] = float(optuna_cfg[param]['high'])
            if isinstance(optuna_cfg[param]['log'], str):
                optuna_cfg[param]['log'] = optuna_cfg[param]['log'].lower() == 'true'

    def optimize_params(self, n_trials: int = 30) -> Tuple[dict, optuna.study.Study]:
        optuna_cfg = self.config['optuna_params']
        base_params = self.config['model_params']

        def objective(trial: optuna.trial.Trial) -> float:
            params = base_params.copy()
            params['lambda_l1'] = trial.suggest_float('lambda_l1', **optuna_cfg['lambda_l1'])
            params['lambda_l2'] = trial.suggest_float('lambda_l2', **optuna_cfg['lambda_l2'])
            params['num_leaves'] = trial.suggest_int('num_leaves', **optuna_cfg['num_leaves'])
            params['feature_fraction'] = trial.suggest_float('feature_fraction', **optuna_cfg['feature_fraction'])
            params['bagging_fraction'] = trial.suggest_float('bagging_fraction', **optuna_cfg['bagging_fraction'])
            params['bagging_freq'] = trial.suggest_int('bagging_freq', **optuna_cfg['bagging_freq'])
            params['min_child_samples'] = trial.suggest_int('min_child_samples', **optuna_cfg['min_child_samples'])

            scores = []
            val2_scores = []
            for train_idx, val_idx in self.folds:
                X_train, X_val = self.X_tr.iloc[train_idx], self.X_tr.iloc[val_idx]
                y_train, y_val = self.y_tr.iloc[train_idx], self.y_tr.iloc[val_idx]
                dtrain = lgb.Dataset(X_train, y_train)
                dvalid = lgb.Dataset(X_val, y_val, reference=dtrain)
                model = lgb.train(params, dtrain, valid_sets=[dvalid], num_boost_round=1000,
                                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
                y_pred = model.predict(X_val)
                scores.append(accuracy_score(y_val, (y_pred > 0.5).astype(int)))
                if self.X_va2 is not None and self.y_va2 is not None:
                    y_pred_val2 = model.predict(self.X_va2)
                    val2_scores.append(accuracy_score(self.y_va2, (y_pred_val2 > 0.5).astype(int)))
            score_cv = np.mean(scores)
            score_val2 = np.mean(val2_scores) if val2_scores else 0
            return 1 - ((score_cv + score_val2) / 2)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        return study.best_params, study

    def evaluate_cv(self, params: dict) -> Tuple[List[float], List[float], Booster]:
        val_scores, base_scores, models = [], [], []
        for fold, (train_idx, val_idx) in enumerate(self.folds):
            print(f"Fold {fold + 1}")
            X_train, X_val = self.X_tr.iloc[train_idx], self.X_tr.iloc[val_idx]
            y_train, y_val = self.y_tr.iloc[train_idx], self.y_tr.iloc[val_idx]
            dtrain = lgb.Dataset(X_train, y_train)
            dvalid = lgb.Dataset(X_val, y_val)
            model = lgb.train(params, dtrain, valid_sets=[dvalid], num_boost_round=1000,
                              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(10)])
            y_pred = model.predict(X_val)
            acc = accuracy_score(y_val, (y_pred > 0.5).astype(int))
            val_scores.append(acc)
            models.append(model)
        print(f"平均CV Accuracy: {np.mean(val_scores):.4f}")
        if self.X_va2 is not None and self.y_va2 is not None:
            y_preds = np.mean([m.predict(self.X_va2) for m in models], axis=0)
            acc = accuracy_score(self.y_va2, (y_preds > 0.5).astype(int))
            print(f"アンサンブルモデルの検証Accuracy: {acc:.4f}")
        best_index = np.argmax(val_scores)
        return val_scores, base_scores, models[best_index]

    def save_model_to_s3(self, model: Booster):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models/lgb_model.pkl")
        joblib.dump(model, model_path)
        with open(model_path, "rb") as f:
            self.s3.upload_fileobj(f, self.bucket_name, self.model_output_key)
        print("最良モデルをS3に保存しました。")


if __name__ == "__main__":
    pipeline = LightGBMPipeline()
    print("Optunaによるハイパーパラメータ最適化中...")
    best_params, _ = pipeline.optimize_params(n_trials=30)
    print("クロスバリデーションでモデル評価中...")
    val_accs, base_accs, best_model = pipeline.evaluate_cv(best_params)
    pipeline.save_model_to_s3(best_model)
