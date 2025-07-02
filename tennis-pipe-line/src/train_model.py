import os
import io
import argparse
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import joblib
import yaml
import boto3
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score

class LightGBMPipeline:
    def __init__(self, train_data_path: str, model_output_dir: str, config_path: str = "../yml/config.yml"):
        self.train_data_path = train_data_path
        self.model_output_dir = model_output_dir

        # config読み込み
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_config_path = os.path.join(base_dir, config_path)
        with open(full_config_path, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        # S3クライアント（もし必要なら）
        self.bucket_name = None
        self.region = None

        # データ読み込み
        self.df_train = pd.read_csv(self.train_data_path, sep="\t")
        self.features = self.config["features"]["columns"]
        self.target = self.config["target"]

        self.X = self.df_train[self.features]
        self.y = self.df_train[self.target]
        self.X_tr, self.X_va2, self.y_tr, self.y_va2 = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        self.folds = list(
            StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(
                self.X_tr, self.y_tr
            )
        )

    def optimize_params(self, n_trials: int = 30):
        optuna_cfg = self.config["optuna_params"]
        base_params = self.config["model_params"]

        def objective(trial):
            params = base_params.copy()
            params["lambda_l1"] = trial.suggest_float("lambda_l1", **optuna_cfg["lambda_l1"])
            params["lambda_l2"] = trial.suggest_float("lambda_l2", **optuna_cfg["lambda_l2"])
            params["num_leaves"] = trial.suggest_int("num_leaves", **optuna_cfg["num_leaves"])
            params["feature_fraction"] = trial.suggest_float("feature_fraction", **optuna_cfg["feature_fraction"])
            params["bagging_fraction"] = trial.suggest_float("bagging_fraction", **optuna_cfg["bagging_fraction"])
            params["bagging_freq"] = trial.suggest_int("bagging_freq", **optuna_cfg["bagging_freq"])
            params["min_child_samples"] = trial.suggest_int("min_child_samples", **optuna_cfg["min_child_samples"])

            scores = []
            val2_scores = []
            for train_idx, val_idx in self.folds:
                X_train, X_val = self.X_tr.iloc[train_idx], self.X_tr.iloc[val_idx]
                y_train, y_val = self.y_tr.iloc[train_idx], self.y_tr.iloc[val_idx]
                dtrain = lgb.Dataset(X_train, y_train)
                dvalid = lgb.Dataset(X_val, y_val, reference=dtrain)
                model = lgb.train(
                    params,
                    dtrain,
                    valid_sets=[dvalid],
                    num_boost_round=1000,
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
                )
            y_pred = np.array(model.predict(X_val))
            scores.append(accuracy_score(y_val, (y_pred > 0.5).astype(int)))

            if self.X_va2 is not None and self.y_va2 is not None:
                y_pred_val2 = np.array(model.predict(self.X_va2))
                val2_scores.append(accuracy_score(self.y_va2, (y_pred_val2 > 0.5).astype(int)))

            score_cv = float(np.mean(scores))
            score_val2 = float(np.mean(val2_scores)) if val2_scores else 0.0
            return 1.0 - ((score_cv + score_val2) / 2.0)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        return study.best_params, study

    def evaluate_cv(self, params):
        val_scores = []
        models = []
        for fold, (train_idx, val_idx) in enumerate(self.folds):
            print(f"Fold {fold + 1}")
            X_train, X_val = self.X_tr.iloc[train_idx], self.X_tr.iloc[val_idx]
            y_train, y_val = self.y_tr.iloc[train_idx], self.y_tr.iloc[val_idx]
            dtrain = lgb.Dataset(X_train, y_train)
            dvalid = lgb.Dataset(X_val, y_val)
            model = lgb.train(
                params,
                dtrain,
                valid_sets=[dvalid],
                num_boost_round=1000,
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(10)],
            )
            y_pred = np.array(model.predict(X_val))
            acc = accuracy_score(y_val, (y_pred > 0.5).astype(int))
            val_scores.append(acc)
            models.append(model)
        print(f"平均CV Accuracy: {np.mean(val_scores):.4f}")
        if self.X_va2 is not None and self.y_va2 is not None:
            y_preds = np.mean([m.predict(self.X_va2) for m in models], axis=0)
            acc = accuracy_score(self.y_va2, (y_preds > 0.5).astype(int))
            print(f"検証用データ Accuracy: {acc:.4f}")
        best_index = np.argmax(val_scores)
        return val_scores, models[best_index]

    def save_model(self, model):
        os.makedirs(self.model_output_dir, exist_ok=True)
        model_path = os.path.join(self.model_output_dir, "lgb_model.pkl")
        joblib.dump(model, model_path)
        print(f"モデルを {model_path} に保存しました。")


def main(args):
    pipeline = LightGBMPipeline(train_data_path=args.train_data, model_output_dir=args.model_dir)
    print("Optunaでハイパーパラメータ最適化中...")
    best_params, _ = pipeline.optimize_params(n_trials=30)
    print("クロスバリデーション評価中...")
    _, best_model = pipeline.evaluate_cv(best_params)
    pipeline.save_model(best_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=str, required=True, help="前処理済みの学習データTSVファイルパス")
    parser.add_argument("--model-dir", type=str, default="/opt/ml/model", help="モデル出力ディレクトリ")
    args = parser.parse_args()
    main(args)
