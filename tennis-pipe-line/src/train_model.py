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


def is_sagemaker():
    return "SM_MODEL_DIR" in os.environ


def cast_optuna_params(optuna_params: dict) -> dict:
    """
    Optunaのパラメータ辞書内の文字列を適切な型に変換する。
    """
    def str_to_bool(s):
        if isinstance(s, bool):
            return s
        if isinstance(s, str):
            return s.lower() in ["true", "yes", "1"]
        return bool(s)

    casted = {}
    for param_name, param_dict in optuna_params.items():
        casted_param = {}
        for k, v in param_dict.items():
            if k in ["low", "high"]:
                casted_param[k] = float(v)
            elif k == "log":
                casted_param[k] = str_to_bool(v)
            elif k == "step":
                casted_param[k] = float(v)
            else:
                casted_param[k] = v
        casted[param_name] = casted_param
    return casted


class LightGBMPipeline:
    def __init__(
        self,
        train_data_path: str = None,
        s3_config_path: str = "../yml/s3_data.yml",
        config_path: str = "config.yml",
        features_path: str = "features.yml",
        model_output_dir: str = "./model",
        use_s3: bool = False,
    ):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 引数の use_s3 をインスタンス変数に保存
        self.use_s3 = use_s3

        # （以降、他の処理...）

        # config.yaml 読み込み
        full_config_path = os.path.join(base_dir, config_path)
        with open(full_config_path, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        # Optunaパラメータの型を変換
        self.config["optuna_params"] = cast_optuna_params(self.config["optuna_params"])

        # features.yml 読み込み
        full_features_path = os.path.join(base_dir, features_path)
        with open(full_features_path, "r") as f:
            features_config = yaml.safe_load(f)

        self.features = features_config["features"]["columns"]
        self.target = features_config["target"]

        if use_s3:
            # s3_data.yml 読み込み
            s3_config_fullpath = os.path.join(base_dir, s3_config_path)
            with open(s3_config_fullpath, "r") as f:
                s3_config = yaml.safe_load(f)

            bucket = s3_config["s3"]["bucket_name"]
            region = s3_config["s3"]["region"]
            key = s3_config["s3"]["train_key"]

            print(f"S3から読み込み: bucket={bucket}, key={key}")
            s3 = boto3.client("s3", region_name=region)
            obj = s3.get_object(Bucket=bucket, Key=key)
            body = obj["Body"].read()
            self.df_train = pd.read_csv(io.BytesIO(body), sep="\t")

        else:
            if train_data_path is None:
                raise ValueError("ローカル実行時は train_data_path を指定してください")
            print(f"ローカルファイル読み込み: {train_data_path}")
            self.df_train = pd.read_csv(train_data_path, sep="\t")

        self.X = self.df_train[self.features]
        self.y = self.df_train[self.target]

        # train/validation split
        self.X_tr, self.X_va2, self.y_tr, self.y_va2 = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        self.folds = list(
            StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(
                self.X_tr, self.y_tr
            )
        )

        self.model_output_dir = model_output_dir

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

        if getattr(self, "use_s3", False):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            s3_config_path = os.path.join(base_dir, "../yml/s3_data.yml")
            with open(s3_config_path, "r") as f:
                s3_config = yaml.safe_load(f)

            bucket = s3_config["s3"]["bucket_name"]
            region = s3_config["s3"]["region"]
            s3_key = s3_config["s3"].get("model_key", "model/lgb_model.pkl")  # もし設定なければここに固定

            s3 = boto3.client("s3", region_name=region)
            s3.upload_file(model_path, bucket, s3_key)
            print(f"S3にモデルをアップロードしました: s3://{bucket}/{s3_key}")



def main(args):
    if is_sagemaker():
        train_data_path = os.path.join(os.environ["SM_CHANNEL_TRAIN"], "train_preprocessed.tsv")
        use_s3 = False
    else:
        # 引数が一切指定されていなければ S3を使うのをデフォルトにする
        if args.train_data is None and not args.use_s3:
            print("引数指定なしなので、ローカル実行でもS3から学習データを読み込みます。")
            use_s3 = True
            train_data_path = None
        else:
            use_s3 = args.use_s3
            train_data_path = args.train_data

    pipeline = LightGBMPipeline(
        train_data_path=train_data_path,
        s3_config_path="../yml/s3_data.yml",
        config_path="config.yml",
        features_path="features.yml",
        model_output_dir=args.model_dir,
        use_s3=use_s3,
    )

    print("Optunaでハイパーパラメータ最適化中...")
    best_params, _ = pipeline.optimize_params(n_trials=30)

    print("クロスバリデーション評価中...")
    _, best_model = pipeline.evaluate_cv(best_params)

    pipeline.save_model(best_model)


if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=str, help="ローカルの学習データファイルパス")
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./model"), help="モデル出力ディレクトリ")
    parser.add_argument("--use-s3", action="store_true", help="ローカルでもS3から学習データを取得する場合に指定")
    args = parser.parse_args()

    main(args)
