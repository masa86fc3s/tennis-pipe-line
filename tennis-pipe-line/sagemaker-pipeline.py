import os
import sys
import subprocess

# ===== パス設定（ローカル / SageMaker 判定） =====
if os.path.exists("/opt/ml/processing/input/tennis-pipe-line/src"):
    BASE_DIR = "/opt/ml/processing/input/tennis-pipe-line"
    sys.path.append(os.path.join(BASE_DIR, "src"))
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(BASE_DIR, "src"))

from submit_to_signate import submit_to_signate

# 必要ライブラリ自動インストール
try:
    import yaml
    import lightgbm
except ImportError:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "pandas",
            "numpy",
            "boto3",
            "matplotlib",
            "lightgbm",
            "optuna",
        ],
        check=True,
    )


# 各ディレクトリのパス設定
train_key = os.path.join(BASE_DIR, "data")
test_key = os.path.join(BASE_DIR, "data")
model_output_key = os.path.join(BASE_DIR, "output")
model_key = os.path.join(BASE_DIR, "models")

# 必要なディレクトリを作成
for path in [train_key, test_key, model_output_key, model_key]:
    os.makedirs(path, exist_ok=True)


def run_script(script_relative_path: str):
    script_path = os.path.join(BASE_DIR, script_relative_path)
    print(f"\n===== 実行中: {script_path} =====")
    result = subprocess.run(["python", script_path], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"[ERROR]:\n{result.stderr}")


if __name__ == "__main__":
    print("========== モデルパイプライン開始 ==========")

    # ステップ1: 前処理
    run_script("src/preprocess.py")

    # ステップ2: モデル学習
    run_script("src/train_model.py")

    # ステップ3: 提出ファイル作成
    run_script("src/submission.py")

    """
    # ステップ4: Signateに提出
    competition_id = 118  # ← 適宜変更してください
    submission_file_path = os.path.join(BASE_DIR, "output", "submission.csv")
    submit_to_signate(competition_id, submission_file_path, comment="今日のモデル結果")
    """

    print("========== モデルパイプライン完了 ==========")
