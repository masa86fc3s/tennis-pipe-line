import subprocess
import os

# ベースディレクトリ（このファイルと同じ階層）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# サブディレクトリの作成先を絶対パスで指定
train_key = os.path.join(BASE_DIR, "data")
test_key = os.path.join(BASE_DIR, "data")
model_output_key = os.path.join(BASE_DIR, "output")
model_key = os.path.join(BASE_DIR, "models")

# ディレクトリ作成
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

    print("========== モデルパイプライン完了 ==========")
