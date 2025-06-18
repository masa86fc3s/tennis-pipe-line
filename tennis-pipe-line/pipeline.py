# pipeline.py
import subprocess
import os


# 保存先のパス（バケットやキーの変数として定義）
# 出力ディレクトリの作成

# 変数で定義して一括管理
train_key = os.path.join( 'data')
test_key = os.path.join( 'data')
model_output_key = os.path.join( 'output')
model_key = os.path.join( 'models')

# ディレクトリ作成
for path in [train_key, test_key, model_output_key, model_key]:
    os.makedirs(path, exist_ok=True)


def run_script(script_name):
    print(f"\n===== 実行中: {script_name} =====")
    result = subprocess.run(['python', script_name], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"[ERROR]:\n{result.stderr}")

if __name__ == '__main__':
    print("========== モデルパイプライン開始 ==========")

    # ステップ1: 前処理（前処理スクリプトを作成済みならここで実行）
    run_script('src/preprocess.py')  # 必要に応じて有効化

    # ステップ2: モデル学習
    run_script('src/train_model.py')
    
    # ステップ3: csvファイル提出
    run_script('src/submission.py')

    print("========== モデルパイプライン完了 ==========")
