# pipeline.py
import subprocess

def run_script(script_name):
    print(f"\n===== 実行中: {script_name} =====")
    result = subprocess.run(['python', script_name], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"[ERROR]:\n{result.stderr}")

if __name__ == '__main__':
    print("========== モデルパイプライン開始 ==========")

    # ステップ1: 前処理（前処理スクリプトを作成済みならここで実行）
    run_script('scripts/preprocess.py')  # 必要に応じて有効化

    # ステップ2: モデル学習
    run_script('scripts/train_model.py')
    
    # ステップ3: csvファイル提出
    run_script('scripts/submission.py')

    print("========== モデルパイプライン完了 ==========")
