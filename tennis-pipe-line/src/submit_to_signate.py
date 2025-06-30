import subprocess
import os


def submit_to_signate(competition_id: int, submission_file: str, comment: str = ""):
    if not os.path.exists(submission_file):
        print(f"[ERROR] 提出ファイルが存在しません: {submission_file}")
        return

    print("\n===== Signate提出開始 =====")
    command = ["signate", "submit", f"--competition-id={competition_id}"]

    # ファイルパスは位置引数として最後に
    command.append(submission_file)

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(result.stdout)
        print("[SUCCESS] 提出が正常に完了しました。")
    except subprocess.CalledProcessError as e:
        print("[ERROR] Signate提出コマンドでエラーが発生しました")
        print(e.stderr)
