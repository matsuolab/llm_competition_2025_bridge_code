import sys
from huggingface_hub import snapshot_download

def download_model(model_id: str, local_dir: str):
    """
    モデルをダウンロード
    
    Args:
        model_id: HuggingFaceのモデルID (例: "Qwen/Qwen2.5-72B-Instruct")
        local_dir: ダウンロード先のローカルディレクトリ
    """
    print(f"Downloading {model_id} to {local_dir}...")
    
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # シンボリックリンクを使わず実ファイルをコピー
            resume_download=True,           # 中断時は再開
            max_workers=4                   # 並列ダウンロード数
        )
        print(f"✓ Download complete: {local_dir}")
    except Exception as e:
        print(f"✗ Download failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python simple_download.py <model_id> <local_dir>")
        print("Example: python simple_download.py Qwen/Qwen2.5-72B-Instruct ./models/qwen72b")
        sys.exit(1)
    
    model_id = sys.argv[1]
    local_dir = sys.argv[2]
    
    download_model(model_id, local_dir)