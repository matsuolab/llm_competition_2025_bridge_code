#!/usr/bin/env bash






# ~下に login.shを作成する
# vim login.shをする
#下の内容(HF_tokenのみ変更)を記入する
# . login.shで有効にする(対話型)
# source $HOME





# --- Modules（必要な場合のみ） ---
module reset
module load nccl/2.22.3
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt
module load miniconda/24.7.1-py311

# --- Conda 初期化（非対話でも使えるように） ---
source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh

# base の自動起動は OFF（初回だけでOK／重複しても害なし）
conda config --set auto_activate_base false || true

# いったん掃除（login.sh由来のbaseが入っていても落とす）
conda deactivate >/dev/null 2>&1 || true

# ★ ここが重要：~ を使わず、絶対パスか $HOME を使う
export CONDA_PATH="$HOME/conda_env"
conda activate "$CONDA_PATH"


# --- 非対話ログイン（必要なら） ---
# HF_TOKEN / WANDB
export HF_TOKEN="hf_token"
if [[ -n "${HF_TOKEN:-}" ]]; then
  huggingface-cli login --token "$HF_TOKEN"
else
  echo "[WARN] HF_TOKEN が未設定: Hugging Face ログインをスキップ" >&2
fi

wandb login



# --- 動作確認（最初だけ残してOK） ---
echo "=== Conda Env Check ==="
echo "which python: $(which python)"
python -V
conda info --envs | sed -n '1,20p'
echo "======================="
~                                                                                                                                                                     
~                                                                                                                                                                     
~                                                                                                                                                                     
~                                                                                                                                                                     
~  