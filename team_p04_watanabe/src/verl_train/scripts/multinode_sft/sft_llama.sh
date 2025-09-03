#!/bin/bash
set -eo pipefail   # ← ここを追加すると途中の失敗で即 abort

# 環境セットアップ
# Python 仮想環境を有効化（複数のパスを試行）
if [ -f "$HOME/venv/bin/activate" ]; then
    source "$HOME/venv/bin/activate"
elif [ -f "$HOME/.venv/bin/activate" ]; then
    source "$HOME/.venv/bin/activate"
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Warning: Virtual environment not found, using system Python"
fi

# 必要な環境変数の設定
# 通信IF（固定で良ければ IFACE=enp25s0np0 をそのまま）
IFACE=${IFACE:-}
if [ -z "$IFACE" ]; then
  if ip -o link show enp25s0np0 >/dev/null 2>&1; then
    IFACE=enp25s0np0
  else
    IFACE=$(ip -o route get 8.8.8.8 | awk '{print $5; exit}')
  fi
fi
export NCCL_SOCKET_IFNAME="$IFACE"
export GLOO_SOCKET_IFNAME="$IFACE"   # ★ 重要：GLOO も同じIFに
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-DETAIL}
# rendezvous 設定
export RDZV_ID=${RDZV_ID:-${SLURM_JOB_ID:-sft}}
export RDZV_TIMEOUT=${RDZV_TIMEOUT:-600}
# 失敗原因を拾う
export TORCHELASTIC_ERROR_FILE="${TORCHELASTIC_ERROR_FILE:-$HOME/te_error_${HOSTNAME}.json}"
ulimit -v unlimited

# 引数の処理
CONFIG_FILE=${1:-config.yaml}
MASTER_ADDR=${2:-}
MASTER_PORT=${3:-}
NODE_RANK=${4:-}
NNODES=${5:-}
GPUS_PER_NODE=${6:-}

# スクリプトディレクトリの取得
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONFIG_PATH="${SCRIPT_DIR}/${CONFIG_FILE}"

# 設定をYAMLから読み込む
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found: $CONFIG_PATH"
    exit 1
fi

# PYTHONPATHにverlモジュールのパスを追加
export PYTHONPATH="${PYTHONPATH}:${HOME}/server_development:${HOME}/server_development/train"

# verlモジュールの存在確認
python3 -c "import verl" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: verl module not found. Please ensure verl is installed or PYTHONPATH is set correctly."
    echo "Current PYTHONPATH: $PYTHONPATH"
    exit 1
fi

# Pythonスクリプトで設定を処理してtorchrunを実行
python3 - <<EOF
import os
import sys
import yaml
import subprocess

# YAMLファイルを読み込む
with open('${CONFIG_PATH}', 'r') as f:
    cfg = yaml.safe_load(f)

# コマンドライン引数でオーバーライド
if '${MASTER_ADDR}':
    cfg['master_addr'] = '${MASTER_ADDR}'
if '${MASTER_PORT}':
    cfg['master_port'] = '${MASTER_PORT}'
if '${NODE_RANK}':
    cfg['node_rank'] = int('${NODE_RANK}')
if '${NNODES}':
    cfg['nnodes'] = int('${NNODES}')
if '${GPUS_PER_NODE}':
    cfg['gpus_per_node'] = int('${GPUS_PER_NODE}')

# WANDB環境変数を設定
os.environ['WANDB_ENTITY'] = cfg['wandb_entity']
os.environ['WANDB_PROJECT_NAME'] = cfg['wandb_project_name']
os.environ['WANDB_RUN_NAME'] = cfg['wandb_run_name']

# ネスト辞書をフラット化
def flatten_dict(d, parent_key='', sep='.'):
    items = {}
    for k, v in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

# 値をシリアライズ
def serialize_value(val):
    if isinstance(val, bool):
        return str(val).lower()
    if isinstance(val, list):
        elems = [
            f'"{x}"' if isinstance(x, str) and not x.startswith('"') else str(x)
            for x in val
        ]
        return "[" + ",".join(elems) + "]"
    return str(val)

# 設定をフラット化
flat = flatten_dict(cfg)

# trainer設定を追加
flat['trainer.project_name'] = cfg['wandb_project_name']
flat['trainer.experiment_name'] = cfg['wandb_run_name']
flat['trainer.default_local_dir'] = os.path.expandvars(cfg['trainer'].get('default_local_dir'))

# response_dict_keys は既存キーなのでそのまま上書きし、要素をクォート
if 'data.response_dict_keys' in flat:
    flat['data.response_dict_keys'] = [
        f'"{x}"' if isinstance(x, str) and not x.startswith('"') else x
        for x in flat['data.response_dict_keys']
    ]

# 環境専用キーを除外
env_only = {'gpus_per_node', 'nnodes', 'master_addr', 'master_port', 'node_rank',
            'wandb_entity', 'wandb_project_name', 'wandb_run_name'}
verl_args = []
for k, v in flat.items():
    if k not in env_only:
        verl_args.append(f'{k}={serialize_value(v)}')

# 出力情報
print(f"Master address: {cfg['master_addr']}")
print(f"Master port: {cfg['master_port']}")
print(f"Node rank: {cfg['node_rank']}")
print(f"Number of nodes: {cfg['nnodes']}")
print(f"GPUs per node: {cfg['gpus_per_node']}")
print("")

# torchrunコマンドを構築
cmd = [
    'torchrun',
    '--rdzv_backend=c10d',
    f'--rdzv_endpoint={cfg["master_addr"]}:{cfg["master_port"]}',
    f'--nnodes={cfg["nnodes"]}',
    f'--nproc_per_node={cfg["gpus_per_node"]}',
    f'--node_rank={cfg["node_rank"]}',
    '--max_restarts=0',
    f'--rdzv_id={os.environ.get("RDZV_ID","sft")}',
    '-m', 'verl.trainer.fsdp_sft_trainer'
] + verl_args

print('Running command:', ' '.join(cmd))
print("")

# コマンドを実行
subprocess.run(cmd)
EOF