#!/bin/bash
#SBATCH --job-name=sft_training
#SBATCH --partition=P04
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --nodelist=osk-gpu62
#SBATCH --cpus-per-task=240
#SBATCH --output=./%x_%j.out
#SBATCH --error=./%x_%j.err

# スクリプトのディレクトリを取得
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# コンフィグファイルのパス（引数で指定可能、デフォルトは同じディレクトリのconfig.yaml）
CONFIG_FILE="${1:-${SCRIPT_DIR}/config.yaml}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Using config file: $CONFIG_FILE"

# Python仮想環境の起動
source ~/.venv/bin/activate

# リソース制限の設定
ulimit -v unlimited
unset ROCR_VISIBLE_DEVICES

# 作業ディレクトリに移動
cd ~/server_development/train/scripts/singlenode_sft

# Pythonスクリプトで動的にYAMLを処理してtorchrunコマンドを実行
python3 -c "
import yaml
import os
import sys
import subprocess

def flatten_dict(d, parent_key='', sep='.'):
    '''ネスト辞書をドット表記にフラット化'''
    items = {}
    for k, v in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

def serialize_value(val):
    '''値をHydra形式の文字列に変換'''
    if isinstance(val, bool):
        return str(val).lower()
    elif isinstance(val, list):
        # リストは[a,b,c]形式
        return '[' + ','.join(str(v) for v in val) + ']'
    else:
        return str(val)

def main():
    # YAMLファイルを読み込み
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    
    # 環境変数を設定（env セクションから）
    if 'env' in config:
        for key, value in config['env'].items():
            env_key = key.upper()
            if key == 'nccl_socket_ifname':
                env_key = 'NCCL_SOCKET_IFNAME'
            elif key == 'nvte_fused_attn':
                env_key = 'NVTE_FUSED_ATTN'
            elif key == 'cuda_visible_devices':
                env_key = 'CUDA_VISIBLE_DEVICES'
            os.environ[env_key] = str(value)
    
    # WANDB環境変数を設定
    if 'wandb' in config:
        os.environ['WANDB_ENTITY'] = config['wandb'].get('entity', '')
        os.environ['WANDB_PROJECT_NAME'] = config['wandb'].get('project_name', '')
        run_name_prefix = config['wandb'].get('run_name_prefix', 'sft_training')
        slurm_job_id = os.environ.get('SLURM_JOB_ID', 'local')
        os.environ['WANDB_RUN_NAME'] = f'{run_name_prefix}_{slurm_job_id}'
    
    # default_local_dirの展開
    if 'trainer' in config and 'default_local_dir' in config['trainer']:
        default_local_dir = os.path.expanduser(config['trainer']['default_local_dir'])
        os.makedirs(default_local_dir, exist_ok=True)
        config['trainer']['default_local_dir'] = default_local_dir
    
    # 設定をフラット化
    flat_config = flatten_dict(config)
    
    # trainer.project_nameとtrainer.experiment_nameをWANDB設定で上書き
    if 'wandb.project_name' in flat_config:
        flat_config['trainer.project_name'] = flat_config['wandb.project_name']
    if 'wandb.run_name_prefix' in flat_config:
        flat_config['trainer.experiment_name'] = os.environ['WANDB_RUN_NAME']
    
    # torchrunコマンドを構築
    cmd = [
        'torchrun',
        '--standalone',
        '--nnodes=1',
        '--nproc_per_node=8',
        '-m', 'verl.trainer.fsdp_sft_trainer'
    ]
    
    # 環境専用のキーを除外してコマンドライン引数を構築
    exclude_keys = {'env', 'wandb', 'hardware'}
    
    for key, value in flat_config.items():
        # トップレベルの除外キーをスキップ
        if key.split('.')[0] in exclude_keys:
            continue
        
        # 特殊な処理が必要なキー
        if key == 'data.response_dict_keys':
            # response_dict_keysは+プレフィックスが必要
            cmd.append(f'+{key}={serialize_value(value)}')
        else:
            cmd.append(f'{key}={serialize_value(value)}')
    
    # logger設定を追加（YAMLに明示的に設定がない場合のデフォルト）
    if 'trainer.logger' not in flat_config:
        cmd.append(\"trainer.logger=['console','wandb']\")
    
    # コマンドを表示
    print('Executing command:')
    print(' '.join(cmd))
    print()
    
    # コマンドを実行
    subprocess.run(cmd)

if __name__ == '__main__':
    main()
"