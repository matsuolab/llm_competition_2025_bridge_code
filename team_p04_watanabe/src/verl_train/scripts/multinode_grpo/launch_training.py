#!/usr/bin/env python3
"""
Multi‑node GRPO launcher for VERL
  • YAML で階層的な設定を管理
  • --override key=value で上書き可能
  • 重複キーを排除し、bool/リストを安全にシリアライズ
"""

import os
import sys
import yaml
import argparse

# ─────────────────────────────────────
# 0. 環境変数整理（ROCR / CUDA）
# ─────────────────────────────────────
for k in ("ROCR_VISIBLE_DEVICES",):
    os.environ.pop(k, None)
os.environ["RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES"] = "1"
os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"


# ─────────────────────────────────────
# 1. ヘルパ関数
# ─────────────────────────────────────
def load_config(path: str) -> dict:
    """YAML ファイルを読み込む"""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def flatten_dict(d, parent_key="", sep="."):
    """ネスト辞書 → ドット表記へ（リストは保持）"""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def serialize_value(val):
    """Hydra 引数文字列へ安全変換"""
    if isinstance(val, bool):
        return str(val).lower()                      # true / false
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, list):                        # リスト → [a,b,c]
        return f"[{','.join(map(str, val))}]"
    return str(val)


# ─────────────────────────────────────
# 2. メイン
# ─────────────────────────────────────
def main():
    # 2‑1. 引数パース
    parser = argparse.ArgumentParser(description="GRPO Training Launcher")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="YAML config path")
    parser.add_argument("--override", type=str, nargs="*", default=[],
                        help='Override (e.g. trainer.total_epochs=10)')
    cli = parser.parse_args()

    # 2‑2. 設定ロード
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = cli.config if os.path.isabs(cli.config) else os.path.join(script_dir, cli.config)
    cfg = load_config(cfg_path)

    # 2‑3. WANDB 環境変数
    os.environ["WANDB_ENTITY"] = cfg["wandb_entity"]
    os.environ["WANDB_PROJECT"] = cfg["wandb_project_name"]
    os.environ["WANDB_RUN_NAME"] = cfg["wandb_run_name"]

    # 2‑4. ドット表記へ展開
    flat = flatten_dict(cfg)

    # 2‑5. オーバーライド反映
    for ov in cli.override:
        if "=" not in ov:
            continue
        k, v = ov.split("=", 1)
        # 型推論
        if v.lower() in ("true", "false"):
            v = v.lower() == "true"
        else:
            for cast in (int, float):
                try:
                    v = cast(v)
                    break
                except ValueError:
                    pass
        flat[k] = v

    # 2‑6. trainer.* で環境依存の値を補填・上書き
    flat.update({
        "trainer.project_name":      cfg["wandb_project_name"],
        "trainer.experiment_name":   cfg["wandb_run_name"],
        "trainer.n_gpus_per_node":   cfg["gpus_per_node"],
        "trainer.nnodes":            cfg["nnodes"],
        "trainer.default_local_dir": (
            f"{os.environ['HOME']}{flat.get('trainer.default_local_dir_suffix', '/training/multinode/grpo/checkpoints')}"
        ),
    })
    # default_local_dir_suffix はここで吸収したので以後不要
    flat.pop("trainer.default_local_dir_suffix", None)

    # 2‑7. ENV 専用キーを除外
    env_only = {"gpus_per_node", "nnodes",
                "wandb_entity", "wandb_project_name", "wandb_run_name"}
    verl_args = [f"{k}={serialize_value(v)}"
                 for k, v in flat.items()
                 if k not in env_only]

    # ─────────────────────────────────
    # 3. VERL (main_ppo) 呼び出し
    # ─────────────────────────────────
    from verl.trainer import main_ppo
    sys.argv = ["verl.trainer.main_ppo"] + verl_args
    #sys.argv = ["verl.trainer.main_ppo"] + ["--config-path=config"] + ["--config-name=ppo_megatron_trainer.yaml"] + verl_args
    main_ppo.main()


if __name__ == "__main__":
    main()
