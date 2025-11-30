#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
単一の JSONL（または JSONL.GZ）を読み込み、
SFT 用の 4 列だけ（question, ground_truth, reasoning, answer）に揃えて
Hugging Face Hub（dataset repo）へ Parquet として push する最小ユーティリティ。

- 入力 JSONL に id/raw_output などが含まれていても無視（Hub には 4 列のみ）
- 空白や改行など文字列内容の改変は一切しない（strip しない・整形しない）
- 4 列がすべて string でない場合はエラー（勝手に型変換しません）

使用例:
  # .env に HUGGINGFACE_TOKEN=hf_xxx を用意した上で
  python push_single_jsonl_to_hub_as_parquet.py \
    --jsonl ./outputs/phi4_crossthink_20250802_134425.jsonl \
    --repo_id yourname/phi4_crossthink_sft \
    --private
"""

import argparse
import os
import sys
from datasets import load_dataset, DatasetDict, Features, Value
from huggingface_hub import login

# <<< 追加 (2025-08-02) .env から HUGGINGFACE_TOKEN を読み込む >>>
from dotenv import load_dotenv  # 追加

REQUIRED_COLS = ["question", "ground_truth", "reasoning", "answer"]

def main():
    ap = argparse.ArgumentParser(
        description="Upload a single JSONL to HF Hub as a Parquet dataset (4 columns)."
    )
    ap.add_argument("--jsonl", type=str, required=True,
                    help="Path to a JSONL/JSONL.GZ file.")
    ap.add_argument("--repo_id", type=str, required=True,
                    help="HF dataset repo id, e.g. yourname/phi4_crossthink_sft")
    ap.add_argument("--private", action="store_true",
                    help="Create/keep the target dataset repo as private.")
    # <<< 変更 (2025-08-02) トークン引数は廃止し、.env の HUGGINGFACE_TOKEN を必須化 >>>
    # ap.add_argument("--hf_token", ...)  # 削除

    args = ap.parse_args()

    # <<< 追加 (2025-08-02) .env を読み込んで HUGGINGFACE_TOKEN を取得 >>>
    load_dotenv()  # .env を読み込む（カレント or 親ディレクトリ）
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("[ERROR] HUGGINGFACE_TOKEN not found in environment (.env).", file=sys.stderr)
        print("        Please set HUGGINGFACE_TOKEN=hf_xxx in your .env (or export it) and retry.", file=sys.stderr)
        sys.exit(1)

    # 1) JSONL 存在チェック（ローカルパス想定）
    if "://" not in args.jsonl and not os.path.exists(args.jsonl):
        print(f"[ERROR] JSONL not found: {args.jsonl}", file=sys.stderr)
        sys.exit(1)

    # 2) ログイン（.env から取得したトークンで必ずログイン）
    try:
        login(token=hf_token)
    except Exception as e:
        print(f"[ERROR] Failed to login with HUGGINGFACE_TOKEN: {e}", file=sys.stderr)
        sys.exit(1)

    # 3) JSONL を読み込み（.gz も可）
    print(f"[Info] Loading JSONL: {args.jsonl}")
    ds_full = load_dataset("json", data_files=args.jsonl, split="train")

    # 4) 必須列の存在チェック
    cols = ds_full.column_names
    missing = [c for c in REQUIRED_COLS if c not in cols]
    if missing:
        print(f"[ERROR] Required columns missing in JSONL: {missing}", file=sys.stderr)
        print(f"        Found columns: {cols}", file=sys.stderr)
        sys.exit(1)

    # 5) 余分な列を削除し、4列に揃える（内容は一切改変しない）
    ds_sft = ds_full.select_columns(REQUIRED_COLS)

    # 6) 型チェック（string 以外があればエラーにする：改変を避けるため）
    sample = ds_sft.select(range(min(5, len(ds_sft))))
    bad = []
    for ex in sample:
        for k in REQUIRED_COLS:
            if not isinstance(ex[k], str):
                bad.append((k, type(ex[k]).__name__))
    if bad:
        print("[ERROR] Non-string values detected (no automatic coercion to keep data unchanged):", file=sys.stderr)
        for k, t in bad:
            print(f"  - column={k}, type={t}", file=sys.stderr)
        print("Fix your JSONL to ensure all four columns are strings.", file=sys.stderr)
        sys.exit(1)

    # 7) Feature を string に明示（Parquet スキーマ）
    features = Features({k: Value("string") for k in REQUIRED_COLS})
    ds_sft = ds_sft.cast(features)

    # 8) 単一 split（train のみ）で push
    print(f"[Info] Pushing to https://huggingface.co/datasets/{args.repo_id} ...")
    dd = DatasetDict({"train": ds_sft})
    dd.push_to_hub(
        args.repo_id,
        private=args.private,
        commit_message=f"Upload from JSONL: {os.path.basename(args.jsonl)}"
    )
    print(f"[Done] Pushed dataset to: https://huggingface.co/datasets/{args.repo_id}")

if __name__ == "__main__":
    main()
