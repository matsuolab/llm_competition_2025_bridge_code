#!/usr/bin/env python

import hydra
import os
import json
import requests
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from omegaconf import OmegaConf
from datasets import load_dataset
from huggingface_hub import HfApi

def check_dataset_cache(dataset_name):
    """データセットがHugging Faceキャッシュに存在するかチェック"""
    try:
        # データセットの読み込みを試行（キャッシュから）
        dataset = load_dataset(dataset_name, split="test", streaming=False)
        return True, f"✅ データセット '{dataset_name}' はキャッシュに存在します ({len(dataset)} samples)"
    except Exception as e:
        return False, f"❌ データセット '{dataset_name}' のアクセスに失敗: {str(e)}"

def check_model_cache(model_name):
    """モデルがローカルに存在するかチェック（共有ディレクトリも含む）"""
    try:
        # 1. 共有ディレクトリをチェック（複数のパターンを試行）
        model_basename = model_name.split('/')[-1]  # "Qwen/Qwen3-235B-A22B" -> "Qwen3-235B-A22B"
        
        # パターン1: モデル名のベース名のみ
        shared_model_path1 = f"/home/Competition2025/P05/shareP05/models/{model_basename}"
        # パターン2: フルパス
        shared_model_path2 = f"/home/Competition2025/P05/shareP05/models/{model_name}"
        
        for shared_path in [shared_model_path1, shared_model_path2]:
            if Path(shared_path).exists():
                try:
                    result = subprocess.run(f"du -sh '{shared_path}' | cut -f1", 
                                          shell=True, capture_output=True, text=True)
                    model_size = result.stdout.strip() if result.returncode == 0 else "unknown size"
                    return True, f"✅ モデル '{model_name}' は共有ディレクトリに存在します ({model_size})"
                except:
                    return True, f"✅ モデル '{model_name}' は共有ディレクトリに存在します"
        
        # 2. 個人のHugging Faceキャッシュをチェック
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        model_cache_dirs = list(cache_dir.glob(f"models--{model_name.replace('/', '--')}*"))
        
        if model_cache_dirs:
            return True, f"✅ モデル '{model_name}' は個人キャッシュに存在します"
        
        # 3. Hugging Face Hub APIで存在確認
        api = HfApi()
        model_info = api.model_info(model_name)
        return True, f"⚠️ モデル '{model_name}' は存在しますが、ローカルキャッシュにありません（初回ダウンロードが必要）"
        
    except Exception as e:
        return False, f"❌ モデル '{model_name}' のアクセスに失敗: {str(e)}"

def check_vllm_server(base_url):
    """vLLMサーバーにアクセス可能かチェック"""
    try:
        # ヘルスチェックエンドポイントを試行
        health_url = base_url.replace('/v1', '/health')
        response = requests.get(health_url, timeout=5)
        if response.status_code == 200:
            return True, f"✅ vLLMサーバー '{base_url}' にアクセス可能です"
        else:
            return False, f"❌ vLLMサーバー '{base_url}' からエラー応答: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, f"❌ vLLMサーバー '{base_url}' に接続できません（サーバーが起動していない可能性）"
    except requests.exceptions.Timeout:
        return False, f"❌ vLLMサーバー '{base_url}' への接続がタイムアウトしました"
    except Exception as e:
        return False, f"❌ vLLMサーバー '{base_url}' のチェックに失敗: {str(e)}"

def check_provider(provider):
    """プロバイダー設定をチェック"""
    if provider == "vllm":
        return True, f"✅ プロバイダー '{provider}' は有効です"
    else:
        return False, f"❌ プロバイダー '{provider}' はサポートされていません（vllmのみサポート）"

def run_dryrun_check(cfg):
    """ドライランチェックを実行"""
    print("=" * 60)
    print("🔍 PREDICT DRYRUN CHECK")
    print("=" * 60)
    print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 設定値の表示
    print("📋 現在の設定値:")
    print("-" * 30)
    
    all_checks_passed = True
    
    # 1. データセットチェック
    print(f"dataset: {cfg.dataset}")
    dataset_ok, dataset_msg = check_dataset_cache(cfg.dataset)
    print(f"  {dataset_msg}")
    all_checks_passed &= dataset_ok
    print()
    
    # 2. プロバイダーチェック
    print(f"provider: {cfg.provider}")
    provider_ok, provider_msg = check_provider(cfg.provider)
    print(f"  {provider_msg}")
    all_checks_passed &= provider_ok
    print()
    
    # 3. base_urlチェック
    print(f"base_url: {cfg.base_url}")
    server_ok, server_msg = check_vllm_server(cfg.base_url)
    print(f"  {server_msg}")
    all_checks_passed &= server_ok
    print()
    
    # 4. モデルチェック
    print(f"model: {cfg.model}")
    model_ok, model_msg = check_model_cache(cfg.model)
    print(f"  {model_msg}")
    all_checks_passed &= model_ok
    print()
    
    # 5. その他の設定値表示
    print("⚙️ その他の設定値:")
    print("-" * 30)
    print(f"max_completion_tokens: {cfg.max_completion_tokens}")
    print(f"num_workers: {cfg.num_workers}")
    print(f"max_samples: {cfg.max_samples}")
    print(f"reasoning: {cfg.reasoning}")
    print()
    
    # 6. 出力ファイル形式の表示
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(cfg.model)
    timestamped_filename = f"predicted-hle-{model_name}-{timestamp}.json"
    
    print("📁 出力ファイル形式:")
    print("-" * 30)
    print("predictions/")
    print(f"├── {timestamped_filename}  # タイムスタンプ付きメインファイル")
    print(f"├── predict-latest.json -> {timestamped_filename}  # 最新結果へのリンク")
    print(f"└── hle_{model_name}.json -> {timestamped_filename}  # judge用互換性リンク")
    print()
    
    # 7. 総合結果
    print("=" * 60)
    if all_checks_passed:
        print("✅ 全てのチェックが成功しました。実行準備完了です！")
        print()
        print("🚀 実行コマンド:")
        print("   python predict_improved.py")
        return_code = 0
    else:
        print("❌ 一部のチェックが失敗しました。問題を解決してから再実行してください。")
        return_code = 1
    
    print("=" * 60)
    return return_code

def create_judge_compatibility_link(output_filepath, model_name):
    """judge.py用の互換性シンボリックリンクを作成"""
    try:
        predictions_dir = Path("predictions")
        
        # judge.pyが期待するファイル名形式: hle_ModelName.json (従来形式維持)
        judge_filename = f"hle_{model_name}.json"
        judge_link = predictions_dir / judge_filename
        
        # 既存のリンクを削除
        if judge_link.exists() or judge_link.is_symlink():
            judge_link.unlink()
        
        # 新しいシンボリックリンクを作成
        target_filename = Path(output_filepath).name
        judge_link.symlink_to(target_filename)
        
        print(f"✅ judge用互換性リンクを作成: {judge_filename} -> {target_filename}")
        return True
        
    except Exception as e:
        print(f"⚠️ judge用互換性リンク作成に失敗: {str(e)}")
        return False

def create_latest_link(output_filepath):
    """predict-latest.jsonシンボリックリンクを作成"""
    try:
        predictions_dir = Path("predictions")
        latest_link = predictions_dir / "predict-latest.json"
        
        # 既存のリンクを削除
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        
        # 新しいシンボリックリンクを作成
        target_filename = Path(output_filepath).name
        latest_link.symlink_to(target_filename)
        
        print(f"✅ 最新結果リンクを作成: predict-latest.json -> {target_filename}")
        return True
        
    except Exception as e:
        print(f"⚠️ 最新結果リンク作成に失敗: {str(e)}")
        return False

@hydra.main(config_name="config", version_base=None, config_path="conf")
def main(cfg):
    # dryrunモードのチェック（Hydraオーバーライドとして処理）
    if hasattr(cfg, 'dryrun') and cfg.dryrun:
        return_code = run_dryrun_check(cfg)
        sys.exit(return_code)
    
    # 通常の実行モード
    # タイムスタンプ付きファイル名の生成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(cfg.model)
    
    # predictionsディレクトリの作成
    predictions_dir = Path("predictions")
    predictions_dir.mkdir(exist_ok=True)
    
    # タイムスタンプ付きファイル名
    timestamped_filename = f"predicted-hle-{model_name}-{timestamp}.json"
    output_filepath = predictions_dir / timestamped_filename
    
    # 最新ファイルへのシンボリックリンク
    latest_link = predictions_dir / "predict-latest.json"
    
    # 継続実行の確認
    resume_file = None
    if hasattr(cfg, 'resume') and cfg.resume:
        # 最新のファイルから継続
        if latest_link.exists() and latest_link.is_symlink():
            resume_file = latest_link.resolve()
            print(f"継続モード: {resume_file} から再開します")
        else:
            print("継続モード指定されましたが、継続可能なファイルが見つかりません")
    
    # OmegaConfの設定を更新（struct modeを無効化）
    OmegaConf.set_struct(cfg, False)
    cfg.output_filepath = str(output_filepath)
    cfg.resume_file = str(resume_file) if resume_file else None
    cfg.latest_link = str(latest_link)
    OmegaConf.set_struct(cfg, True)
    
    # 予測実行
    print(f"🚀 予測処理を開始します...")
    print(f"出力ファイル: {output_filepath}")
    
    if cfg.provider == "openai":
        from hle_benchmark import openai_predictions
        openai_predictions.main(cfg)
    elif cfg.provider == "ollama":
        from hle_benchmark import ollama_predictions
        ollama_predictions.main(cfg)
    elif cfg.provider == "vllm":
        from hle_benchmark import vllm_predictions_improved
        vllm_predictions_improved.main(cfg)
    
    # 予測完了後、シンボリックリンクを作成
    if output_filepath.exists():
        print(f"\n📁 シンボリックリンクを作成中...")
        
        # 1. predict-latest.json リンクを作成
        create_latest_link(output_filepath)
        
        # 2. judge用互換性リンクを作成
        create_judge_compatibility_link(output_filepath, model_name)
        
        print(f"\n✅ 予測処理が完了しました！")
        print(f"📊 結果ファイル:")
        print(f"  - メイン: {output_filepath.name}")
        print(f"  - 最新: predict-latest.json")
        print(f"  - judge用: hle_{model_name}.json")
        
    else:
        print(f"\n⚠️ 出力ファイルが作成されませんでした: {output_filepath}")

if __name__ == "__main__":
    main()
