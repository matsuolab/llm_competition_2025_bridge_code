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

def check_openai_api_key():
    """OPENAI_API_KEY環境変数をチェック"""
    api_key = os.environ.get('OPENAI_API_KEY')
    if api_key:
        # APIキーの一部を隠して表示
        masked_key = api_key[:8] + '*' * (len(api_key) - 12) + api_key[-4:] if len(api_key) > 12 else api_key[:4] + '*' * (len(api_key) - 4)
        return True, f"✅ OPENAI_API_KEY が設定されています ({masked_key})"
    else:
        return False, f"❌ OPENAI_API_KEY 環境変数が設定されていません"

def check_huggingface_login():
    """HuggingFace Loginステータスをチェック"""
    try:
        from huggingface_hub import whoami
        
        # HF_TOKEN環境変数をチェック
        hf_token = os.environ.get('HF_TOKEN')
        if hf_token:
            try:
                user_info = whoami(token=hf_token)
                username = user_info.get('name', 'unknown')
                return True, f"✅ HuggingFace にログイン済みです (ユーザー: {username})"
            except Exception as e:
                return False, f"❌ HF_TOKEN は設定されていますが無効です: {str(e)}"
        
        # デフォルトトークンでチェック
        try:
            user_info = whoami()
            username = user_info.get('name', 'unknown')
            return True, f"✅ HuggingFace にログイン済みです (ユーザー: {username})"
        except Exception:
            # huggingface-cli loginの状態をチェック
            try:
                result = subprocess.run(['huggingface-cli', 'whoami'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    username = result.stdout.strip()
                    return True, f"✅ HuggingFace にログイン済みです (ユーザー: {username})"
                else:
                    return False, f"❌ HuggingFace にログインしていません"
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return False, f"❌ HuggingFace ログイン状態を確認できません"
                
    except ImportError:
        return False, f"❌ huggingface_hub がインストールされていません"
    except Exception as e:
        return False, f"❌ HuggingFace ログインチェックに失敗: {str(e)}"

def check_prediction_files(model_name):
    """予測結果ファイルの存在をチェック"""
    predictions_dir = Path("predictions")
    if not predictions_dir.exists():
        return False, "❌ predictionsディレクトリが存在しません"
    
    # judge_improved.pyが実際に使用するファイル名パターンをチェック
    model_basename = os.path.basename(model_name)
    prediction_file = predictions_dir / f"hle_{model_basename}.json"
    
    if not prediction_file.exists():
        return False, f"❌ 予測結果ファイルが存在しません: {prediction_file.name}（先にpredict_improved.pyを実行してください）"
    
    if prediction_file.is_symlink():
        target = prediction_file.resolve()
        if target.exists():
            # ファイルサイズをチェックして空でないことを確認
            if target.stat().st_size < 100:  # 100バイト未満は空とみなす
                return False, f"❌ 予測結果ファイルが空です: {target.name}"
            return True, f"✅ 予測結果ファイルが利用可能です: {target.name}"
        else:
            return False, f"❌ 予測結果ファイルのリンク先が存在しません: {target}"
    else:
        if prediction_file.stat().st_size < 100:  # 100バイト未満は空とみなす
            return False, f"❌ 予測結果ファイルが空です: {prediction_file.name}"
        return True, f"✅ 予測結果ファイルが利用可能です: {prediction_file.name}"

def run_dryrun_check(cfg):
    """ドライランチェックを実行"""
    print("=" * 60)
    print("🔍 JUDGE DRYRUN CHECK")
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
    
    # 5. 予測結果ファイルチェック
    print("prediction_files: 予測結果ファイル")
    pred_ok, pred_msg = check_prediction_files(cfg.model)
    print(f"  {pred_msg}")
    all_checks_passed &= pred_ok
    print()
    
    # 6. OPENAI_API_KEYチェック
    print("OPENAI_API_KEY: 環境変数")
    api_key_ok, api_key_msg = check_openai_api_key()
    print(f"  {api_key_msg}")
    all_checks_passed &= api_key_ok
    print()
    
    # 7. HuggingFace Loginチェック
    print("HuggingFace_Login: ログイン状態")
    hf_login_ok, hf_login_msg = check_huggingface_login()
    print(f"  {hf_login_msg}")
    all_checks_passed &= hf_login_ok
    print()
    
    # 8. その他の設定値表示
    print("⚙️ その他の設定値:")
    print("-" * 30)
    print(f"max_completion_tokens: {cfg.max_completion_tokens}")
    print(f"num_workers: {cfg.num_workers}")
    print(f"max_samples: {cfg.max_samples}")
    print()
    
    # 9. 出力ファイル形式の表示
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(cfg.model)
    timestamped_filename = f"judged-hle-{model_name}-{timestamp}.json"
    
    print("📁 出力ファイル形式:")
    print("-" * 30)
    print("judged/")
    print(f"├── {timestamped_filename}  # タイムスタンプ付き")
    print(f"└── judge-latest.json -> {timestamped_filename}  # 最新へのリンク")
    print()
    
    # 10. 総合結果
    print("=" * 60)
    if all_checks_passed:
        print("✅ 全てのチェックが成功しました。実行準備完了です！")
        print()
        print("🚀 実行コマンド:")
        print("   python judge_improved.py")
        return_code = 0
    else:
        print("❌ 一部のチェックが失敗しました。問題を解決してから再実行してください。")
        return_code = 1
    
    print("=" * 60)
    return return_code

@hydra.main(config_name="config", version_base=None, config_path="conf")
def main(cfg):
    # dryrunモードのチェック（Hydraオーバーライドとして処理）
    if hasattr(cfg, 'dryrun') and cfg.dryrun:
        return_code = run_dryrun_check(cfg)
        sys.exit(return_code)
    
    # 通常の実行モード
    from hle_benchmark import run_judge_results_improved
    run_judge_results_improved.main(cfg)

if __name__ == "__main__":
    main()
