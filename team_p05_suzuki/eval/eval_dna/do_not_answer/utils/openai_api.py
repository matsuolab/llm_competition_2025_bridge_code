# /home/Competition2025/P05/P05U025/team_suzuki/eval/eval_dna/do_not_answer/utils/openai_api.py

import os
import yaml
import time
from openai import OpenAI, OpenAIError

# グローバル変数としてクライアントとAPIキーを保持
__client = None
__api_key = None
__info_data_loaded = False

def _load_info_once():
    """
    設定ファイル(info.yaml)を一度だけ読み込む。
    環境変数を優先し、なければ同じディレクトリにあるinfo.yamlを試す。
    """
    global __api_key, __info_data_loaded
    if __info_data_loaded:
        return

    # 1. 環境変数を優先
    api_key_env = os.getenv("OPENAI_API_KEY")
    if api_key_env:
        __api_key = api_key_env
        __info_data_loaded = True
        return

    # 2. 同じディレクトリにあるinfo.yamlを読み込む
    try:
        # このスクリプト(openai_api.py)があるディレクトリのパスを取得
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # ★★★ 修正点: 同じディレクトリにあるファイルを指定 ★★★
        info_yaml_path = os.path.join(current_dir, 'info.yaml')

        if os.path.exists(info_yaml_path):
            with open(info_yaml_path, "r") as f:
                info_data = yaml.safe_load(f)
                # 一般的なキー名を両方試す
                yaml_key = info_data.get("openai_api_key") or info_data.get("OpenAI")
                if yaml_key:
                    __api_key = yaml_key
    except Exception as e:
        # 読み込みに失敗してもクラッシュせず、警告のみ表示
        print(f"警告: info.yaml の読み込み中にエラー: {e}")

    __info_data_loaded = True

def get_client() -> OpenAI:
    """
    OpenAIクライアントのシングルトンインスタンスを取得する。
    """
    global __client
    if __client is None:
        _load_info_once()  # 設定読み込みを試みる
        if __api_key:
            __client = OpenAI(api_key=__api_key)
        else:
            raise ValueError("OpenAI APIキーが見つかりません。環境変数またはinfo.yamlを確認してください。")
    return __client

def gpt(messages, model="gpt-4-0613", num_retries=3):
    """
    API呼び出しをリトライ機能付きで実行します。
    """
    client = get_client()
    for i in range(num_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages
            )
            return response.choices[0].message.content
        except OpenAIError as exception:
            print(f"GPT呼び出しエラー: {exception} ({i + 1}/{num_retries}回目)")
            if i < num_retries - 1:
                time.sleep(6)
    return "Error: GPT evaluation failed after multiple retries."