# config/settings.py
import os

class Settings:
    def __init__(self):
        # --- モデル設定 ---
        # vLLMサーバーでロードするモデル名を指定します。
        # まずは8Bモデルのような小規模なものからテストすることを推奨します。
        self.PRIMARY_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.VERIFICATION_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

        # --- ディレクトリ設定 ---
        self.INPUT_DIR = "input_pdfs"
        self.OUTPUT_DIR = "output_qa"
        self.TEMP_DIR = "temp"

        # 以下の設定はローカルvLLMサーバーとの通信では不要
        # self.OPEN_ROUTER_KEY = None
        # self.PROXIES = {}
        # self.SITE_URL = "local"
        # self.SITE_TITLE = "PDF-QA-Generator-vLLM"