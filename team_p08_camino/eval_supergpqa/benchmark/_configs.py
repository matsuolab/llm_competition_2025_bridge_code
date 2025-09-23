from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """
    Hydraで読み込む設定を定義するデータクラス。
    conf/config.yamlのキーと対応しています。
    """
    # データセット設定
    dataset: str

    # プロバイダー設定 (vllm, openaiなど)
    provider: str
    base_url: str

    # モデル設定
    model: str
    max_completion_tokens: int
    reasoning: bool

    # パフォーマンス設定
    num_workers: int
    max_samples: Optional[int] = None # null許容のためOptionalを使用