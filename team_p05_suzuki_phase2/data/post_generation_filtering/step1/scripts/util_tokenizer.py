"""
軽量トークナイザーユーティリティ
Qwen3-14Bなど軽量モデルのトークナイザーを使用
"""

import logging
from typing import Optional
from functools import lru_cache

logger = logging.getLogger(__name__)


class SimpleTokenizer:
    """単純な単語ベースのトークナイザー（軽量フォールバック）"""
    
    def count_tokens(self, text: str) -> int:
        """単語数をトークン数として返す"""
        return len(text.split())


@lru_cache(maxsize=1)
def get_tokenizer(model_name: Optional[str] = None):
    """
    トークナイザーを取得（軽量版）
    
    Args:
        model_name: モデル名（現在は単純トークナイザーのみ）
        
    Returns:
        トークナイザーインスタンス
    """
    # 現在は単純な単語ベースのトークナイザーを使用
    # 必要に応じて transformers ライブラリのトークナイザーに切り替え可能
    logger.info("Using simple word-based tokenizer (lightweight)")
    return SimpleTokenizer()
