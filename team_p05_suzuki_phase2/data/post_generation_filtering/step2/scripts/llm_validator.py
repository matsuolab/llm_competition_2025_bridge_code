"""
DeepSeek-R1を使用した必要最小限の3つの検証：
1. 論理的整合性チェック
2. 意味的矛盾検出
3. 回答品質チェック
"""

# --- must be at the very top ---
import os
import multiprocessing as mp

# 余計なバックエンドを抑止
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("VLLM_USE_V1", "0")

# これを最初に追加（必ず vllm を import する前）
os.environ.pop("VLLM_MAX_CONCURRENT_WORKERS", None)

# vLLM ワーカーは fork に（Linux 前提）
try:
    mp.set_start_method("fork", force=True)
except RuntimeError:
    pass
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "fork")

# --- add above this line ---

import re
import logging
from typing import Dict, Tuple, Optional, List

# ===== OOM対策: 親・子共通で有効 =====
os.environ.setdefault("PYTORCH_DISABLE_DYNAMO", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512,garbage_collection_threshold:0.9")

# ulimit 相当（権限なければ無視）
def _raise_rlimits():
    try:
        import resource as r
        soft, hard = r.getrlimit(r.RLIMIT_AS)
        if soft < hard: r.setrlimit(r.RLIMIT_AS, (hard, hard))
        soft, hard = r.getrlimit(r.RLIMIT_MEMLOCK)
        if soft < hard: r.setrlimit(r.RLIMIT_MEMLOCK, (hard, hard))
        soft, hard = r.getrlimit(r.RLIMIT_NOFILE)
        tgt = min(1048576, hard)
        if soft < tgt: r.setrlimit(r.RLIMIT_NOFILE, (tgt, hard))
    except Exception:
        pass

_raise_rlimits()

# 子プロセスでは __name__ は "__mp_main__" になる
def is_worker_process() -> bool:
    return __name__ == "__mp_main__"

# --- 追加：重い依存を親でプリロードして子にCoW共有させる ---
def _preload_heavy_modules():
    import importlib  # noqa: F401
    try:
        import sympy      # noqa: F401
    except ImportError:
        pass
    try:
        from torch.utils import checkpoint  # noqa: F401
        from torch.fx.experimental import symbolic_shapes  # noqa: F401
    except ImportError:
        pass
    try:
        import transformers  # noqa: F401
    except ImportError:
        pass
    # ここでは torch.cuda を絶対に触らない（CUDAコンテキストが親に張り付くため）

if not is_worker_process():
    _preload_heavy_modules()
# --- ここまで追加 ---

# vLLM使用時のインポート
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logging.warning("vLLMが利用できません。LLMベースのチェックは無効です。")

logger = logging.getLogger(__name__)

# グローバルなLLMインスタンス（シングルトン）
_GLOBAL_LLM_INSTANCE = None
_GLOBAL_SAMPLING_PARAMS = None


def _get_or_create_llm(config: Dict):
    """LLMインスタンスの取得または作成（シングルトン）"""
    global _GLOBAL_LLM_INSTANCE, _GLOBAL_SAMPLING_PARAMS
    
    if _GLOBAL_LLM_INSTANCE is not None:
        return _GLOBAL_LLM_INSTANCE, _GLOBAL_SAMPLING_PARAMS
    
    if not VLLM_AVAILABLE:
        return None, None
    
    llm_config = config.get('llm_config', {})
    
    try:
        # run_vllm_with_memlog_fix.pyと完全に同じ設定を使用
        logger.info("DeepSeek-R1モデル初期化中: QuixiAI/DeepSeek-R1-0528-AWQ")
        
        _GLOBAL_LLM_INSTANCE = LLM(
            model="QuixiAI/DeepSeek-R1-0528-AWQ",
            quantization="awq_marlin",
            dtype="float16",
            tensor_parallel_size=llm_config.get('tensor_parallel_size', 8),
            max_model_len=llm_config.get('max_model_len', 16384),
            enforce_eager=True,
            gpu_memory_utilization=llm_config.get('gpu_memory_utilization', 0.90)
        )
        
        # 検証用パラメータ（低温度で確実性重視）
        _GLOBAL_SAMPLING_PARAMS = SamplingParams(
            temperature=llm_config.get('temperature', 0.1),  # 低温度で一貫性を確保
            max_tokens=llm_config.get('max_tokens', 512),   # 十分な長さを確保
            top_p=0.95,
            stop=["\n\n", "---"]  # 余計な生成を防ぐ
        )
        
        logger.info("LLMモデル初期化完了")
        return _GLOBAL_LLM_INSTANCE, _GLOBAL_SAMPLING_PARAMS
        
    except Exception as e:
        logger.error(f"LLMモデル初期化エラー: {e}")
        return None, None


class LLMValidator:
    """DeepSeek-R1を使用した必要最小限の3つの検証クラス"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 設定（llm_configセクション）
        """
        self.config = config
        # シングルトンのLLMインスタンスを取得
        self.llm_model, self.sampling_params = _get_or_create_llm(config)
    
    def validate_logical_consistency(self, question: str, think: str, answer: str) -> Tuple[bool, Optional[str], float]:
        """
        論理的整合性の包括的検証
        
        Args:
            question: 問題文
            think: 思考過程
            answer: 回答
            
        Returns:
            (有効か, 問題説明, 信頼度スコア)
        """
        if not self.llm_model:
            return True, None, 0.0  # LLM無しの場合はスキップ
        
        # validate_and_fix_dataset_hf.pyのアプローチを採用
        prompt = f"""You are an expert at validating academic problem solutions for logical consistency.
Your task is to check if the thinking process (reasoning) logically leads to the given answer, and if both are correct for the given question.

Check for:
1. The thinking process must be logically sound and free from contradictions
2. The thinking process must correctly lead to the stated answer
3. The answer must be correct for the given question
4. For multiple-choice questions, verify the selected option is justified
5. For short-answer questions, verify the answer is accurate

Output format:
VALID - if everything is logically consistent and correct
INVALID: [brief explanation of the issue] - if there are logical inconsistencies or errors

Be strict but fair. Minor stylistic issues are acceptable, but logical errors are not.

Question: {question}

Thinking Process: {think}

Answer: {answer}

Is this logically consistent and correct?
Validation Result:"""
        
        try:
            # LLM推論
            outputs = self.llm_model.generate([prompt], self.sampling_params)
            response = outputs[0].outputs[0].text.strip()
            
            # 判定解析
            if response.startswith("VALID"):
                return True, None, 1.0
            elif response.startswith("INVALID:"):
                issue = response.replace("INVALID:", "").strip()
                # 信頼度スコアを問題の深刻度から推定（簡略化）
                confidence = 0.8
                return False, issue, confidence
            else:
                # 予期しない形式
                logger.warning(f"Unexpected validation result: {response[:100]}")
                return True, None, 0.5
                
        except Exception as e:
            logger.error(f"LLM検証エラー: {e}")
            return True, None, 0.0
    
    def check_answer_quality(self, question: str, answer: str, answer_type: str) -> Tuple[str, Optional[str]]:
        """
        回答品質の詳細チェック
        
        Args:
            question: 問題文
            answer: 回答
            answer_type: 回答タイプ
            
        Returns:
            (品質レベル, 問題説明)
        """
        if not self.llm_model:
            return "unknown", None
        
        # 回答タイプ別の検証プロンプト
        if answer_type == "multipleChoice":
            prompt = f"""Evaluate if this multiple-choice answer is properly formatted and justified.

Question: {question}
Answer: {answer}

Check:
1. Is a clear choice (A-F) specified?
2. Is only ONE option selected?
3. Does the answer make sense for the question?

Respond with:
GOOD - if properly formatted
POOR: [reason] - if there are issues
Result:"""
        else:
            prompt = f"""Evaluate if this answer properly addresses the question.

Question: {question}
Answer: {answer}

Check:
1. Does it directly answer what was asked?
2. Is it complete (not cut off)?
3. Is it factually reasonable?

Respond with:
GOOD - if properly addresses the question
POOR: [reason] - if there are issues
Result:"""
        
        try:
            outputs = self.llm_model.generate([prompt], self.sampling_params)
            response = outputs[0].outputs[0].text.strip()
            
            if response.startswith("GOOD"):
                return "good", None
            elif response.startswith("POOR:"):
                issue = response.replace("POOR:", "").strip()
                return "poor", issue
            else:
                return "unknown", None
                
        except Exception as e:
            logger.error(f"回答品質チェックエラー: {e}")
            return "unknown", str(e)
    
    def detect_contradictions(self, text: str) -> List[str]:
        """
        テキスト内の矛盾検出
        
        Args:
            text: チェック対象テキスト
            
        Returns:
            矛盾のリスト
        """
        if not self.llm_model:
            return []
        
        # 構造化されたプロンプトで明確な判定を要求
        prompt = f"""Task: Identify ONLY genuine logical contradictions where the text makes conflicting claims about the same fact.

Definition of contradiction: When the text states X is true and also states X is false (or not-X is true).

Text to analyze:
{text}

Instructions:
1. A contradiction exists ONLY when the same text makes incompatible claims
2. Analyzing incorrect options in multiple-choice questions is NOT a contradiction
3. Comparing different options or explaining why something is wrong is NOT a contradiction

Response format:
- If there are genuine contradictions, list them starting with "CONTRADICTION:"
- If there are NO genuine contradictions, respond with exactly: "NO_CONTRADICTIONS_FOUND"

Analysis:"""
        
        try:
            outputs = self.llm_model.generate([prompt], self.sampling_params)
            response = outputs[0].outputs[0].text.strip()
            
            # 明確な「矛盾なし」の判定
            if "NO_CONTRADICTIONS_FOUND" in response or "no contradiction" in response.lower():
                return []
            
            # CONTRADICTION:で始まる行のみを抽出
            contradictions = []
            for line in response.split('\n'):
                line = line.strip()
                if line.upper().startswith('CONTRADICTION:'):
                    # "CONTRADICTION:" プレフィックスを除去
                    contradiction = line[len('CONTRADICTION:'):].strip()
                    if contradiction:
                        contradictions.append(contradiction)
            
            return contradictions
            
        except Exception as e:
            logger.error(f"矛盾検出エラー: {e}")
            return []
    
    def cleanup(self):
        """LLMリソースのクリーンアップ"""
        global _GLOBAL_LLM_INSTANCE, _GLOBAL_SAMPLING_PARAMS
        
        if _GLOBAL_LLM_INSTANCE is not None:
            try:
                logger.info("LLMインスタンスをクリーンアップ中...")
                # vLLMのクリーンアップ処理
                if hasattr(_GLOBAL_LLM_INSTANCE, 'llm_engine'):
                    if hasattr(_GLOBAL_LLM_INSTANCE.llm_engine, 'model_executor'):
                        if hasattr(_GLOBAL_LLM_INSTANCE.llm_engine.model_executor, 'shutdown'):
                            _GLOBAL_LLM_INSTANCE.llm_engine.model_executor.shutdown()
                
                # PyTorch分散処理のクリーンアップ
                try:
                    import torch.distributed as dist
                    if dist.is_initialized():
                        dist.destroy_process_group()
                        logger.info("PyTorch分散処理グループを破棄")
                except Exception as e:
                    logger.debug(f"分散処理グループの破棄をスキップ: {e}")
                
                # グローバル変数をクリア
                _GLOBAL_LLM_INSTANCE = None
                _GLOBAL_SAMPLING_PARAMS = None
                logger.info("LLMクリーンアップ完了")
                
            except Exception as e:
                logger.warning(f"LLMクリーンアップ中の警告: {e}")


def cleanup_llm_resources():
    """グローバルなLLMリソースのクリーンアップ関数"""
    global _GLOBAL_LLM_INSTANCE
    
    if _GLOBAL_LLM_INSTANCE is not None:
        validator = LLMValidator({'llm_config': {}})
        validator.cleanup()