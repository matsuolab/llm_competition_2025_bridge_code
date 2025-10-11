"""
ログユーティリティ
監査ログとサマリー出力
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import hashlib
import sys


class AuditLogger:
    """監査ログ管理クラス"""
    
    def __init__(self, audit_file: str = "./logs/audit_log.jsonl", summary_file: str = "./logs/summary.json"):
        """
        Args:
            audit_file: 監査ログファイルパス（JSONL形式）
            summary_file: サマリーファイルパス（JSON形式）
        """
        self.audit_file = Path(audit_file)
        self.summary_file = Path(summary_file)
        self.stats = {
            "total": 0,
            "pass": 0,
            "review": 0,
            "fail": 0,
            "ng_counts": {},
            "start_time": datetime.now().isoformat(),
            "config_hash": None,
            "errors": []
        }
        
        # logsフォルダ準備
        logs_dir = Path("./logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # ファイル準備
        self.audit_file.parent.mkdir(parents=True, exist_ok=True)
        self.summary_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 既存ファイルをバックアップ
        if self.audit_file.exists():
            backup_path = self.audit_file.with_suffix(
                f".{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            )
            self.audit_file.rename(backup_path)
            logging.info(f"既存監査ログをバックアップ: {backup_path}")
    
    def set_config_hash(self, config: Dict[str, Any]):
        """設定のハッシュ値を記録"""
        config_str = json.dumps(config, sort_keys=True)
        self.stats["config_hash"] = hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def log_sample(self, 
                  sample_id: str,
                  status: str,
                  ng_list: List[str],
                  scores: Optional[Dict[str, Any]] = None,
                  metadata: Optional[Dict[str, Any]] = None):
        """
        サンプルの処理結果をログ記録
        
        Args:
            sample_id: サンプルID
            status: pass/review/fail
            ng_list: NGフラグのリスト
            scores: スコア情報（類似度、長さ等）
            metadata: その他メタデータ
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "sample_id": sample_id,
            "status": status,
            "ng_list": ng_list,
            "scores": scores or {},
            "metadata": metadata or {}
        }
        
        # ファイルに追記
        with open(self.audit_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        
        # 統計更新
        self.stats["total"] += 1
        self.stats[status] = self.stats.get(status, 0) + 1
        
        for ng in ng_list:
            self.stats["ng_counts"][ng] = self.stats["ng_counts"].get(ng, 0) + 1
    
    def log_error(self, sample_id: str, error_msg: str, error_type: str = "unknown"):
        """
        エラーログ記録
        
        Args:
            sample_id: サンプルID
            error_msg: エラーメッセージ
            error_type: エラータイプ
        """
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "sample_id": sample_id,
            "error_type": error_type,
            "error_msg": error_msg
        }
        
        self.stats["errors"].append(error_entry)
        
        # エラーも監査ログに記録
        with open(self.audit_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                **error_entry,
                "status": "error"
            }, ensure_ascii=False) + "\n")
    
    def save_summary(self):
        """サマリー保存"""
        self.stats["end_time"] = datetime.now().isoformat()
        
        # 計算統計
        total = self.stats["total"]
        if total > 0:
            self.stats["pass_rate"] = self.stats["pass"] / total
            self.stats["review_rate"] = self.stats["review"] / total
            self.stats["fail_rate"] = self.stats["fail"] / total
        
        # NG統計をソート
        self.stats["ng_counts_sorted"] = sorted(
            self.stats["ng_counts"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # JSON保存
        with open(self.summary_file, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        
        logging.info(f"サマリー保存完了: {self.summary_file}")
    
    def print_summary(self):
        """サマリーを標準出力"""
        print("\n" + "="*60)
        print("フィルタリング結果サマリー")
        print("="*60)
        print(f"総サンプル数: {self.stats['total']:,}")
        print(f"  - Pass:   {self.stats['pass']:,} ({self.stats.get('pass_rate', 0):.1%})")
        print(f"  - Review: {self.stats['review']:,} ({self.stats.get('review_rate', 0):.1%})")
        print(f"  - Fail:   {self.stats['fail']:,} ({self.stats.get('fail_rate', 0):.1%})")
        
        if self.stats["ng_counts"]:
            print("\n頻出NGフラグ（上位10件）:")
            for ng, count in self.stats.get("ng_counts_sorted", [])[:10]:
                print(f"  - {ng}: {count:,}")
        
        if self.stats["errors"]:
            print(f"\nエラー発生数: {len(self.stats['errors'])}")
        
        print("="*60 + "\n")


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """
    ログ設定のセットアップ
    
    Args:
        level: ログレベル
        log_file: ログファイルパス（オプション）
    """
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        # logsフォルダが存在しない場合は作成
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers
    )
    
    # 外部ライブラリのログレベル調整
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("faiss").setLevel(logging.WARNING)


class ProgressTracker:
    """進捗トラッキング"""
    
    def __init__(self, total: int, desc: str = "Processing", update_interval: int = 10):
        """
        Args:
            total: 総件数
            desc: 説明文
            update_interval: 更新間隔（デフォルト: 10件ごと）
        """
        self.total = total
        self.desc = desc
        self.current = 0
        self.start_time = datetime.now()
        self.update_interval = update_interval
        self.logger = logging.getLogger(__name__)
        self.last_log_time = datetime.now()
    
    def _format_time(self, seconds: float) -> str:
        """秒を人間が読みやすい形式に変換"""
        if seconds < 60:
            return f"{seconds:.0f}秒"
        elif seconds < 3600:
            minutes = seconds / 60
            secs = seconds % 60
            return f"{minutes:.0f}分{secs:.0f}秒"
        else:
            hours = seconds / 3600
            minutes = (seconds % 3600) / 60
            return f"{hours:.0f}時間{minutes:.0f}分"
    
    def update(self, n: int = 1):
        """進捗更新"""
        self.current += n
        
        # 更新間隔ごと、または最後のサンプル、または5秒経過時に表示
        should_log = (
            self.current % self.update_interval == 0 or 
            self.current == self.total or
            (datetime.now() - self.last_log_time).total_seconds() > 5
        )
        
        if should_log:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            rate = self.current / elapsed if elapsed > 0 else 0
            eta = (self.total - self.current) / rate if rate > 0 else 0
            
            # 残り件数を計算
            remaining = self.total - self.current
            
            # 時間を読みやすい形式に変換
            elapsed_str = self._format_time(elapsed)
            eta_str = self._format_time(eta)
            
            # ログとして出力（他のログと混在しないように）
            progress_msg = (
                f"{self.desc}: {self.current:,}/{self.total:,} "
                f"({self.current/self.total:.1%}) - "
                f"残り: {remaining:,}件 - "
                f"経過: {elapsed_str} - "
                f"推定残り時間: {eta_str}"
            )
            
            # 処理速度も追加（件/分）
            if elapsed > 0:
                speed_per_min = (self.current / elapsed) * 60
                progress_msg += f" - 速度: {speed_per_min:.1f}件/分"
            
            self.logger.info(progress_msg)
            self.last_log_time = datetime.now()
            
            # 完了時は追加メッセージ
            if self.current == self.total:
                total_time = (datetime.now() - self.start_time).total_seconds()
                avg_time = total_time / self.total if self.total > 0 else 0
                total_time_str = self._format_time(total_time)
                self.logger.info(
                    f"{self.desc} 完了: 総処理時間 {total_time_str}, "
                    f"平均処理時間 {avg_time:.2f}秒/件"
                )
    
    def finish(self):
        """終了処理"""
        if self.current < self.total:
            self.current = self.total
            self.update(0)
