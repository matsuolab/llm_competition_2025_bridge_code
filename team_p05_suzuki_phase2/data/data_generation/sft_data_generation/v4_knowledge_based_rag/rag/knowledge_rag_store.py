"""
Knowledge-based RAG Vector Store for v4 Data Generation
専門知識を検索して活用するRAGストア（問題の直接参照を避ける）
"""

import os
import json
import numpy as np
import faiss
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import hashlib
import pickle
from datetime import datetime
import logging
import threading

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class KnowledgeDocument:
    """知識ドキュメント"""
    doc_id: str
    content: str
    doc_type: str  # 'abstract', 'content', 'formula', 'concept'
    metadata: Dict[str, Any]
    score: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'doc_id': self.doc_id,
            'content': self.content,
            'doc_type': self.doc_type,
            'metadata': self.metadata,
            'score': self.score
        }


class KnowledgeRAGStore:
    """知識ベース用RAGストア"""
    
    def __init__(
        self,
        index_path: str = None,
        metadata_path: str = None,
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        device: str = "cpu",
        cache_dir: str = "/tmp/knowledge_rag_cache"
    ):
        """
        Args:
            index_path: FAISSインデックスファイルのパス
            metadata_path: メタデータJSONファイルのパス
            embedding_model: 埋め込みモデル名
            device: 'cuda' or 'cpu'
            cache_dir: キャッシュディレクトリ
        """
        self.index_path = index_path or os.getenv(
            'KNOWLEDGE_INDEX_PATH',
            '/home/Competition2025/P05/shareP05/knowledge_indexes_bge/knowledge_index.faiss'
        )
        self.metadata_path = metadata_path or os.getenv(
            'KNOWLEDGE_METADATA_PATH',
            '/home/Competition2025/P05/shareP05/knowledge_indexes_bge/knowledge_metadata.json'
        )
        
        self.embedding_model_name = embedding_model
        self.device = device
        self.cache_dir = cache_dir
        
        # エンベディングモデルの初期化
        self.encoder = None
        self.dimension = None
        self._initialize_encoder()
        
        # インデックスとメタデータ
        self.index = None
        self.documents = []
        self.metadata = {}
        
        # スレッドセーフティのためのロック
        self._lock = threading.RLock()
        
        # キャッシュディレクトリ作成
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def _initialize_encoder(self):
        """埋め込みモデルを初期化"""
        try:
            logger.info(f"Initializing embedding model {self.embedding_model_name} on {self.device}")
            # Clear cache periodically to prevent memory buildup
            import shutil
            import time
            cache_timestamp_file = os.path.join(self.cache_dir, '.last_cleaned')
            if os.path.exists(cache_timestamp_file):
                with open(cache_timestamp_file, 'r') as f:
                    last_cleaned = float(f.read().strip())
                # Clean cache if older than 7 days
                if time.time() - last_cleaned > 7 * 24 * 3600:
                    logger.info("Cleaning old embedding cache...")
                    shutil.rmtree(self.cache_dir, ignore_errors=True)
                    os.makedirs(self.cache_dir, exist_ok=True)
            
            self.encoder = SentenceTransformer(
                self.embedding_model_name,
                device=self.device,
                cache_folder=self.cache_dir
            )
            self.dimension = self.encoder.get_sentence_embedding_dimension()
            
            # Update cache timestamp
            with open(cache_timestamp_file, 'w') as f:
                f.write(str(time.time()))
            
            logger.info(f"Embedding model loaded successfully. Dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise
    
    def load_index(self) -> bool:
        """既存のインデックスを読み込み"""
        try:
            if not os.path.exists(self.index_path):
                logger.error(f"Index file not found: {self.index_path}")
                return False
            
            if not os.path.exists(self.metadata_path):
                logger.error(f"Metadata file not found: {self.metadata_path}")
                return False
            
            # FAISSインデックスを読み込み
            logger.info(f"Loading index from {self.index_path}")
            self.index = faiss.read_index(self.index_path)
            
            # Verify dimension matches encoder (BGE-large-en-v1.5 = 1024)
            if self.index.d != self.dimension:
                logger.error(f"Dimension mismatch: Index has {self.index.d} dimensions, "
                           f"but encoder has {self.dimension} dimensions")
                return False
            logger.info(f"Index dimension verified: {self.index.d}")
            
            # メタデータを読み込み
            logger.info(f"Loading metadata from {self.metadata_path}")
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            # ドキュメントを復元
            self.documents = []
            for doc_data in self.metadata.get('documents', []):
                # Extract doc_type from doc_id prefix or metadata
                doc_id = doc_data['doc_id']
                if doc_id.startswith('abstract_'):
                    doc_type = 'abstract'
                elif doc_id.startswith('content_'):
                    doc_type = 'content'
                elif doc_id.startswith('formula_'):
                    doc_type = 'formula'
                elif doc_id.startswith('concept_'):
                    doc_type = 'concept'
                else:
                    doc_type = doc_data.get('metadata', {}).get('type', 'unknown')
                
                doc = KnowledgeDocument(
                    doc_id=doc_id,
                    content=doc_data['content'],
                    doc_type=doc_type,
                    metadata=doc_data.get('metadata', {})
                )
                self.documents.append(doc)
            
            logger.info(f"Loaded index with {len(self.documents)} documents")
            logger.info(f"Document types: {self._get_document_type_stats()}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def _get_document_type_stats(self) -> Dict[str, int]:
        """ドキュメントタイプの統計を取得"""
        stats = {}
        for doc in self.documents:
            doc_type = doc.doc_type
            stats[doc_type] = stats.get(doc_type, 0) + 1
        return stats
    
    def search_knowledge(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.0,
        doc_type_filter: Optional[str] = None,
        subject_filter: Optional[str] = None,
        difficulty_filter: Optional[str] = None
    ) -> List[KnowledgeDocument]:
        """
        知識を検索
        
        Args:
            query: 検索クエリ
            k: 取得する結果数
            threshold: 最小類似度スコア
            doc_type_filter: ドキュメントタイプでフィルタ ('concept', 'pattern', etc.)
            subject_filter: 科目でフィルタ
            difficulty_filter: 難易度でフィルタ
            
        Returns:
            検索結果のリスト
        """
        if not self.index or not self.encoder:
            logger.error("Index or encoder not initialized")
            return []
        
        try:
            # クエリをエンベディング（効率的なバッチサイズで処理）
            # バッチサイズを動的に調整（長いクエリは小さく、短いクエリは大きく）
            # Avoid division by zero and ensure reasonable batch size
            query_len = max(1, len(query))
            batch_size = min(32, max(1, 8192 // query_len))
            query_embedding = self.encoder.encode([query], 
                                                 show_progress_bar=False,
                                                 batch_size=batch_size,
                                                 convert_to_numpy=True)[0]
            query_embedding = query_embedding.astype('float32').reshape(1, -1)
            
            # Normalize for cosine similarity (BGE models work best with normalized vectors)
            # Check if index uses IndexFlatIP (inner product)
            if hasattr(self.index, '__class__') and 'IP' in self.index.__class__.__name__:
                faiss.normalize_L2(query_embedding)
            
            # 検索（フィルタリングのため適応的に取得数を調整）
            # Adaptive search_k based on filters and available documents
            with self._lock:  # スレッドセーフな検索
                filter_count = sum([doc_type_filter is not None, 
                                  subject_filter is not None, 
                                  difficulty_filter is not None])
                # More filters require more candidates
                multiplier = 5 + filter_count * 3
                # データセットサイズに応じた動的調整
                max_search_k = min(1000, self.index.ntotal // 2)  # 最大でも全体の半分まで
                search_k = min(k * multiplier, self.index.ntotal, max_search_k)
                distances, indices = self.index.search(query_embedding, search_k)
            
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                # 厳密な境界チェック
                if not isinstance(idx, (int, np.integer)) or idx < 0 or idx >= len(self.documents):
                    logger.debug(f"Invalid index {idx} encountered, skipping")
                    continue
                
                doc = self.documents[idx]
                
                # Convert distance to similarity score based on index type
                if hasattr(self.index, '__class__') and 'IP' in self.index.__class__.__name__:
                    # For IndexFlatIP (inner product), distance is already the similarity
                    similarity_score = distance
                else:
                    # For IndexFlatL2, convert L2 distance to similarity
                    similarity_score = 1 / (1 + distance)
                
                # 閾値チェック
                if similarity_score < threshold:
                    continue
                
                # フィルタリング
                if doc_type_filter and doc.doc_type != doc_type_filter:
                    continue
                if subject_filter and doc.metadata.get('subject') != subject_filter:
                    continue
                if difficulty_filter and doc.metadata.get('difficulty') != difficulty_filter:
                    continue
                
                # スコアを設定
                doc.score = similarity_score
                results.append(doc)
                
                if len(results) >= k:
                    break
            
            logger.info(f"Found {len(results)} knowledge documents for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_related_concepts(self, concept: str, k: int = 3) -> List[KnowledgeDocument]:
        """関連する概念を取得"""
        # Search in both concept and abstract types
        results = []
        for doc_type in ['concept', 'abstract']:
            docs = self.search_knowledge(
                query=concept,
                k=k//2 + 1,
                doc_type_filter=doc_type
            )
            results.extend(docs)
        # Sort by score and limit to k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]
    
    def get_solution_patterns(self, problem_type: str, k: int = 3) -> List[KnowledgeDocument]:
        """解法パターンを取得"""
        # Search in content type for solution patterns
        return self.search_knowledge(
            query=problem_type + " solution method approach",
            k=k,
            doc_type_filter='content'
        )
    
    def build_knowledge_context(
        self,
        topic: str,
        subject: Optional[str] = None,
        question_type: Optional[str] = None,
        max_context_length: int = 2000
    ) -> Tuple[str, List[KnowledgeDocument]]:
        """
        トピックに基づいて知識コンテキストを構築
        
        Returns:
            (コンテキスト文字列, 使用したドキュメントリスト)
        """
        all_docs = []
        
        # 1. 主要な概念を検索 (abstracts and concepts)
        abstracts = self.search_knowledge(
            query=topic,
            k=2,
            doc_type_filter='abstract',
            subject_filter=subject
        )
        all_docs.extend(abstracts)
        
        concepts = self.search_knowledge(
            query=topic,
            k=2,
            doc_type_filter='concept',
            subject_filter=subject
        )
        all_docs.extend(concepts)
        
        # 2. 関連するコンテンツを検索
        if question_type:
            content = self.search_knowledge(
                query=f"{topic} {question_type}",
                k=2,
                doc_type_filter='content',
                subject_filter=subject
            )
            all_docs.extend(content)
            
            # 3. 数式・定理を検索
            formulas = self.search_knowledge(
                query=topic,
                k=1,
                doc_type_filter='formula',
                subject_filter=subject
            )
            all_docs.extend(formulas)
        
        # 3. コンテキストを構築
        context_parts = []
        current_length = 0
        
        # スコア順にソート
        all_docs.sort(key=lambda x: x.score, reverse=True)
        
        for doc in all_docs:
            # ドキュメントをフォーマット
            if doc.doc_type == 'abstract':
                text = f"【論文要約】{doc.metadata.get('title', '')[:50]}: {doc.content[:300]}...\n"
            elif doc.doc_type == 'concept':
                text = f"【概念】{doc.metadata.get('concept', '')}: {doc.content}\n"
            elif doc.doc_type == 'content':
                text = f"【内容】{doc.content[:400]}...\n"
            elif doc.doc_type == 'formula':
                text = f"【数式】{doc.content}\n"
            else:
                text = f"{doc.content[:300]}...\n"
            
            # 長さチェック
            if current_length + len(text) > max_context_length:
                break
            
            context_parts.append(text)
            current_length += len(text)
        
        context = "参考知識:\n" + "".join(context_parts)
        
        logger.info(f"Built knowledge context with {len(all_docs)} documents")
        return context, all_docs
    
    def format_knowledge_for_prompt(
        self,
        documents: List[KnowledgeDocument],
        include_metadata: bool = True,
        max_length: int = 2000
    ) -> str:
        """
        知識ドキュメントをプロンプト用にフォーマット
        
        Args:
            documents: フォーマットするドキュメント
            include_metadata: メタデータを含めるか
            max_length: 最大文字数
            
        Returns:
            フォーマットされた文字列
        """
        if not documents:
            return ""
        
        # 入力検証
        if not isinstance(documents, list):
            logger.error(f"Invalid documents type: {type(documents)}")
            return ""
        
        # 最大長の妥当性チェック
        if max_length <= 0:
            logger.warning(f"Invalid max_length: {max_length}, using default 2000")
            max_length = 2000
        
        formatted_parts = []
        current_length = 0
        
        for i, doc in enumerate(documents, 1):
            # ドキュメントの妥当性チェック
            if not isinstance(doc, KnowledgeDocument):
                logger.debug(f"Skipping invalid document at index {i}")
                continue
            
            # コンテンツの検証
            if not doc.content or not isinstance(doc.content, str):
                logger.debug(f"Skipping document with invalid content at index {i}")
                continue
            
            # 基本情報
            part = f"[K{i}] "
            
            # ドキュメントタイプ別のフォーマット
            if doc.doc_type == 'abstract':
                title = doc.metadata.get('title', 'Unknown')
                part += f"論文: {title[:80]}\n"
                part += f"要約: {doc.content[:300]}...\n"
                
                if include_metadata and doc.metadata.get('authors'):
                    part += f"著者: {doc.metadata['authors'][:100]}\n"
                    
            elif doc.doc_type == 'concept':
                concept = doc.metadata.get('concept', 'Unknown')
                part += f"概念: {concept}\n"
                part += f"説明: {doc.content}\n"
                
                if include_metadata and doc.metadata.get('paper_id'):
                    part += f"出典: {doc.metadata['paper_id']}\n"
                    
            elif doc.doc_type == 'content':
                title = doc.metadata.get('title', 'Unknown')
                chunk = doc.metadata.get('chunk_index', 0)
                part += f"内容 (論文: {title[:50]}, パート{chunk+1}):\n"
                part += f"{doc.content[:400]}...\n"
                
            elif doc.doc_type == 'formula':
                title = doc.metadata.get('title', 'Unknown')
                part += f"数式 (論文: {title[:50]}):\n"
                part += f"{doc.content}\n"
                    
            else:
                part += f"{doc.content[:300]}...\n"
            
            # スコアを追加
            part += f"(関連度: {doc.score:.3f})\n\n"
            
            # 長さチェック
            if current_length + len(part) > max_length:
                break
            
            formatted_parts.append(part)
            current_length += len(part)
        
        return "".join(formatted_parts)
    
    def get_statistics(self) -> Dict:
        """統計情報を取得"""
        if not self.documents:
            return {"status": "not_loaded"}
        
        stats = {
            "total_documents": len(self.documents),
            "index_dimension": self.dimension,
            "document_types": self._get_document_type_stats(),
            "subjects": {},
            "difficulties": {},
            "index_path": self.index_path,
            "metadata_path": self.metadata_path
        }
        
        # 科目と難易度の統計
        for doc in self.documents:
            subject = doc.metadata.get('subject', 'unknown')
            difficulty = doc.metadata.get('difficulty', 'unknown')
            
            stats['subjects'][subject] = stats['subjects'].get(subject, 0) + 1
            stats['difficulties'][difficulty] = stats['difficulties'].get(difficulty, 0) + 1
        
        return stats
    
    def __del__(self):
        """デストラクタ - リソースのクリーンアップ"""
        try:
            # FAISSインデックスのクリーンアップ
            if hasattr(self, 'index') and self.index is not None:
                # FAISSは明示的なクローズメソッドを持たないが、参照を削除
                self.index = None
            
            # ドキュメントとメタデータのクリア
            if hasattr(self, 'documents'):
                self.documents.clear()
            if hasattr(self, 'metadata'):
                self.metadata.clear()
            
            # エンコーダーのクリーンアップ
            if hasattr(self, 'encoder') and self.encoder is not None:
                self.encoder = None
                
            logger.debug("KnowledgeRAGStore resources cleaned up")
        except Exception as e:
            logger.debug(f"Error during cleanup: {e}")
    
    def close(self):
        """明示的なリソースクローズメソッド"""
        self.__del__()


# ユーティリティ関数
def test_knowledge_search():
    """知識検索のテスト"""
    store = KnowledgeRAGStore()
    
    if not store.load_index():
        print("Failed to load index")
        return
    
    # 統計情報を表示
    stats = store.get_statistics()
    print("\n=== Knowledge Base Statistics ===")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    # テスト検索
    test_queries = [
        "タンパク質の構造",
        "DNA複製",
        "化学反応の速度",
        "量子力学"
    ]
    
    for query in test_queries:
        print(f"\n=== Searching for: {query} ===")
        results = store.search_knowledge(query, k=3, threshold=0.3)
        
        for i, doc in enumerate(results, 1):
            print(f"\n{i}. [{doc.doc_type}] Score: {doc.score:.3f}")
            print(f"   Content: {doc.content[:100]}...")
            if doc.metadata.get('term'):
                print(f"   Term: {doc.metadata['term']}")
            if doc.metadata.get('subject'):
                print(f"   Subject: {doc.metadata['subject']}")


if __name__ == "__main__":
    test_knowledge_search()