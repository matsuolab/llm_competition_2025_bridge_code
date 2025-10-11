"""
重複検出処理
埋め込みベクトルによる類似度計算とANN検索
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import faiss
import hashlib
from sentence_transformers import SentenceTransformer
from datasketch import MinHash, MinHashLSH

logger = logging.getLogger(__name__)


class DuplicateDetector:
    """重複検出クラス"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 設定（embedding, dedupセクション）
        """
        self.config = config
        
        # 閾値設定
        self.hard_threshold = config['dedup']['hard_threshold']  # 0.92
        self.soft_threshold = config['dedup']['soft_threshold']  # 0.85
        
        # 高速ハッシュベース重複検出
        self.use_hash_filter = config['dedup'].get('use_hash_filter', True)
        self.hash_prefix_length = config['dedup'].get('hash_prefix_length', 100)
        self.question_hashes = {}  # ハッシュ -> サンプルIDのマッピング
        
        # 埋め込みモデル
        model_name = config['embedding']['model']
        logger.info(f"埋め込みモデル読み込み: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        self.embedding_dim = config['embedding']['dim']
        
        # FAISSインデックス
        self.index = None
        self.id_to_text = {}
        self.text_to_id = {}
        self.current_id = 0
        
        # 訓練用データ蓄積
        self.training_embeddings = []
        self.training_data = []  # (text, sample_id)のペア
        
        # MinHash LSH（高速な字面類似度チェック）
        self.use_minhash = config['dedup'].get('use_minhash', True)
        if self.use_minhash:
            self.lsh = MinHashLSH(threshold=0.85, num_perm=128)
            self.minhashes = {}
        
        # キャッシュ
        self.embedding_cache = {}
        
        self._initialize_index()
    
    def _initialize_index(self):
        """FAISSインデックスの初期化"""
        # 初期は小規模データ用のFlat indexで開始
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index_type = 'faiss_flat'
        
        logger.info(f"FAISSインデックス初期化完了: {self.index_type}")
    
    def _should_upgrade_to_ivf(self, total_vectors: int) -> bool:
        """
        IVFインデックスにアップグレードすべきかを判定
        
        Args:
            total_vectors: 現在のベクトル総数
            
        Returns:
            アップグレードすべきかどうか
        """
        target_index_type = self.config['embedding'].get('ann_index', 'faiss_ivf_pq')
        # FAISSのIVFPQには最低1500件のデータが推奨される
        min_data_for_ivf = 1500
        
        return (target_index_type == 'faiss_ivf_pq' and 
                self.index_type == 'faiss_flat' and 
                total_vectors >= min_data_for_ivf)
    
    def _create_and_train_ivf_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        IVFインデックスを作成して訓練
        
        Args:
            embeddings: 訓練用埋め込みベクトル
            
        Returns:
            訓練済みIVFインデックス
        """
        num_vectors = len(embeddings)
        
        # 動的にクラスタ数を決定（より保守的な設定）
        nlist = min(100, max(20, num_vectors // 20))  # データ数の1/20、最小20、最大100
        # PQパラメータmを小さくして256クラスタ問題を回避
        m = min(4, max(2, self.embedding_dim // 8))  # 埋め込み次元の1/8、最小2、最大4
        
        # クラスタ数がデータ数を超えないように調整
        nlist = min(nlist, num_vectors // 8)  # より保守的に設定
        nlist = max(10, nlist)  # 最小10
        
        # 十分なデータがない場合はエラー
        if num_vectors < nlist * 8:  # 1クラスタあたり最低8件
            raise ValueError(f"訓練データ不足: {num_vectors}件（必要: {nlist * 8}件以上）")
        
        logger.info(f"IVFパラメータ: nlist={nlist}, m={m}, データ数={num_vectors}")
        
        # IVFインデックス作成
        quantizer = faiss.IndexFlatL2(self.embedding_dim)
        ivf_index = faiss.IndexIVFPQ(
            quantizer, 
            self.embedding_dim, 
            nlist, 
            m, 
            8  # ビット数
        )
        
        # 訓練実行
        logger.info("IVFインデックス訓練開始...")
        ivf_index.train(embeddings)
        logger.info("IVFインデックス訓練完了")
        
        return ivf_index
    
    def _get_all_embeddings_from_flat_index(self) -> np.ndarray:
        """
        既存のFLATインデックスからすべてのベクトルを取得
        
        Returns:
            既存のすべての埋め込みベクトル
        """
        if self.index.ntotal == 0:
            return np.array([])
        
        # FLATインデックスからすべてのベクトルを取得
        all_vectors = self.index.reconstruct_n(0, self.index.ntotal)
        return all_vectors
    
    def _perform_ivf_upgrade(self):
        """
        蓄積されたデータと既存データを使ってIVFインデックスにアップグレード
        """
        if not self.training_embeddings:
            return
        
        logger.info(f"IVFインデックスへのアップグレード開始（蓄積データ: {len(self.training_embeddings)}件）")
        
        # 訓練用埋め込みを配列に変換
        training_array = np.array(self.training_embeddings)
        
        # 既存のFLATインデックスからすべてのベクトルを取得
        existing_vectors = None
        if self.index.ntotal > 0 and self.index_type == 'faiss_flat':
            try:
                existing_vectors = self._get_all_embeddings_from_flat_index()
                logger.info(f"既存FLATデータ: {len(existing_vectors)}件を取得")
            except Exception as e:
                logger.warning(f"既存データの取得に失敗: {e}")
                existing_vectors = None
        
        # 訓練データと既存データを結合
        if existing_vectors is not None and len(existing_vectors) > 0:
            all_embeddings = np.vstack([existing_vectors, training_array])
        else:
            all_embeddings = training_array
        
        logger.info(f"合計訓練データ: {len(all_embeddings)}件")
        
        try:
            # 新しいIVFインデックスを作成・訓練
            new_index = self._create_and_train_ivf_index(all_embeddings)
            
            # すべてのデータを新しいインデックスに追加
            logger.info("すべてのデータを新しいIVFインデックスに追加中...")
            new_index.add(all_embeddings)
            
            # id_to_text辞書とtext_to_id辞書を再構築
            logger.info("id_to_text辞書とtext_to_id辞書を再構築中...")
            old_id_to_text = self.id_to_text.copy()
            new_id_to_text = {}
            new_text_to_id = {}
            
            # 既存データのマッピングを保持
            if existing_vectors is not None and len(existing_vectors) > 0:
                # 既存データのマッピングを新しいインデックスに合わせて再構築
                for old_id, (text, sample_id) in old_id_to_text.items():
                    if old_id < len(existing_vectors):
                        new_id_to_text[old_id] = (text, sample_id)
                        new_text_to_id[text] = old_id
            
            # 訓練データのマッピングを追加
            offset = len(existing_vectors) if existing_vectors is not None else 0
            for i, (text, sample_id) in enumerate(self.training_data):
                new_id = offset + i
                new_id_to_text[new_id] = (text, sample_id)
                new_text_to_id[text] = new_id
            
            # インデックスとマッピングを置き換え
            self.index = new_index
            self.index_type = 'faiss_ivf_pq'
            self.id_to_text = new_id_to_text
            self.text_to_id = new_text_to_id
            
            # current_idを更新
            self.current_id = len(new_id_to_text)
            
            # 蓄積データをクリア
            self.training_embeddings.clear()
            self.training_data.clear()
            
            logger.info(f"IVFインデックスへのアップグレード完了 (総エントリ数: {self.current_id})")
            
        except ValueError as ve:
            # データ不足エラーの場合は蓄積を継続
            logger.warning(f"IVFアップグレード: {ve}")
            logger.info("データ蓄積を継続し、FLATインデックスを継続使用")
            # データ不足の場合はクリアせず蓄積を継続
        except Exception as e:
            logger.error(f"IVFアップグレードエラー: {e}")
            logger.info("FLATインデックスを継続使用")
            # その他のエラーの場合は蓄積データを保持（クリアしない）
            logger.info("蓄積データを保持し、後で再試行")
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        テキストの埋め込みベクトルを取得
        
        Args:
            text: 入力テキスト
            
        Returns:
            埋め込みベクトル
        """
        # キャッシュチェック
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        # 埋め込み生成（単一テキストなので進捗バー非表示）
        embedding = self.encoder.encode(text, convert_to_numpy=True, show_progress_bar=False)
        
        # 正規化（コサイン類似度用）
        embedding = embedding / np.linalg.norm(embedding)
        
        # キャッシュ保存
        self.embedding_cache[text_hash] = embedding
        
        return embedding
    
    def _create_minhash(self, text: str, num_perm: int = 128) -> MinHash:
        """
        MinHashの生成
        
        Args:
            text: 入力テキスト
            num_perm: パーミュテーション数
            
        Returns:
            MinHashオブジェクト
        """
        # n-gram生成（3-gram）
        ngrams = set()
        text_lower = text.lower()
        for i in range(len(text_lower) - 2):
            ngrams.add(text_lower[i:i+3])
        
        # MinHash生成
        mh = MinHash(num_perm=num_perm)
        for ngram in ngrams:
            mh.update(ngram.encode('utf-8'))
        
        return mh
    
    def add_to_index(self, text: str, sample_id: str):
        """
        インデックスにテキストを追加
        
        Args:
            text: 追加するテキスト
            sample_id: サンプルID
        """
        # ハッシュベースインデックスに追加
        if self.use_hash_filter:
            prefix = text[:self.hash_prefix_length].lower()
            prefix_hash = hashlib.md5(prefix.encode()).hexdigest()
            
            if prefix_hash in self.question_hashes:
                if sample_id not in self.question_hashes[prefix_hash]:
                    self.question_hashes[prefix_hash].append(sample_id)
            else:
                self.question_hashes[prefix_hash] = [sample_id]
        
        # 埋め込み取得
        embedding = self._get_embedding(text)
        
        # IVFアップグレードの判定（ただし実際のアップグレードはバッチ処理で行う）
        if self._should_upgrade_to_ivf(self.current_id + 1):
            # 訓練用データとして蓄積
            self.training_embeddings.append(embedding)
            self.training_data.append((text, sample_id))
            
            # 十分なデータが蓄積されたらIVFインデックスに移行
            if len(self.training_embeddings) >= 1500:
                self._perform_ivf_upgrade()
        
        # 現在のインデックスに追加（Flatインデックスまたは訓練済みIVF）
        try:
            self.index.add(embedding.reshape(1, -1))
        except Exception as e:
            logger.error(f"ベクトル追加エラー: {e}")
            # エラー時はFlatインデックスで再試行
            if self.index_type != 'faiss_flat':
                logger.info("Flatインデックスで再試行")
                self.index = faiss.IndexFlatL2(self.embedding_dim)
                self.index_type = 'faiss_flat'
                self.index.add(embedding.reshape(1, -1))
            else:
                raise
        
        # マッピング保存
        self.id_to_text[self.current_id] = (text, sample_id)
        self.text_to_id[text] = self.current_id
        
        # MinHash LSHに追加
        if self.use_minhash:
            mh = self._create_minhash(text)
            self.minhashes[sample_id] = mh
            self.lsh.insert(sample_id, mh)
        
        self.current_id += 1
    
    def search_similar(self, text: str, k: int = 10) -> List[Tuple[float, str, str]]:
        """
        類似テキストを検索
        
        Args:
            text: クエリテキスト
            k: 返す結果数
            
        Returns:
            [(類似度, テキスト, サンプルID), ...]のリスト
        """
        if self.index.ntotal == 0:
            return []
        
        # 埋め込み取得
        embedding = self._get_embedding(text)
        
        # FAISS検索
        distances, indices = self.index.search(
            embedding.reshape(1, -1), 
            min(k, self.index.ntotal)
        )
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # 無効なインデックス
                continue
            
            # id_to_textにIDが存在するか確認
            if idx not in self.id_to_text:
                logger.warning(f"FAISSインデックスID {idx} がid_to_text辞書に存在しません。")
                logger.debug(f"FAISSインデックス総数: {self.index.ntotal}, id_to_text辞書サイズ: {len(self.id_to_text)}")
                logger.debug(f"id_to_text辞書の有効なキー範囲: {min(self.id_to_text.keys()) if self.id_to_text else 'empty'} - {max(self.id_to_text.keys()) if self.id_to_text else 'empty'}")
                continue
            
            # L2距離をコサイン類似度に変換
            # cos_sim = 1 - (L2_dist^2 / 2) for normalized vectors
            cos_sim = 1 - (dist / 2)
            
            orig_text, sample_id = self.id_to_text[idx]
            results.append((cos_sim, orig_text, sample_id))
        
        return results
    
    def check_duplicate(self, text: str, sample_id: str) -> Tuple[str, Optional[float], Optional[str]]:
        """
        重複チェック
        
        Args:
            text: チェック対象テキスト
            sample_id: サンプルID
            
        Returns:
            (ステータス, 類似度, 重複元ID)
            ステータス: 'duplicate'/'near_duplicate'/'unique'
        """
        # 高速ハッシュベースの事前フィルタリング
        if self.use_hash_filter:
            # 質問の最初N文字のハッシュを計算
            prefix = text[:self.hash_prefix_length].lower()
            prefix_hash = hashlib.md5(prefix.encode()).hexdigest()
            
            # ハッシュが完全一致する場合のみ詳細チェック
            if prefix_hash in self.question_hashes:
                # 同じハッシュを持つサンプルIDを取得
                candidate_ids = self.question_hashes[prefix_hash]
                logger.debug(f"Hash match found for sample {sample_id}: {len(candidate_ids)} candidates")
                
                # 候補に対してのみ詳細な類似度チェックを実行
                for cand_id in candidate_ids:
                    if cand_id != sample_id and cand_id in [v[1] for v in self.id_to_text.values()]:
                        # 完全一致チェック
                        idx = [k for k, v in self.id_to_text.items() if v[1] == cand_id][0]
                        cand_text = self.id_to_text[idx][0]
                        if text == cand_text:
                            return 'duplicate', 1.0, cand_id
            else:
                # 新しいハッシュの場合は登録
                self.question_hashes[prefix_hash] = [sample_id]
        
        # 高速な完全一致チェック
        if text in self.text_to_id:
            text_id = self.text_to_id[text]
            if text_id in self.id_to_text:
                return 'duplicate', 1.0, self.id_to_text[text_id][1]
            else:
                logger.warning(f"text_to_id辞書のID {text_id} がid_to_text辞書に存在しません。")
        
        # MinHashによる高速チェック
        if self.use_minhash:
            mh = self._create_minhash(text)
            candidates = self.lsh.query(mh)
            
            if candidates:
                # 候補が見つかった場合、詳細な類似度計算
                for cand_id in candidates:
                    if cand_id != sample_id:
                        # Jaccard類似度が高い場合は重複の可能性大
                        jaccard = mh.jaccard(self.minhashes[cand_id])
                        if jaccard >= 0.9:
                            logger.debug(f"MinHash高類似度検出: {jaccard:.3f}")
        
        # 埋め込みベクトルによる類似度検索
        similar_items = self.search_similar(text, k=5)
        
        if not similar_items:
            return 'unique', None, None
        
        # 最も類似度の高いアイテム
        max_sim, _, dup_id = similar_items[0]
        
        # 自分自身は除外
        if dup_id == sample_id:
            if len(similar_items) > 1:
                max_sim, _, dup_id = similar_items[1]
            else:
                return 'unique', None, None
        
        # 閾値判定
        if max_sim >= self.hard_threshold:
            return 'duplicate', max_sim, dup_id
        elif max_sim >= self.soft_threshold:
            return 'near_duplicate', max_sim, dup_id
        else:
            return 'unique', max_sim, None
    
    def batch_add_to_index(self, texts: List[str], sample_ids: List[str]):
        """
        バッチでインデックスに追加
        
        Args:
            texts: テキストのリスト
            sample_ids: サンプルIDのリスト
        """
        if not texts:
            return
        
        # バッチ埋め込み（進捗バーは10件以上の場合のみ表示）
        show_progress = len(texts) >= 10
        embeddings = self.encoder.encode(texts, convert_to_numpy=True, show_progress_bar=show_progress)
        
        # 正規化
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # IVFアップグレードの判定
        total_after_add = self.current_id + len(texts)
        if self._should_upgrade_to_ivf(total_after_add) and len(embeddings) >= 1500:
            # 既存データも含めてIVFインデックスを作成
            logger.info(f"バッチデータでIVFインデックス作成（新規データ数: {len(embeddings)}）")
            try:
                # 既存データを取得
                existing_vectors = None
                if self.index.ntotal > 0 and self.index_type == 'faiss_flat':
                    existing_vectors = self._get_all_embeddings_from_flat_index()
                    logger.info(f"既存データ: {len(existing_vectors)}件を取得")
                
                # 既存データと新規データを結合
                if existing_vectors is not None and len(existing_vectors) > 0:
                    all_embeddings = np.vstack([existing_vectors, embeddings])
                else:
                    all_embeddings = embeddings
                
                logger.info(f"合計データ数: {len(all_embeddings)}件でIVFインデックス作成")
                new_index = self._create_and_train_ivf_index(all_embeddings)
                new_index.add(all_embeddings)
                self.index = new_index
                self.index_type = 'faiss_ivf_pq'
                logger.info("IVFインデックス作成完了")
            except Exception as e:
                logger.error(f"IVFインデックス作成エラー: {e}")
                logger.info("Flatインデックスを継続使用")
        
        # バッチ追加
        try:
            self.index.add(embeddings)
        except Exception as e:
            logger.error(f"バッチ追加エラー: {e}")
            # Flatインデックスで再試行
            if self.index_type != 'faiss_flat':
                logger.info("Flatインデックスで再試行")
                self.index = faiss.IndexFlatL2(self.embedding_dim)
                self.index_type = 'faiss_flat'
                self.index.add(embeddings)
            else:
                raise
        
        # マッピング保存
        for text, sample_id, embedding in zip(texts, sample_ids, embeddings):
            self.id_to_text[self.current_id] = (text, sample_id)
            self.text_to_id[text] = self.current_id
            
            # キャッシュ保存
            text_hash = hashlib.md5(text.encode()).hexdigest()
            self.embedding_cache[text_hash] = embedding
            
            # MinHash追加
            if self.use_minhash:
                mh = self._create_minhash(text)
                self.minhashes[sample_id] = mh
                self.lsh.insert(sample_id, mh)
            
            self.current_id += 1
        
        logger.info(f"{len(texts)}件のテキストをインデックスに追加完了")
    
    def save_index(self, path: str):
        """インデックスの保存"""
        import pickle
        
        # FAISSインデックス保存
        faiss.write_index(self.index, f"{path}.faiss")
        
        # メタデータ保存
        metadata = {
            'id_to_text': self.id_to_text,
            'text_to_id': self.text_to_id,
            'current_id': self.current_id,
            'minhashes': self.minhashes if self.use_minhash else None
        }
        
        with open(f"{path}.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"インデックス保存完了: {path}")
    
    def load_index(self, path: str):
        """インデックスの読み込み"""
        import pickle
        
        # FAISSインデックス読み込み
        self.index = faiss.read_index(f"{path}.faiss")
        
        # メタデータ読み込み
        with open(f"{path}.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        self.id_to_text = metadata['id_to_text']
        self.text_to_id = metadata['text_to_id']
        self.current_id = metadata['current_id']
        
        if metadata.get('minhashes') and self.use_minhash:
            self.minhashes = metadata['minhashes']
            # LSH再構築
            self.lsh = MinHashLSH(threshold=0.85, num_perm=128)
            for sample_id, mh in self.minhashes.items():
                self.lsh.insert(sample_id, mh)
        
        logger.info(f"インデックス読み込み完了: {path}")
