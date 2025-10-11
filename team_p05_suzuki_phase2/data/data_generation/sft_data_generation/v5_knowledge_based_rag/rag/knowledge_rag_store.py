"""
Knowledge-based RAG Vector Store for v5 Data Generation
RAG store for searching and utilizing specialized knowledge (avoiding direct problem references)
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

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class KnowledgeDocument:
    """Knowledge document representation"""
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
    """RAG store for knowledge base"""
    
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
            index_path: Path to FAISS index file
            metadata_path: Path to metadata JSON file
            embedding_model: Name of embedding model
            device: 'cuda' or 'cpu'
            cache_dir: Cache directory
        """
        self.index_path = index_path or os.getenv(
            'KNOWLEDGE_INDEX_PATH',
            '/home/Competition2025/P05/shareP05/data_generation/knowledge_indexes/knowledge_index.faiss'
        )
        self.metadata_path = metadata_path or os.getenv(
            'KNOWLEDGE_METADATA_PATH',
            '/home/Competition2025/P05/shareP05/data_generation/knowledge_indexes/knowledge_metadata.json'
        )
        
        self.embedding_model_name = embedding_model
        self.device = device
        self.cache_dir = cache_dir
        
        # Initialize embedding model
        self.encoder = None
        self.dimension = None
        self._initialize_encoder()
        
        # Index and metadata
        self.index = None
        self.documents = []
        self.metadata = {}
        
        # Thread safety lock
        self._lock = threading.RLock()
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def _initialize_encoder(self):
        """Initialize embedding model"""
        try:
            logger.info(f"Initializing embedding model {self.embedding_model_name} on {self.device}")
            # Clear cache periodically to prevent memory buildup
            import shutil
            import time
            cache_timestamp_file = os.path.join(self.cache_dir, '.last_cleaned')
            should_clean = False
            
            # Check both time and cache size
            if os.path.exists(cache_timestamp_file):
                with open(cache_timestamp_file, 'r') as f:
                    last_cleaned = float(f.read().strip())
                # Clean cache if older than 7 days
                if time.time() - last_cleaned > 7 * 24 * 3600:
                    should_clean = True
                    logger.info("Cache is older than 7 days, cleaning...")
            
            # Also check cache directory size (clean if > 5GB)
            if os.path.exists(self.cache_dir):
                cache_size = sum(os.path.getsize(os.path.join(dirpath, f))
                               for dirpath, dirnames, filenames in os.walk(self.cache_dir)
                               for f in filenames) / (1024**3)  # Size in GB
                if cache_size > 5:
                    should_clean = True
                    logger.info(f"Cache size is {cache_size:.2f}GB, cleaning...")
            
            if should_clean:
                logger.info("Cleaning embedding cache...")
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
    
    def infer_subject_from_categories(self, categories: str) -> str:
        """Infer subject from ArXiv categories"""
        category_mapping = {
            'math': 'Mathematics',
            'cs': 'Computer Science',
            'stat': 'Statistics',
            'q-bio': 'BiologyAndMedicine',
            'physics': 'Physics',
            'quant-ph': 'Physics',
            'cond-mat': 'Physics',
            'hep-ph': 'Physics',
            'hep-th': 'Physics',
            'astro-ph': 'Physics',
            'gr-qc': 'Physics',
            'chem': 'Chemistry',
            'eess': 'Engineering',
            'econ': 'Economics',
            'q-fin': 'Economics',
        }

        if categories:
            # Convert categories to lowercase and split
            cats = categories.lower().split()
            for cat in cats:
                # Extract main category (e.g., "math.CO" -> "math")
                main_cat = cat.split('.')[0].replace(',', '').strip()
                if main_cat in category_mapping:
                    return category_mapping[main_cat]

        return 'Unknown'

    def load_index(self) -> bool:
        """Load existing index"""
        try:
            if not os.path.exists(self.index_path):
                logger.error(f"Index file not found: {self.index_path}")
                return False

            if not os.path.exists(self.metadata_path):
                logger.error(f"Metadata file not found: {self.metadata_path}")
                return False

            # Load FAISS index
            logger.info(f"Loading index from {self.index_path}")
            self.index = faiss.read_index(self.index_path)
            
            # Verify dimension matches encoder (BGE-large-en-v1.5 = 1024)
            if self.index.d != self.dimension:
                logger.error(f"Dimension mismatch: Index has {self.index.d} dimensions, "
                           f"but encoder has {self.dimension} dimensions")
                return False
            logger.info(f"Index dimension verified: {self.index.d}")
            
            # Load metadata
            logger.info(f"Loading metadata from {self.metadata_path}")
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            # Restore documents
            self.documents = []
            subject_stats = {}  # Subject statistics

            for doc_data in self.metadata.get('documents', []):
                # Extract doc_type from section_type or doc_id prefix
                doc_id = doc_data.get('doc_id', '')
                section_type = doc_data.get('section_type', '').lower()

                # Map section_type to doc_type
                if 'abstract' in section_type or doc_id.startswith('abstract_'):
                    doc_type = 'abstract'
                elif 'introduction' in section_type or 'background' in section_type:
                    doc_type = 'content'
                elif 'method' in section_type or 'approach' in section_type:
                    doc_type = 'content'
                elif 'result' in section_type or 'experiment' in section_type:
                    doc_type = 'content'
                elif 'conclusion' in section_type or 'discussion' in section_type:
                    doc_type = 'content'
                elif 'formula' in section_type or 'equation' in section_type:
                    doc_type = 'formula'
                elif 'theorem' in section_type or 'proof' in section_type:
                    doc_type = 'concept'
                else:
                    doc_type = 'content'  # Default to content

                # Prepare metadata - use doc_data directly as it contains all fields
                metadata = {
                    'arxiv_id': doc_data.get('arxiv_id', ''),
                    'title': doc_data.get('title', ''),
                    'authors': doc_data.get('authors', ''),
                    'categories': doc_data.get('categories', ''),
                    'section_title': doc_data.get('section_title', ''),
                    'section_type': doc_data.get('section_type', ''),
                    'section_index': doc_data.get('section_index', 0),
                    'chunk_index': doc_data.get('chunk_index', 0),
                    'published_date': doc_data.get('published_date', ''),
                    'license': doc_data.get('license', ''),
                    'quality_score': doc_data.get('quality_score', 0.0),
                    'is_chunked': doc_data.get('is_chunked', False)
                }

                # Use subject if already extracted, otherwise infer from categories
                subject = doc_data.get('subject', '')
                categories = doc_data.get('categories', '')

                if not subject or subject == 'Unknown':
                    subject = self.infer_subject_from_categories(categories)

                metadata['subject'] = subject
                subject_stats[subject] = subject_stats.get(subject, 0) + 1

                doc = KnowledgeDocument(
                    doc_id=doc_id,
                    content=doc_data.get('content', ''),
                    doc_type=doc_type,
                    metadata=metadata
                )
                self.documents.append(doc)
            
            logger.info(f"Loaded index with {len(self.documents)} documents")
            logger.info(f"Document types: {self._get_document_type_stats()}")

            # Display subject statistics
            if subject_stats:
                logger.info(f"Subject distribution: {dict(sorted(subject_stats.items(), key=lambda x: x[1], reverse=True)[:10])}")

            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def _get_document_type_stats(self) -> Dict[str, int]:
        """Get document type statistics"""
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
        Search knowledge
        
        Args:
            query: Search query
            k: Number of results to retrieve
            threshold: Minimum similarity score
            doc_type_filter: Filter by document type ('concept', 'pattern', etc.)
            subject_filter: Filter by subject
            difficulty_filter: Filter by difficulty
            
        Returns:
            List of search results
        """
        if not self.index or not self.encoder:
            logger.error("Index or encoder not initialized")
            return []
        
        try:
            # Embed query with efficient batch size
            # Dynamically adjust batch size based on query length
            query_len = max(1, len(query))
            batch_size = min(32, max(1, 16384 // query_len))  # Increased denominator for safer memory usage
            query_embedding = self.encoder.encode([query], 
                                                 show_progress_bar=False,
                                                 batch_size=batch_size,
                                                 convert_to_numpy=True)[0]
            query_embedding = query_embedding.astype('float32').reshape(1, -1)
            
            # Normalize for cosine similarity (BGE models work best with normalized vectors)
            # Check if index uses IndexFlatIP (inner product)
            if hasattr(self.index, '__class__') and 'IP' in self.index.__class__.__name__:
                faiss.normalize_L2(query_embedding)
            
            # Search with adaptive retrieval count for filtering
            with self._lock:  # Thread-safe search
                filter_count = sum([doc_type_filter is not None, 
                                  subject_filter is not None, 
                                  difficulty_filter is not None])
                # More filters require more candidates
                multiplier = 5 + filter_count * 3
                # Dynamic adjustment based on dataset size
                max_search_k = min(1000, self.index.ntotal // 2)  # At most half of total
                search_k = min(k * multiplier, self.index.ntotal, max_search_k)
                distances, indices = self.index.search(query_embedding, search_k)
            
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                # Strict boundary check
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
                
                # Threshold check
                if similarity_score < threshold:
                    continue
                
                # Filtering
                if doc_type_filter and doc.doc_type != doc_type_filter:
                    continue
                # Check for partial match if subject_filter is specified
                # e.g., subject_filter="Mathematics" matches "Computer Science, Mathematics"
                if subject_filter:
                    doc_subject = doc.metadata.get('subject', '')
                    # Check each comma-separated part
                    subject_parts = [s.strip() for s in doc_subject.split(',')]
                    if subject_filter not in subject_parts:
                        continue
                if difficulty_filter and doc.metadata.get('difficulty') != difficulty_filter:
                    continue
                
                # Set score
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
        """Get related concepts"""
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
        """Get solution patterns"""
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
        Build knowledge context based on topic
        
        Returns:
            (context string, list of used documents)
        """
        all_docs = []
        
        # 1. Search main concepts (abstracts and concepts)
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
        
        # 2. Search related content
        if question_type:
            content = self.search_knowledge(
                query=f"{topic} {question_type}",
                k=2,
                doc_type_filter='content',
                subject_filter=subject
            )
            all_docs.extend(content)
            
            # 3. Search formulas and theorems
            formulas = self.search_knowledge(
                query=topic,
                k=1,
                doc_type_filter='formula',
                subject_filter=subject
            )
            all_docs.extend(formulas)
        
        # 3. Build context
        context_parts = []
        current_length = 0
        
        # Sort by score
        all_docs.sort(key=lambda x: x.score, reverse=True)
        
        for doc in all_docs:
            # Format document
            if doc.doc_type == 'abstract':
                text = f"[Abstract] {doc.metadata.get('title', '')[:50]}: {doc.content[:300]}...\n"
            elif doc.doc_type == 'concept':
                text = f"[Concept] {doc.metadata.get('concept', '')}: {doc.content}\n"
            elif doc.doc_type == 'content':
                text = f"[Content] {doc.content[:400]}...\n"
            elif doc.doc_type == 'formula':
                text = f"[Formula] {doc.content}\n"
            else:
                text = f"{doc.content[:300]}...\n"
            
            # Length check
            if current_length + len(text) > max_context_length:
                break
            
            context_parts.append(text)
            current_length += len(text)
        
        context = "Reference Knowledge:\n" + "".join(context_parts)
        
        logger.info(f"Built knowledge context with {len(all_docs)} documents")
        return context, all_docs
    
    def format_knowledge_for_prompt(
        self,
        documents: List[KnowledgeDocument],
        include_metadata: bool = True,
        max_length: int = 2000
    ) -> str:
        """
        Format knowledge documents for prompt
        
        Args:
            documents: Documents to format
            include_metadata: Whether to include metadata
            max_length: Maximum character count
            
        Returns:
            Formatted string
        """
        if not documents:
            return ""
        
        # Input validation
        if not isinstance(documents, list):
            logger.error(f"Invalid documents type: {type(documents)}")
            return ""
        
        # Check max_length validity
        if max_length <= 0:
            logger.warning(f"Invalid max_length: {max_length}, using default 2000")
            max_length = 2000
        
        formatted_parts = []
        current_length = 0
        
        for i, doc in enumerate(documents, 1):
            # Document validity check
            if not isinstance(doc, KnowledgeDocument):
                logger.debug(f"Skipping invalid document at index {i}")
                continue
            
            # Content validation
            if not doc.content or not isinstance(doc.content, str):
                logger.debug(f"Skipping document with invalid content at index {i}")
                continue
            
            # Basic information
            part = f"[K{i}] "
            
            # Format by document type
            if doc.doc_type == 'abstract':
                title = doc.metadata.get('title', 'Unknown')
                part += f"Paper: {title[:80]}\n"
                part += f"Abstract: {doc.content[:300]}...\n"
                
                if include_metadata and doc.metadata.get('authors'):
                    part += f"Authors: {doc.metadata['authors'][:100]}\n"
                    
            elif doc.doc_type == 'concept':
                concept = doc.metadata.get('concept', 'Unknown')
                part += f"Concept: {concept}\n"
                part += f"Explanation: {doc.content}\n"
                
                if include_metadata and doc.metadata.get('paper_id'):
                    part += f"Source: {doc.metadata['paper_id']}\n"
                    
            elif doc.doc_type == 'content':
                title = doc.metadata.get('title', 'Unknown')
                chunk = doc.metadata.get('chunk_index', 0)
                part += f"Content (Paper: {title[:50]}, Part {chunk+1}):\n"
                part += f"{doc.content[:400]}...\n"
                
            elif doc.doc_type == 'formula':
                title = doc.metadata.get('title', 'Unknown')
                part += f"Formula (Paper: {title[:50]}):\n"
                part += f"{doc.content}\n"
                    
            else:
                part += f"{doc.content[:300]}...\n"
            
            # Add score
            part += f"(Relevance: {doc.score:.3f})\n\n"
            
            # Length check
            if current_length + len(part) > max_length:
                break
            
            formatted_parts.append(part)
            current_length += len(part)
        
        return "".join(formatted_parts)
    
    def get_statistics(self) -> Dict:
        """Get statistics"""
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
        
        # Subject and difficulty statistics
        for doc in self.documents:
            subject = doc.metadata.get('subject', 'unknown')
            difficulty = doc.metadata.get('difficulty', 'unknown')
            
            stats['subjects'][subject] = stats['subjects'].get(subject, 0) + 1
            stats['difficulties'][difficulty] = stats['difficulties'].get(difficulty, 0) + 1
        
        return stats
    
    def __del__(self):
        """Destructor - clean up resources"""
        try:
            # Clean up FAISS index
            if hasattr(self, 'index') and self.index is not None:
                # FAISS doesn't have explicit close method, but delete reference
                self.index = None
            
            # Clear documents and metadata
            if hasattr(self, 'documents'):
                self.documents.clear()
            if hasattr(self, 'metadata'):
                self.metadata.clear()
            
            # Clean up encoder
            if hasattr(self, 'encoder') and self.encoder is not None:
                self.encoder = None
                
            logger.debug("KnowledgeRAGStore resources cleaned up")
        except Exception as e:
            logger.debug(f"Error during cleanup: {e}")
    
    def close(self):
        """Explicit resource close method"""
        self.__del__()


# Utility functions
def test_knowledge_search():
    """Test knowledge search"""
    store = KnowledgeRAGStore()
    
    if not store.load_index():
        print("Failed to load index")
        return
    
    # Display statistics
    stats = store.get_statistics()
    print("\n=== Knowledge Base Statistics ===")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    # Test search
    test_queries = [
        "protein structure",
        "DNA replication",
        "chemical reaction rate",
        "quantum mechanics"
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