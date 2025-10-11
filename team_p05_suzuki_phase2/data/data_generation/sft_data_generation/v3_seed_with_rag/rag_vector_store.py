#!/usr/bin/env python3
"""
RAG Vector Store Module for Data Generation
===========================================

This module provides vector store and retrieval functionality for RAG-enhanced data generation.
It supports building indices from existing datasets and retrieving similar problems.

Author: Generated with Claude Code
"""

import json
import os
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset

# Try to import FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("WARNING: FAISS not installed. Install with: pip install faiss-cpu or faiss-gpu")

# Try to import sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("WARNING: sentence-transformers not installed. Install with: pip install sentence-transformers")


@dataclass
class RetrievedDocument:
    """Class representing a retrieved document with metadata."""
    doc_id: str
    text: str
    score: float
    metadata: Dict[str, Any]
    chunk_id: Optional[int] = None
    source: Optional[str] = None


class RAGVectorStore:
    """Vector store for RAG-enhanced data generation."""
    
    def __init__(
        self,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        index_path: Optional[str] = None,
        dimension: int = 384,
        use_gpu: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the RAG Vector Store.
        
        Args:
            embedding_model: Name of the sentence transformer model
            index_path: Path to save/load the FAISS index
            dimension: Dimension of embeddings (auto-detected if None)
            use_gpu: Whether to use GPU for embeddings and FAISS
            cache_dir: Directory to cache models and indices
        """
        self.embedding_model_name = embedding_model
        self.index_path = index_path or "vector_store/rag_index.faiss"
        self.dimension = dimension
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.cache_dir = cache_dir or "vector_store/cache"
        
        # Initialize embedding model with memory management
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            # Force garbage collection before loading model
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            try:
                # Try GPU first if available and requested
                device = 'cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu'
                print(f"Initializing embedding model {embedding_model} on {device}")
                self.encoder = SentenceTransformer(
                    embedding_model,
                    device=device,
                    cache_folder=self.cache_dir
                )
                # Update dimension based on model
                self.dimension = self.encoder.get_sentence_embedding_dimension()
                print(f"Embedding model loaded successfully. Dimension: {self.dimension}")
            except Exception as e:
                # Fallback to CPU if GPU initialization fails
                if self.use_gpu:
                    print(f"GPU initialization failed: {e}")
                    print("Falling back to CPU for embedding model...")
                    self.use_gpu = False
                    self.encoder = SentenceTransformer(
                        embedding_model,
                        device='cpu',
                        cache_folder=self.cache_dir
                    )
                    self.dimension = self.encoder.get_sentence_embedding_dimension()
                    print(f"Embedding model loaded on CPU. Dimension: {self.dimension}")
                else:
                    raise e
        else:
            self.encoder = None
            print("WARNING: Cannot initialize embeddings without sentence-transformers")
        
        # Initialize FAISS index
        self.index = None
        self.documents = []  # Store documents with metadata
        self.doc_embeddings = []  # Store embeddings
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
    
    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def encode_texts(self, texts: List[str], batch_size: int = 8, show_progress: bool = True) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings
        """
        if not self.encoder:
            raise RuntimeError("Embedding model not initialized")
        
        embeddings = []
        
        # Process in batches
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding texts", total=len(texts)//batch_size + 1)
        
        for i in iterator:
            batch = texts[i:i+batch_size]
            batch_embeddings = self.encoder.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            embeddings.append(batch_embeddings)
            
            # Periodic memory cleanup for large datasets
            if len(embeddings) % 10 == 0:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return np.vstack(embeddings).astype('float32')
    
    def build_index_from_dataset(
        self,
        dataset_name: Optional[str] = None,
        jsonl_path: Optional[str] = None,
        text_field: str = "question",
        metadata_fields: Optional[List[str]] = None,
        limit: Optional[int] = None,
        chunk_size: int = 300,
        chunk_overlap: int = 50
    ):
        """
        Build FAISS index from a dataset.
        
        Args:
            dataset_name: Hugging Face dataset name
            jsonl_path: Path to JSONL file (alternative to dataset_name)
            text_field: Field to use for embedding
            metadata_fields: Additional fields to store as metadata
            limit: Maximum number of documents to index
            chunk_size: Size of text chunks (in tokens)
            chunk_overlap: Overlap between chunks
        """
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS is not installed")
        
        print(f"Building vector index from {'dataset' if dataset_name else 'file'}...")
        
        # Load data
        documents = []
        if dataset_name:
            # Load from Hugging Face
            hf_token = os.getenv("HF_TOKEN")
            ds = load_dataset(dataset_name, token=hf_token)
            if 'train' in ds:
                data = list(ds['train'])
            else:
                data = list(ds[list(ds.keys())[0]])
        elif jsonl_path:
            # Load from JSONL file
            data = []
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
        else:
            raise ValueError("Either dataset_name or jsonl_path must be provided")
        
        # Apply limit if specified
        if limit:
            data = data[:limit]
        
        print(f"Processing {len(data)} documents...")
        
        # Default metadata fields
        if metadata_fields is None:
            metadata_fields = ["data_id", "subject", "question_type", "answer", "think"]
        
        # Process documents and create chunks
        all_texts = []
        all_metadata = []
        
        for idx, item in enumerate(tqdm(data, desc="Processing documents")):
            # Get main text
            text = str(item.get(text_field, ""))
            
            # Add think content if available for better context
            if "think" in item and item["think"]:
                text = f"{text}\n\nSolution:\n{item['think']}"
            
            # Create metadata
            metadata = {
                "doc_index": idx,
                "source": dataset_name or jsonl_path
            }
            for field in metadata_fields:
                if field in item:
                    metadata[field] = item[field]
            
            # For now, we'll use the whole text as one chunk
            # In production, you might want to implement proper chunking
            all_texts.append(text)
            all_metadata.append(metadata)
        
        # Encode all texts
        print(f"Encoding {len(all_texts)} texts...")
        embeddings = self.encode_texts(all_texts, show_progress=True)
        
        # Create FAISS index
        print("Building FAISS index...")
        if self.use_gpu and FAISS_AVAILABLE:
            # Use GPU index if available
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            except:
                print("GPU index creation failed, falling back to CPU")
                self.index = faiss.IndexFlatIP(self.dimension)
        else:
            self.index = faiss.IndexFlatIP(self.dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store documents and embeddings
        self.documents = all_metadata
        self.doc_embeddings = embeddings
        
        print(f"Index built with {len(self.documents)} documents")
        
        # Save index and metadata
        self.save_index()
    
    def save_index(self):
        """Save FAISS index and metadata to disk."""
        if not self.index:
            print("No index to save")
            return
        
        print(f"Saving index to {self.index_path}...")
        
        # Save FAISS index
        if self.use_gpu and hasattr(self.index, 'index'):
            # Convert GPU index to CPU for saving
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, self.index_path)
        else:
            faiss.write_index(self.index, self.index_path)
        
        # Save metadata
        metadata_path = self.index_path.replace('.faiss', '_metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'embeddings': self.doc_embeddings,
                'dimension': self.dimension,
                'model_name': self.embedding_model_name
            }, f)
        
        print(f"Index and metadata saved successfully")
    
    def load_index(self):
        """Load FAISS index and metadata from disk."""
        if not os.path.exists(self.index_path):
            print(f"Index file not found: {self.index_path}")
            return False
        
        print(f"Loading index from {self.index_path}...")
        
        # Load FAISS index
        self.index = faiss.read_index(self.index_path)
        
        # Convert to GPU if needed
        if self.use_gpu and FAISS_AVAILABLE:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            except:
                print("Failed to move index to GPU, using CPU")
        
        # Load metadata
        metadata_path = self.index_path.replace('.faiss', '_metadata.pkl')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                self.documents = metadata['documents']
                self.doc_embeddings = metadata['embeddings']
                self.dimension = metadata['dimension']
                
                # Check model compatibility
                if metadata['model_name'] != self.embedding_model_name:
                    print(f"WARNING: Index was built with {metadata['model_name']}, "
                          f"but current model is {self.embedding_model_name}")
        
        print(f"Loaded index with {len(self.documents)} documents")
        return True
    
    def search(
        self,
        query: Union[str, List[str]],
        k: int = 5,
        threshold: float = 0.0,
        filter_fn: Optional[callable] = None
    ) -> List[List[RetrievedDocument]]:
        """
        Search for similar documents.
        
        Args:
            query: Query text or list of queries
            k: Number of results to return
            threshold: Minimum similarity threshold
            filter_fn: Optional function to filter results
            
        Returns:
            List of retrieved documents (list of lists if multiple queries)
        """
        if not self.index:
            raise RuntimeError("Index not initialized. Call build_index_from_dataset or load_index first.")
        
        # Convert single query to list
        if isinstance(query, str):
            queries = [query]
            single_query = True
        else:
            queries = query
            single_query = False
        
        # Encode queries
        query_embeddings = self.encode_texts(queries, show_progress=False)
        faiss.normalize_L2(query_embeddings)
        
        # Search
        scores, indices = self.index.search(query_embeddings, k * 2)  # Get more results for filtering
        
        # Process results
        all_results = []
        for q_idx, (q_scores, q_indices) in enumerate(zip(scores, indices)):
            results = []
            for score, idx in zip(q_scores, q_indices):
                if idx < 0:  # Invalid index
                    continue
                
                if score < threshold:
                    continue
                
                # Get document metadata
                doc_metadata = self.documents[idx]
                
                # Apply filter if provided
                if filter_fn and not filter_fn(doc_metadata):
                    continue
                
                # Create retrieved document
                retrieved = RetrievedDocument(
                    doc_id=doc_metadata.get('data_id', f'doc_{idx}'),
                    text=doc_metadata.get('question', ''),
                    score=float(score),
                    metadata=doc_metadata,
                    source=doc_metadata.get('source', 'unknown')
                )
                
                results.append(retrieved)
                
                if len(results) >= k:
                    break
            
            all_results.append(results)
        
        return all_results[0] if single_query else all_results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        if not self.index:
            return {"status": "not_initialized"}
        
        stats = {
            "num_documents": len(self.documents),
            "embedding_dimension": self.dimension,
            "embedding_model": self.embedding_model_name,
            "index_type": type(self.index).__name__,
            "use_gpu": self.use_gpu
        }
        
        # Add subject distribution if available
        if self.documents:
            subjects = [doc.get('subject', 'Unknown') for doc in self.documents]
            from collections import Counter
            subject_dist = dict(Counter(subjects))
            stats["subject_distribution"] = subject_dist
        
        return stats


def format_evidence_for_prompt(
    retrieved_docs: List[RetrievedDocument],
    max_evidence_length: int = 2000,
    include_solutions: bool = True
) -> str:
    """
    Format retrieved documents as evidence for prompt injection.
    
    Args:
        retrieved_docs: List of retrieved documents
        max_evidence_length: Maximum length of evidence text
        include_solutions: Whether to include solutions/think content
        
    Returns:
        Formatted evidence string
    """
    if not retrieved_docs:
        return ""
    
    evidence_parts = []
    current_length = 0
    
    for i, doc in enumerate(retrieved_docs, 1):
        # Format document
        evidence = f"[P{i}] Problem: {doc.text}\n"
        
        # Add solution if available and requested
        if include_solutions and 'think' in doc.metadata:
            evidence += f"Solution approach: {doc.metadata['think'][:500]}...\n"
        
        # Add answer if available
        if 'answer' in doc.metadata:
            evidence += f"Answer: {doc.metadata['answer']}\n"
        
        evidence += f"(Similarity: {doc.score:.3f})\n"
        
        # Check length
        if current_length + len(evidence) > max_evidence_length:
            break
        
        evidence_parts.append(evidence)
        current_length += len(evidence)
    
    return "Evidence Pack:\n" + "\n".join(evidence_parts)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Vector Store for Data Generation")
    parser.add_argument("--build", action="store_true", help="Build index from dataset")
    parser.add_argument("--dataset", type=str, default="team-suzuki/SEED_000_origin_0813",
                        help="Dataset to index")
    parser.add_argument("--search", type=str, help="Search query")
    parser.add_argument("--k", type=int, default=5, help="Number of results")
    parser.add_argument("--index-path", type=str, default="vector_store/rag_index.faiss",
                        help="Path to index file")
    
    args = parser.parse_args()
    
    # Initialize vector store
    vector_store = RAGVectorStore(index_path=args.index_path)
    
    if args.build:
        # Build index from dataset
        vector_store.build_index_from_dataset(
            dataset_name=args.dataset,
            limit=1000  # Limit for testing
        )
        print("\nIndex statistics:")
        print(json.dumps(vector_store.get_statistics(), indent=2))
    
    elif args.search:
        # Load index and search
        if vector_store.load_index():
            results = vector_store.search(args.search, k=args.k)
            print(f"\nSearch results for: {args.search}")
            print("-" * 50)
            for i, doc in enumerate(results, 1):
                print(f"\n{i}. {doc.doc_id} (Score: {doc.score:.3f})")
                print(f"   Question: {doc.text[:200]}...")
                if 'subject' in doc.metadata:
                    print(f"   Subject: {doc.metadata['subject']}")
    
    else:
        print("Please specify --build or --search")