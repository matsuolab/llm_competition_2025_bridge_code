#!/usr/bin/env python3
"""
Build Knowledge Base from Hugging Face Dataset
Hugging Faceデータセットから知識ベースを構築
"""

import os
import sys
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import hashlib
from datetime import datetime
import pickle
import gc
import time
from datasets import load_dataset
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HuggingFaceKnowledgeBuilder:
    """Knowledge base builder from Hugging Face dataset"""
    
    def __init__(
        self,
        dataset_name: str = None,
        output_dir: str = None,
        embedding_model: str = None,
        device: str = None,
        batch_size: int = 32,
        max_chunk_size: int = 2000,
        hf_token: str = None
    ):
        """
        Args:
            dataset_name: Hugging Face dataset name (e.g., 'team-suzuki/RAG_0912')
            output_dir: Output directory for index files
            embedding_model: Embedding model name
            device: 'cuda' or 'cpu'
            batch_size: Batch size for embedding generation
            max_chunk_size: Maximum chunk size for document splitting
            hf_token: Hugging Face API token
        """
        # Load from environment variables if not provided
        self.dataset_name = dataset_name or os.getenv('HF_DATASET_NAME', 'team-suzuki/RAG_0912')
        self.output_dir = Path(output_dir or os.getenv('KNOWLEDGE_INDEX_OUTPUT_DIR', 
                                                       '/home/Competition2025/P05/shareP05/data_generation/knowledge_indexes'))
        self.embedding_model = embedding_model or os.getenv('EMBEDDING_MODEL', 'BAAI/bge-large-en-v1.5')
        self.device = device or os.getenv('DEVICE', 'cuda')
        self.batch_size = batch_size or int(os.getenv('BATCH_SIZE', '32'))
        self.hf_token = hf_token or os.getenv('HF_TOKEN')
        
        self.max_chunk_size = max_chunk_size
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._set_permissions(self.output_dir)
        
        # Initialize encoder
        logger.info(f"🤗 Loading dataset: {self.dataset_name}")
        logger.info(f"🔧 Initializing embedding model: {self.embedding_model}")
        self.encoder = SentenceTransformer(self.embedding_model, device=self.device)
        self.dimension = self.encoder.get_sentence_embedding_dimension()
        
        # Storage
        self.documents = []
        self.embeddings = []
        
        # Statistics
        self.stats = {
            "total_papers": 0,
            "total_chunks": 0,
            "total_documents": 0,
            "categories": {},
            "license_distribution": {},
            "quality_distribution": {},
            "processing_time": 0
        }
    
    def _set_permissions(self, path: Path):
        """Set appropriate permissions for output files"""
        try:
            os.chmod(path, 0o775)
            for child in path.iterdir():
                os.chmod(child, 0o664)
        except Exception as e:
            logger.warning(f"Could not set permissions: {e}")
    
    def load_from_huggingface(self) -> List[Dict]:
        """Load dataset from Hugging Face"""
        try:
            logger.info(f"📥 Loading dataset from Hugging Face: {self.dataset_name}")
            
            # Load dataset with authentication
            dataset = load_dataset(
                self.dataset_name,
                token=self.hf_token,
                split='train'  # Assuming data is in 'train' split
            )
            
            logger.info(f"✅ Loaded {len(dataset)} entries from Hugging Face")
            
            # Convert to list of dictionaries
            papers = []
            for item in dataset:
                papers.append(item)
            
            return papers
            
        except Exception as e:
            logger.error(f"❌ Failed to load dataset from Hugging Face: {e}")
            raise
    
    def process_papers(self, papers: List[Dict]) -> Tuple[List[Dict], List[np.ndarray]]:
        """Process papers and generate embeddings"""
        documents = []
        texts_to_encode = []
        
        for idx, paper in enumerate(papers):
            # Create document metadata with doc_id
            arxiv_id = paper.get("arxiv_id", f"unknown_{idx}")
            doc_id = f"abstract_{arxiv_id}_{idx}"
            
            # Prepare text for embedding
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            
            doc = {
                "doc_id": doc_id,
                "arxiv_id": arxiv_id,
                "title": title,
                "abstract": abstract,
                "content": f"{title}\n\n{abstract}",  # Combined content for search
                "authors": paper.get("authors", []),
                "categories": paper.get("categories", []),
                "license": paper.get("license", ""),
                "quality_score": paper.get("quality_score", 0),
                "difficulty_level": paper.get("difficulty_level", ""),
                "published": paper.get("published", ""),
                "url": paper.get("url", "")
            }
            
            # Prepare text for embedding
            text = f"{doc['title']}\n\n{doc['abstract']}"
            
            # Add PDF content if available
            if "pdf_content" in paper and paper["pdf_content"]:
                text += f"\n\n{paper['pdf_content'][:self.max_chunk_size]}"
            
            documents.append(doc)
            texts_to_encode.append(text)
            
            # Update statistics
            self.stats["total_papers"] += 1
            for cat in doc["categories"]:
                self.stats["categories"][cat] = self.stats["categories"].get(cat, 0) + 1
            
            license_type = doc["license"]
            self.stats["license_distribution"][license_type] = \
                self.stats["license_distribution"].get(license_type, 0) + 1
        
        # Generate embeddings in batches
        logger.info(f"🔄 Generating embeddings for {len(texts_to_encode)} documents...")
        embeddings = []
        
        for i in range(0, len(texts_to_encode), self.batch_size):
            batch = texts_to_encode[i:i + self.batch_size]
            batch_embeddings = self.encoder.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            embeddings.extend(batch_embeddings)
            
            if (i + self.batch_size) % (self.batch_size * 10) == 0:
                logger.info(f"  Processed {i + self.batch_size}/{len(texts_to_encode)} documents")
        
        self.stats["total_documents"] = len(documents)
        
        return documents, np.array(embeddings)
    
    def build_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build FAISS index"""
        logger.info(f"🏗️ Building FAISS index with {len(embeddings)} vectors...")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create index
        index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        
        # Add vectors
        index.add(embeddings)
        
        logger.info(f"✅ Index built with {index.ntotal} vectors")
        
        return index
    
    def save_index(self, index: faiss.Index, documents: List[Dict]):
        """Save index and metadata"""
        # Save FAISS index
        index_path = self.output_dir / "knowledge_index.faiss"
        faiss.write_index(index, str(index_path))
        logger.info(f"💾 Saved FAISS index to {index_path}")
        
        # Save metadata
        metadata = {
            "documents": documents,
            "stats": self.stats,
            "config": {
                "dataset_name": self.dataset_name,
                "embedding_model": self.embedding_model,
                "dimension": self.dimension,
                "total_documents": len(documents),
                "created_at": datetime.now().isoformat()
            }
        }
        
        metadata_path = self.output_dir / "knowledge_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 Saved metadata to {metadata_path}")
        
        # Set permissions
        self._set_permissions(self.output_dir)
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print processing summary"""
        print("\n" + "=" * 60)
        print("📊 Knowledge Index Build Summary")
        print("=" * 60)
        print(f"Dataset: {self.dataset_name}")
        print(f"Total papers: {self.stats['total_papers']}")
        print(f"Total documents: {self.stats['total_documents']}")
        print(f"Output directory: {self.output_dir}")
        print(f"\nCategory distribution:")
        for cat, count in sorted(self.stats['categories'].items(), 
                                 key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {cat}: {count}")
        print(f"\nLicense distribution:")
        for license_type, count in sorted(self.stats['license_distribution'].items(),
                                         key=lambda x: x[1], reverse=True):
            print(f"  {license_type}: {count}")
        print("=" * 60)
    
    def run(self):
        """Main execution method"""
        start_time = time.time()
        
        try:
            # Load dataset from Hugging Face
            papers = self.load_from_huggingface()
            
            if not papers:
                logger.warning("No papers found in dataset")
                return
            
            # Process papers and generate embeddings
            documents, embeddings = self.process_papers(papers)
            
            # Build FAISS index
            index = self.build_index(embeddings)
            
            # Save index and metadata
            self.save_index(index, documents)
            
            self.stats["processing_time"] = time.time() - start_time
            logger.info(f"✅ Processing completed in {self.stats['processing_time']:.2f} seconds")
            
        except Exception as e:
            logger.error(f"❌ Error during processing: {e}")
            raise


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Build knowledge index from Hugging Face dataset")
    parser.add_argument("--dataset-name", type=str, 
                       help="Hugging Face dataset name")
    parser.add_argument("--output-dir", type=str,
                       help="Output directory for index files")
    parser.add_argument("--embedding-model", type=str,
                       help="Embedding model to use")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"],
                       help="Device to use for embeddings")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for embedding generation")
    parser.add_argument("--hf-token", type=str,
                       help="Hugging Face API token")
    
    args = parser.parse_args()
    
    # Build knowledge index
    builder = HuggingFaceKnowledgeBuilder(
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        embedding_model=args.embedding_model,
        device=args.device,
        batch_size=args.batch_size,
        hf_token=args.hf_token
    )
    
    builder.run()


if __name__ == "__main__":
    main()