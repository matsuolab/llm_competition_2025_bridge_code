#!/usr/bin/env python3
"""
Build FAISS knowledge index from team-suzuki/RAG_0915 dataset
Properly extracts content and metadata from ArXiv papers
"""

import json
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import torch
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KnowledgeIndexBuilder:
    """Build FAISS index from knowledge dataset"""

    def __init__(self, config_path: str = None):
        """Initialize the index builder"""
        # Load environment variables
        if config_path:
            load_dotenv(config_path)
        else:
            load_dotenv(Path(__file__).parent.parent / "config" / ".env")

        # Configuration
        self.dataset_name = os.getenv("HF_DATASET_NAME", "team-suzuki/RAG_0915")
        self.output_dir = os.getenv("KNOWLEDGE_INDEX_OUTPUT_DIR",
                                    "/home/Competition2025/P05/shareP05/data_generation/knowledge_indexes")
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
        self.device = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = int(os.getenv("BATCH_SIZE", "32"))
        self.hf_token = os.getenv("HF_TOKEN")

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize embedding model
        logger.info(f"Loading embedding model: {self.embedding_model_name} on {self.device}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name, device=self.device)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

    def load_dataset_from_hf(self) -> List[Dict[str, Any]]:
        """Load dataset from HuggingFace"""
        logger.info(f"Loading dataset: {self.dataset_name}")

        try:
            dataset = load_dataset(self.dataset_name, token=self.hf_token)

            # Get the train split (or first available split)
            if 'train' in dataset:
                data = dataset['train']
            else:
                # Get first available split
                split_name = list(dataset.keys())[0]
                data = dataset[split_name]

            logger.info(f"Loaded {len(data)} documents from {self.dataset_name}")
            return data

        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    def process_documents(self, dataset) -> tuple:
        """Process documents and extract relevant fields"""
        documents = []
        metadata_list = []

        logger.info("Processing documents...")

        for idx, item in enumerate(tqdm(dataset, desc="Processing documents")):
            # Extract content - the actual paper content
            content = item.get('content', '')

            # Skip empty documents
            if not content or len(content.strip()) < 10:
                continue

            # Create searchable text combining multiple fields
            searchable_text_parts = []

            # Add title if available
            title = item.get('paper_title', '').strip()
            if title:
                searchable_text_parts.append(f"Title: {title}")

            # Add section information
            section_title = item.get('section_title', '').strip()
            section_type = item.get('section_type', '').strip()
            if section_title:
                searchable_text_parts.append(f"Section: {section_title}")
            if section_type:
                searchable_text_parts.append(f"Type: {section_type}")

            # Add categories (subjects)
            categories = item.get('categories', '').strip()
            if categories:
                searchable_text_parts.append(f"Categories: {categories}")

            # Add main content
            searchable_text_parts.append(content)

            # Combine all parts
            searchable_text = "\n".join(searchable_text_parts)

            # Store document for embedding
            documents.append(searchable_text)

            # Create metadata
            metadata = {
                'doc_id': item.get('chunk_id', f"doc_{idx}"),
                'arxiv_id': item.get('arxiv_id', ''),
                'title': title,
                'authors': item.get('authors', ''),
                'categories': categories,  # This contains subject areas like "math.OC"
                'section_title': section_title,
                'section_type': section_type,
                'section_index': item.get('section_index', 0),
                'chunk_index': item.get('chunk_index', 0),
                'content': content,  # Store actual content
                'content_length': len(content),
                'published_date': str(item.get('published_date', '')),
                'license': item.get('license_type', ''),
                'quality_score': item.get('paper_quality_score', 0.0),
                'is_chunked': item.get('is_chunked', False)
            }

            # Extract subject from categories (e.g., "math.OC" -> "Mathematics")
            subject = self.extract_subject_from_categories(categories)
            metadata['subject'] = subject

            metadata_list.append(metadata)

        logger.info(f"Processed {len(documents)} non-empty documents")
        return documents, metadata_list

    def extract_subject_from_categories(self, categories: str) -> str:
        """Extract subject from ArXiv categories"""
        if not categories:
            return "Unknown"

        # Map ArXiv categories to subjects
        category_map = {
            'math': 'Mathematics',
            'cs': 'Computer Science',
            'physics': 'Physics',
            'q-bio': 'Quantitative Biology',
            'q-fin': 'Quantitative Finance',
            'stat': 'Statistics',
            'eess': 'Electrical Engineering and Systems Science',
            'econ': 'Economics',
            'astro-ph': 'Astrophysics',
            'cond-mat': 'Condensed Matter',
            'gr-qc': 'General Relativity and Quantum Cosmology',
            'hep': 'High Energy Physics',
            'nlin': 'Nonlinear Sciences',
            'nucl': 'Nuclear'
        }

        # Parse categories (e.g., "cs.LG, math.OC, stat.ML")
        cats = categories.lower().split(',')
        subjects = set()

        for cat in cats:
            cat = cat.strip()
            for prefix, subject in category_map.items():
                if cat.startswith(prefix):
                    subjects.add(subject)
                    break

        if subjects:
            return ', '.join(sorted(subjects))
        return "Unknown"

    def create_embeddings(self, documents: List[str]) -> np.ndarray:
        """Create embeddings for documents"""
        logger.info(f"Creating embeddings for {len(documents)} documents...")

        embeddings = []
        for i in tqdm(range(0, len(documents), self.batch_size), desc="Creating embeddings"):
            batch = documents[i:i + self.batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            embeddings.append(batch_embeddings)

        embeddings = np.vstack(embeddings).astype('float32')
        logger.info(f"Created embeddings with shape: {embeddings.shape}")
        return embeddings

    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build FAISS index from embeddings"""
        logger.info("Building FAISS index...")

        # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
        index = faiss.IndexFlatIP(self.embedding_dim)

        # Add vectors to index
        index.add(embeddings)

        logger.info(f"FAISS index built with {index.ntotal} vectors")
        return index

    def save_index_and_metadata(self, index: faiss.Index, metadata_list: List[Dict]):
        """Save FAISS index and metadata"""
        # Save FAISS index
        index_path = os.path.join(self.output_dir, "knowledge_index.faiss")
        faiss.write_index(index, index_path)
        logger.info(f"Saved FAISS index to {index_path}")

        # Save metadata with statistics
        metadata_path = os.path.join(self.output_dir, "knowledge_metadata.json")

        # Calculate statistics
        categories_count = {}
        subjects_count = {}
        for doc in metadata_list:
            # Count categories
            cats = doc.get('categories', '').split(',')
            for cat in cats:
                cat = cat.strip()
                if cat:
                    categories_count[cat] = categories_count.get(cat, 0) + 1

            # Count subjects
            subject = doc.get('subject', 'Unknown')
            subjects_count[subject] = subjects_count.get(subject, 0) + 1

        metadata_output = {
            'documents': metadata_list,
            'stats': {
                'total_documents': len(metadata_list),
                'categories_distribution': categories_count,
                'subjects_distribution': subjects_count,
                'math_documents': sum(1 for doc in metadata_list if 'math' in doc.get('categories', '').lower())
            },
            'config': {
                'dataset_name': self.dataset_name,
                'embedding_model': self.embedding_model_name,
                'dimension': self.embedding_dim,
                'total_documents': len(metadata_list)
            }
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata_output, f)

        logger.info(f"Saved metadata to {metadata_path}")
        logger.info(f"Total documents: {len(metadata_list)}")
        logger.info(f"Math-related documents: {metadata_output['stats']['math_documents']}")
        logger.info(f"Subject distribution: {subjects_count}")

    def build(self):
        """Main build process"""
        try:
            # Load dataset
            dataset = self.load_dataset_from_hf()

            # Process documents
            documents, metadata_list = self.process_documents(dataset)

            if not documents:
                logger.error("No valid documents found to index!")
                return

            # Create embeddings
            embeddings = self.create_embeddings(documents)

            # Build FAISS index
            index = self.build_faiss_index(embeddings)

            # Save index and metadata
            self.save_index_and_metadata(index, metadata_list)

            logger.info("Knowledge index building completed successfully!")

        except Exception as e:
            logger.error(f"Failed to build index: {e}")
            raise


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Build FAISS knowledge index")
    parser.add_argument("--config", type=str, help="Path to .env config file")
    args = parser.parse_args()

    # Build index
    builder = KnowledgeIndexBuilder(config_path=args.config)
    builder.build()


if __name__ == "__main__":
    main()