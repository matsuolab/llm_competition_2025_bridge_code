import json
import pickle
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# import torch
import umap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalModelEmbedder:
    """Local model embedder using HuggingFace models"""

    def __init__(self, model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"):
        """Initialize the local model embedder.

        Args:
            model_id: HuggingFace model ID to use for embeddings
        """
        self.model_id = model_id
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the model and tokenizer"""
        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                load_in_4bit=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)

    def get_embedding(self, text: Union[str, List[str]], max_length: int = 2048) -> np.ndarray:
        """Get embeddings for text using the local model.

        Args:
            text: Input text or list of texts
            max_length: Maximum sequence length

        Returns:
            numpy.ndarray: Embedding vector(s)
        """
        if self.model is None:
            self.load_model()

        # Handle single text input
        if isinstance(text, str):
            text = [text]

        # Tokenize
        batch = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.model.device)

        # Get hidden states
        out = self.model(**batch, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states[-1]

        # Apply attention mask and average
        mask = batch["attention_mask"].unsqueeze(-1)
        hs = hs * mask
        summed = hs.sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        emb = summed / lengths

        # Apply layer normalization if available
        if hasattr(self.model, "model") and hasattr(self.model.model, "final_layernorm"):
            emb = self.model.model.final_layernorm(emb)
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "ln_f"):
            emb = self.model.transformer.ln_f(emb)

        # Normalize embeddings
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)

        # Convert to numpy array
        return emb.float().cpu().numpy()


class SFTEmbeddingAnalyzer:
    """Advanced embedding analysis for academic datasets"""

    def __init__(self, api_key=None, use_local_model=True, model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"):
        """Initialize the analyzer.

        Args:
            api_key: API key (optional)
            use_local_model: Whether to use local model for embeddings
            model_id: HuggingFace model ID for local embeddings
        """
        self.use_local_model = use_local_model
        if use_local_model:
            self.embedder = LocalModelEmbedder(model_id=model_id)
        else:
            import os

            from openai import OpenAI

            # TODO: Change to your favorite API key
            API_KEY = "********"
            self.client = OpenAI(api_key=API_KEY)
        self.embeddings = {}
        self.embedding_matrix = None
        self.df = None

    def get_embedding(
        self, text: Union[str, List[str]], model: str = "text-embedding-3-small", max_length: int = 8000
    ) -> Optional[np.ndarray]:
        """Get embedding for text(s).

        Args:
            text: Input text or list of texts
            model: Model name (for API) or ignored for local model
            max_length: Maximum sequence length

        Returns:
            numpy.ndarray: Embedding vector(s) or None if error
        """
        if self.use_local_model:
            try:
                embeddings = self.embedder.get_embedding(text, max_length=min(max_length, 2048))
                # If input was a single string, embeddings will be a 2D array with shape (1, dim)
                # If input was a list, embeddings will be a 2D array with shape (len(text), dim)
                # In both cases, we can return embeddings directly
                return embeddings
            except Exception as e:
                print(f"Error getting local embedding: {e}")
                return None
        else:
            # Truncate text if too long
            if isinstance(text, str):
                text = text[:max_length]
            else:
                text = [t[:max_length] for t in text]

            try:
                response = self.client.embeddings.create(input=text, model=model)
                if isinstance(text, str):
                    embedding = response.data[0].embedding
                    return embedding
                embeddings = [d.embedding for d in response.data]
                return embeddings
            except Exception as e:
                print(f"Error getting API embedding: {e}")
                return None

    def compute_hle_embeddings(
        self,
        df,
        text_columns=['question'],
        text_column='question',  # Deprecated, kept for backward compatibility
        batch_size=50,
        save_path=None,
    ):
        """Compute embeddings for HLE dataset with configurable text combination

        Args:
            df: DataFrame containing the data
            text_columns: List of column names or dict mapping column names to labels.
                         If None, defaults to ['question', 'rationale']
                         Examples:
                         - ['question', 'rationale', 'answer']
                         - {'question': 'Q', 'rationale': 'Explanation', 'category': 'Subject'}
            text_column: Deprecated, use text_columns instead. Kept for backward compatibility.
            batch_size: Number of samples to process in each batch
            save_path: Path to save embeddings pickle file

        Returns:
            numpy array of embeddings
        """
        self.df = df.copy()
        embeddings_list = []

        # Handle backward compatibility and default configuration
        if text_columns is None:
            # Smart defaults based on available columns
            available_columns = df.columns.tolist()
            default_columns = []

            # Always include the primary text column
            if text_column in available_columns:
                default_columns.append(text_column)
            elif 'question' in available_columns:
                default_columns.append('question')

            # Add other important columns if available
            priority_columns = ['rationale', 'answer', 'category', 'raw_subject']
            for col in priority_columns:
                if col in available_columns and col not in default_columns:
                    default_columns.append(col)

            text_columns = default_columns[:3]  # Limit to top 3 to avoid too long text
            print(f"Auto-detected text columns: {text_columns}")

        # Handle different input formats for text_columns
        if isinstance(text_columns, dict):
            column_labels = text_columns
            columns_to_use = list(text_columns.keys())
        else:
            # Create default labels for list input
            columns_to_use = text_columns
            column_labels = {}
            for col in columns_to_use:
                if col == 'question':
                    column_labels[col] = 'Question'
                elif col == 'rationale':
                    column_labels[col] = 'Rationale'
                elif col == 'answer':
                    column_labels[col] = 'Answer'
                elif col == 'category':
                    column_labels[col] = 'Category'
                elif col == 'raw_subject':
                    column_labels[col] = 'Subject'
                else:
                    column_labels[col] = col.replace('_', ' ').title()

        print(f"Computing embeddings for {len(df)} samples using columns: {columns_to_use}")

        for i in tqdm(range(0, len(df), batch_size)):
            batch = df.iloc[i : i + batch_size]
            batch_embeddings = []

            for _, row in batch.iterrows():
                # Combine text from specified columns
                text_parts = []

                for col in columns_to_use:
                    if col in df.columns and pd.notna(row.get(col, '')):
                        value = str(row[col]).strip()
                        if value:  # Only add non-empty values
                            label = column_labels[col]
                            text_parts.append(f"{label}: {value}")

                # Fallback if no text parts found
                if not text_parts:
                    if 'question' in df.columns:
                        text_parts.append(f"Question: {row.get('question', 'No content')}")
                    else:
                        text_parts.append("No content available")

                combined_text = "\n".join(text_parts)

                embedding = self.get_embedding(combined_text)
                if embedding is not None:
                    # Handle both API (1D) and local model (2D) embeddings
                    if isinstance(embedding, list):
                        # API returns list of embeddings
                        batch_embeddings.append(embedding)
                    elif hasattr(embedding, 'ndim') and embedding.ndim == 2:
                        # Local model returns 2D array
                        batch_embeddings.append(embedding[0])
                    else:
                        # API returns 1D array for single text
                        batch_embeddings.append(embedding)
                else:
                    # Fallback to zeros if embedding fails
                    # Use appropriate embedding size based on model
                    embedding_size = 1536 if not self.use_local_model else 3584
                    batch_embeddings.append([0.0] * embedding_size)

            embeddings_list.extend(batch_embeddings)

        self.embedding_matrix = np.array(embeddings_list)

        # Save embeddings if path provided
        if save_path:
            # Include text configuration in metadata for reproducibility
            metadata = df.to_dict('records')
            embedding_config = {
                'text_columns': columns_to_use,
                'column_labels': column_labels,
                'batch_size': batch_size,
            }

            with open(save_path, 'wb') as f:
                pickle.dump({'embeddings': self.embedding_matrix, 'metadata': metadata, 'config': embedding_config}, f)
            print(f"Embeddings saved to {save_path}")

        return self.embedding_matrix

    def load_embeddings(self, load_path):
        """Load pre-computed embeddings"""
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
            self.embedding_matrix = data['embeddings']
            self.df = pd.DataFrame(data['metadata'])

        print(f"Loaded embeddings for {len(self.df)} samples")
        return self.embedding_matrix

    def find_similar_questions(self, query_idx, top_k=5):
        """Find most similar questions to a given question"""
        if self.embedding_matrix is None:
            raise ValueError("No embeddings computed. Run compute_hle_embeddings first.")

        query_embedding = self.embedding_matrix[query_idx].reshape(1, -1)
        similarities = cosine_similarity(query_embedding, self.embedding_matrix)[0]

        # Get top k similar questions (excluding the query itself)
        similar_indices = np.argsort(similarities)[::-1][1 : top_k + 1]

        results = []
        for idx in similar_indices:
            results.append(
                {
                    'index': idx,
                    'similarity': similarities[idx],
                    'question': self.df.iloc[idx]['question'],
                    'category': self.df.iloc[idx]['category'],
                    'subject': self.df.iloc[idx]['raw_subject'],
                }
            )

        return results

    def cluster_questions(self, n_clusters=8, random_state=42):
        """Cluster questions based on embeddings"""
        if self.embedding_matrix is None:
            raise ValueError("No embeddings computed. Run compute_hle_embeddings first.")

        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        cluster_labels = kmeans.fit_predict(self.embedding_matrix)

        self.df['cluster'] = cluster_labels

        # Analyze cluster characteristics
        cluster_analysis = {}
        for cluster_id in range(n_clusters):
            cluster_data = self.df[self.df['cluster'] == cluster_id]
            cluster_analysis[str(cluster_id)] = {
                'size': int(len(cluster_data)),
                'top_categories': {
                    k: int(v) for k, v in cluster_data['category'].value_counts().head(3).to_dict().items()
                },
                'top_subjects': {
                    k: int(v) for k, v in cluster_data['raw_subject'].value_counts().head(3).to_dict().items()
                },
                'avg_question_length': float(cluster_data['question_length'].mean()),
                'sample_questions': cluster_data['question'].head(3).tolist(),
            }

        return cluster_labels, cluster_analysis

    def visualize_embeddings_2d(self, method='umap', color_by='category', save_path=None):
        """Visualize embeddings in 2D space"""
        if self.embedding_matrix is None:
            raise ValueError("No embeddings computed. Run compute_hle_embeddings first.")

        print(f"Computing 2D projection using {method}...")

        if method == 'umap':
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
            embedding_2d = reducer.fit_transform(self.embedding_matrix)
        elif method == 'tsne':
            # Adaptive perplexity based on sample size
            perplexity = min(30, max(5, len(self.embedding_matrix) // 4))
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            embedding_2d = tsne.fit_transform(self.embedding_matrix)
        elif method == 'pca':
            pca = PCA(n_components=2, random_state=42)
            embedding_2d = pca.fit_transform(self.embedding_matrix)
        else:
            raise ValueError("Method must be 'umap', 'tsne', or 'pca'")

        # Create visualization
        plt.figure(figsize=(12, 8))

        if color_by in self.df.columns:
            unique_values = self.df[color_by].unique()

            # Use a combination of distinct color palettes for better separation
            if len(unique_values) <= 10:
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_values)))
            elif len(unique_values) <= 20:
                colors = plt.cm.tab20(np.linspace(0, 1, len(unique_values)))
            else:
                # For more than 20 categories, use a combination of palettes
                base_colors = plt.cm.Set1(np.linspace(0, 1, min(9, len(unique_values))))
                additional_colors = plt.cm.Set2(np.linspace(0, 1, max(0, len(unique_values) - 9)))
                colors = np.vstack([base_colors, additional_colors]) if len(unique_values) > 9 else base_colors

            for i, value in enumerate(unique_values):
                mask = self.df[color_by] == value

                # Check if data is from seed dataset (detect by presence of 'dataset_type' column or other indicators)
                is_seed_data = (
                    'dataset' in self.df.columns and (self.df[mask]['dataset'] == 'SEED').any()
                    if len(self.df[mask]) > 0
                    else False
                )

                # Set marker style - keep distinct markers for dataset differentiation
                marker = '^' if is_seed_data else 'o'  # Triangle for seed, circle for HLE
                # Remove color intensity modification to maintain distinct colors
                alpha_value = 0.5 if is_seed_data else 0.8  # Slightly different transparency

                plt.scatter(
                    embedding_2d[mask, 0],
                    embedding_2d[mask, 1],
                    c=[colors[i]],
                    label=value,
                    alpha=alpha_value,
                    s=50,
                    marker=marker,
                    edgecolors='black' if is_seed_data else 'none',  # Add edge for seed data
                    linewidth=0.5 if is_seed_data else 0,
                )

            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # Default case when not coloring by a specific column
            # Check if we can detect dataset type
            is_seed_data = 'dataset_type' in self.df.columns and (self.df['dataset_type'] == 'seed').any()

            if is_seed_data and 'dataset_type' in self.df.columns:
                # Plot seed and HLE data separately
                seed_mask = self.df['dataset_type'] == 'seed'
                hle_mask = self.df['dataset_type'] != 'seed'

                if seed_mask.any():
                    plt.scatter(
                        embedding_2d[seed_mask, 0],
                        embedding_2d[seed_mask, 1],
                        alpha=0.6,
                        s=50,
                        marker='^',
                        label='Seed',
                        color='lightblue',
                    )
                if hle_mask.any():
                    plt.scatter(
                        embedding_2d[hle_mask, 0],
                        embedding_2d[hle_mask, 1],
                        alpha=0.8,
                        s=50,
                        marker='o',
                        label='HLE',
                        color='darkblue',
                    )
                plt.legend()
            else:
                plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.6, s=50)

        plt.title(f'HLE Questions Embedding Visualization ({method.upper()})\nColored by {color_by}')
        plt.xlabel(f'{method.upper()} 1')
        plt.ylabel(f'{method.upper()} 2')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        # plt.tight_layout()
        # plt.show()

        return embedding_2d

    def analyze_category_similarity(self, save_path=None):
        """Analyze similarity between different categories"""
        if self.embedding_matrix is None:
            raise ValueError("No embeddings computed. Run compute_hle_embeddings first.")

        # Compute average embeddings for each category
        categories = self.df['category'].unique()
        category_embeddings = {}

        for category in categories:
            category_mask = self.df['category'] == category
            category_embeddings[category] = np.mean(self.embedding_matrix[category_mask], axis=0)

        # Compute similarity matrix
        category_names = list(category_embeddings.keys())
        similarity_matrix = np.zeros((len(category_names), len(category_names)))

        for i, cat1 in enumerate(category_names):
            for j, cat2 in enumerate(category_names):
                similarity = cosine_similarity(
                    category_embeddings[cat1].reshape(1, -1), category_embeddings[cat2].reshape(1, -1)
                )[0, 0]
                similarity_matrix[i, j] = similarity

        # Visualize similarity matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            similarity_matrix,
            xticklabels=category_names,
            yticklabels=category_names,
            annot=True,
            fmt='.3f',
            cmap='viridis',
            center=0.5,
        )
        plt.title('Category Similarity Matrix (Cosine Similarity)')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.show()

        return similarity_matrix, category_names

    def find_outlier_questions(self, threshold_percentile=95):
        """Find questions that are outliers (very different from others)"""
        if self.embedding_matrix is None:
            raise ValueError("No embeddings computed. Run compute_hle_embeddings first.")

        # Compute average distance to all other questions
        avg_distances = []

        print("Computing outlier scores...")
        for i in tqdm(range(len(self.embedding_matrix))):
            query_embedding = self.embedding_matrix[i].reshape(1, -1)
            distances = 1 - cosine_similarity(query_embedding, self.embedding_matrix)[0]
            avg_distance = np.mean(distances[distances > 0])  # Exclude self-similarity
            avg_distances.append(avg_distance)

        avg_distances = np.array(avg_distances)
        threshold = np.percentile(avg_distances, threshold_percentile)

        outlier_indices = np.where(avg_distances >= threshold)[0]

        outliers = []
        for idx in outlier_indices:
            outliers.append(
                {
                    'index': int(idx),
                    'outlier_score': float(avg_distances[idx]),
                    'question': str(self.df.iloc[idx]['question']),
                    'category': str(self.df.iloc[idx]['category']),
                    'subject': str(self.df.iloc[idx]['raw_subject']),
                    'question_length': int(self.df.iloc[idx]['question_length']),
                }
            )

        # Sort by outlier score
        outliers.sort(key=lambda x: x['outlier_score'], reverse=True)

        return outliers

    def semantic_search(self, query_text, top_k=10):
        """Perform semantic search to find relevant questions"""
        if self.embedding_matrix is None:
            raise ValueError("No embeddings computed. Run compute_hle_embeddings first.")

        # Get embedding for query
        query_embedding = self.get_embedding(query_text)
        if query_embedding is None:
            raise ValueError("Could not compute embedding for query")

        # query_embedding is already a 2D array with shape (1, dim)

        # Compute similarities
        similarities = cosine_similarity(query_embedding, self.embedding_matrix)[0]

        # Get top k results
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append(
                {
                    'index': int(idx),
                    'similarity': float(similarities[idx]),
                    'question': str(self.df.iloc[idx]['question']),
                    'answer': str(self.df.iloc[idx]['answer']),
                    'category': str(self.df.iloc[idx]['category']),
                    'subject': str(self.df.iloc[idx]['raw_subject']),
                }
            )

        return results


# Example usage functions
def run_comprehensive_embedding_analysis(
    df, save_embeddings=True, use_local_model=False, model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
):
    """Run complete embedding analysis pipeline

    Args:
        df: DataFrame containing the data
        save_embeddings: Whether to save embeddings to disk
        use_local_model: Whether to use local model for embeddings
        model_id: HuggingFace model ID for local embeddings
    """
    # Initialize analyzer with appropriate embedding method
    analyzer = SFTEmbeddingAnalyzer(use_local_model=use_local_model, model_id=model_id if use_local_model else None)

    # Compute embeddings
    embeddings = analyzer.compute_hle_embeddings(  # NOQA F841
        df, save_path='hle_embeddings.pkl' if save_embeddings else None
    )

    # Perform clustering
    cluster_labels, cluster_analysis = analyzer.cluster_questions(n_clusters=8)

    # Visualize embeddings
    analyzer.visualize_embeddings_2d(method='umap', color_by='category', save_path='hle_embeddings_umap.png')

    analyzer.visualize_embeddings_2d(method='tsne', color_by='cluster', save_path='hle_embeddings_tsne.png')

    # # Analyze category similarity
    # similarity_matrix, category_names = analyzer.analyze_category_similarity()

    # # Find outliers
    # outliers = analyzer.find_outlier_questions()

    # Save analysis results
    analysis_results = {
        'cluster_analysis': cluster_analysis,
        # 'outliers': outliers[:20],  # Top 20 outliers
        # 'category_similarity': {'matrix': similarity_matrix.tolist(), 'categories': category_names},
    }

    with open('hle_embedding_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=2)

    print("Comprehensive embedding analysis completed!")
    return analyzer, analysis_results


def example_semantic_search_queries(analyzer):
    """Example semantic search queries"""

    example_queries = [
        "quantum mechanics and wave functions",
        "historical battles and military strategy",
        "calculus and differential equations",
        "molecular biology and DNA",
        "economic theory and market dynamics",
    ]

    print("\n=== Semantic Search Examples ===")
    for query in example_queries:
        print(f"\nQuery: '{query}'")
        results = analyzer.semantic_search(query, top_k=3)

        for i, result in enumerate(results, 1):
            print(f"{i}. [{result['category']}] Similarity: {result['similarity']:.3f}")
            print(f"   Subject: {result['subject']}")
            print(f"   Question: {result['question'][:200]}...")
            print()


def compare_datasets_embeddings(
    seed_embeddings_path='seed_analysis_output/embeddings/seed_embeddings.pkl',
    hle_embeddings_path='hle_analysis_output/embeddings/hle_embeddings.pkl',
    output_dir='combined_analysis_output',
    n_clusters=8,
):
    """
    Load and compare embeddings from two datasets (SEED and HLE)

    Args:
        seed_embeddings_path: Path to SEED embeddings pickle file
        hle_embeddings_path: Path to HLE embeddings pickle file
        output_dir: Directory to save combined analysis results
        n_clusters: Number of clusters for analysis

    Returns:
        dict: Combined analysis results
    """

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # try:
    # Load SEED embeddings
    print("Loading SEED embeddings...")
    seed_analyzer = SFTEmbeddingAnalyzer()
    seed_embeddings = seed_analyzer.load_embeddings(seed_embeddings_path)
    seed_df = seed_analyzer.df.copy()

    # Load HLE embeddings
    print("Loading HLE embeddings...")
    hle_analyzer = SFTEmbeddingAnalyzer()
    hle_embeddings = hle_analyzer.load_embeddings(hle_embeddings_path)
    hle_df = hle_analyzer.df.copy()

    # Add dataset identifier to dataframes
    seed_df['dataset'] = 'SEED'
    hle_df['dataset'] = 'HLE'

    # Combine dataframes and embeddings
    print("Combining datasets...")
    combined_df = pd.concat([seed_df, hle_df], ignore_index=True)
    combined_embeddings = np.vstack([seed_embeddings, hle_embeddings])

    # Create analyzer with combined data
    analyzer = SFTEmbeddingAnalyzer()
    analyzer.df = combined_df
    analyzer.embedding_matrix = combined_embeddings

    print(f"Combined dataset: {len(combined_df)} samples")
    print(f"SEED samples: {len(seed_df)}")
    print(f"HLE samples: {len(hle_df)}")

    # Perform clustering analysis
    print("Performing clustering analysis...")
    cluster_labels, cluster_analysis = analyzer.cluster_questions(n_clusters=n_clusters)

    # Visualize embeddings colored by dataset
    print("Creating visualizations...")
    analyzer.visualize_embeddings_2d(
        method='umap', color_by='dataset', save_path=str(output_path / 'combined_embeddings_by_dataset_umap.png')
    )

    # Visualize embeddings colored by category
    analyzer.visualize_embeddings_2d(
        method='umap', color_by='category', save_path=str(output_path / 'combined_embeddings_by_category_umap.png')
    )

    # Analyze category similarity
    print("Analyzing category similarity...")
    similarity_matrix, category_names = analyzer.analyze_category_similarity(
        save_path=str(output_path / 'category_similarity_matrix.png')
    )

    # Find outlier questions
    print("Finding outlier questions...")
    outliers = analyzer.find_outlier_questions()

    # Analyze dataset differences
    print("Analyzing dataset differences...")
    dataset_analysis = {}
    for dataset in ['SEED', 'HLE']:
        dataset_data = combined_df[combined_df['dataset'] == dataset]
        dataset_analysis[dataset] = {
            'sample_count': len(dataset_data),
            'categories': dataset_data['category'].value_counts().to_dict(),
            'avg_question_length': (
                dataset_data['question_length'].mean() if 'question_length' in dataset_data.columns else None
            ),
            'cluster_distribution': dataset_data['cluster'].value_counts().to_dict(),
        }

    # Prepare combined analysis results
    combined_results = {
        'dataset_info': {
            'total_samples': len(combined_df),
            'seed_samples': len(seed_df),
            'hle_samples': len(hle_df),
            'combined_categories': combined_df['category'].unique().tolist(),
        },
        'cluster_analysis': cluster_analysis,
        'outliers': outliers[:20],  # Top 20 outliers
        'category_similarity': {'matrix': similarity_matrix.tolist(), 'categories': category_names},
        'dataset_analysis': dataset_analysis,
    }

    # Save analysis results
    results_file = output_path / 'combined_embedding_analysis.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(combined_results, f, ensure_ascii=False, indent=2, default=str)

    # Save combined dataframe with cluster assignments
    combined_df.to_csv(output_path / 'combined_dataset_with_clusters.csv', index=False)

    print("Combined analysis completed!")
    print(f"Results saved to: {output_path}")
    print(f"Analysis file: {results_file}")

    return combined_results

    # except FileNotFoundError as e:
    #     print(f"Error: Could not find embeddings file - {e}")
    #     print("Make sure to run embedding analysis on both datasets first")
    #     return None
    # except Exception as e:
    #     print(f"Error in combined analysis: {e}")
    #     return None


if __name__ == "__main__":
    compare_datasets_embeddings()
