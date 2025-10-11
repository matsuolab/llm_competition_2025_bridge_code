#!/usr/bin/env python3
"""
Fast Data Sampling Methods for DPO Dataset Analysis
=================================================

This script implements several lightweight sampling algorithms as alternatives to k-DPP
for selecting diverse, high-quality samples from DPO (Direct Preference Optimization) candidate datasets.

1. Output format matches DPO structure with messages and rejected_messages
2. Groups samples by data_id to ensure all related records are sampled together
3. Quality scoring considers the preferred response from messages

Available methods:
1. Diversity-Quality Greedy: Fast greedy selection balancing diversity and quality
2. Stratified Quality: Stratified sampling based on quality bins with diversity within bins
3. Top-K Diverse: Select top quality samples and then diversify using clustering
4. Random Quality: Simple quality-weighted random sampling

Author: AI Assistant
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Import token length utilities
from token_length_histogram_utils import calculate_token_lengths, load_tokenizer


class DPOFastSampler:
    """Fast sampler for selecting diverse, high-quality data points from DPO candidate datasets"""

    def __init__(
        self,
        k: int = 100,
        alpha: float = 1.0,
        sigma_percentile: float = 50.0,
        seed: int = 42,
        dry_run: bool = False,
        reference_category_filter: Optional[str] = None,
        reference_subject_filter: Optional[str] = None,
        max_token_length: Optional[int] = None,
        sampling_method: str = "diversity_quality_greedy",
        diversity_weight: float = 0.5,
        n_clusters: Optional[int] = None,
        dataset_ratios: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the DPO fast sampler

        Args:
            k: Number of unique data_ids to select (each data_id may have multiple records)
            alpha: Power of quality score (q_i^alpha)
            sigma_percentile: Percentile of distances to use for quality scoring sigma
            seed: Random seed for reproducibility
            dry_run: If True, use small parameters for testing
            reference_category_filter: Filter reference data by category (e.g., 'chemistry')
            reference_subject_filter: Filter reference data by subject (e.g., 'chemistry')
            max_token_length: Maximum token length for filtering candidates (if None, no filtering)
            sampling_method: Sampling method to use
            diversity_weight: Weight for diversity vs quality (0=only quality, 1=only diversity)
            n_clusters: Number of clusters for clustering-based methods (if None, uses k//4)
            dataset_ratios: Dictionary mapping dataset names to sampling ratios (e.g., {'dataset1': 0.4, 'dataset2': 0.6})
        """
        self.k = k if not dry_run else min(k, 10)
        self.alpha = alpha
        self.sigma_percentile = sigma_percentile
        self.seed = seed
        self.dry_run = dry_run
        self.reference_category_filter = reference_category_filter
        self.reference_subject_filter = reference_subject_filter
        self.max_token_length = max_token_length
        self.sampling_method = sampling_method
        self.diversity_weight = diversity_weight
        self.n_clusters = n_clusters or max(k // 4, 1)
        self.dataset_ratios = dataset_ratios

        # Initialize random state
        self.rng = np.random.RandomState(seed)

        # Storage for loaded data
        self.reference_embeddings: Optional[np.ndarray] = None
        self.reference_df: Optional[pd.DataFrame] = None
        self.candidate_embeddings: Optional[np.ndarray] = None
        self.candidate_df: Optional[pd.DataFrame] = None
        self.candidate_raw_df: Optional[pd.DataFrame] = None  # Store raw DPO data
        self.candidates_name: str = ""

        # DPO-specific storage
        self.data_id_to_indices: Dict[str, List[int]] = {}  # Map data_id to candidate indices
        self.unique_data_ids: List[str] = []  # List of unique data_ids

        # Results storage
        self.quality_scores: Optional[np.ndarray] = None
        self.distances: Optional[np.ndarray] = None
        self.selected_indices: Optional[List[int]] = None
        self.selected_data_ids: Optional[List[str]] = None
        self.leak_mask: Optional[np.ndarray] = None
        self.nearest_ref_indices: Optional[np.ndarray] = None

        # Timing information
        self.timings: Dict[str, float] = {}

    def _l2_normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize embeddings"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)  # Avoid division by zero
        return embeddings / norms

    def load_embeddings_data(
        self,
        reference_path: str = "supergpqa_analysis_output/embeddings/supergpqa_embeddings.pkl",
        candidate_paths: List[str] = None,
        output_dir: str = "analysis_outputs",
    ) -> None:
        """Load reference and candidate embeddings from pickle files"""
        if candidate_paths is None:
            candidate_paths = [
                "dpo_006_1_analysis_output/embeddings/dpo_006_1_embeddings.pkl",
                "dpo_arxivdata_widerange_analysis_output/embeddings/dpo_arxivdata_widerange_embeddings.pkl",
            ]

        print(f"Loading reference embeddings from {reference_path}...")
        ref_full_path = Path(output_dir) / reference_path
        with open(ref_full_path, 'rb') as f:
            ref_data = pickle.load(f)
            reference_embeddings = np.array(ref_data['embeddings'], dtype=np.float32)
            reference_df = pd.DataFrame(ref_data['metadata'])

            # Add dataset identifier
            dataset_name = Path(reference_path).stem.replace('_embeddings', '').upper()
            reference_df['dataset'] = dataset_name

        # Apply category and subject filters to reference data (same logic as original)
        filter_mask = np.ones(len(reference_df), dtype=bool)

        if self.reference_category_filter and self.reference_category_filter.lower() != 'all':
            print(f"Filtering reference data by category: {self.reference_category_filter}")
            category_mask = reference_df['category'].str.contains(self.reference_category_filter, case=False, na=False)
            filter_mask &= category_mask

        if self.reference_subject_filter and self.reference_subject_filter.lower() != 'all':
            print(f"Filtering reference data by subject: {self.reference_subject_filter}")
            subject_mask = np.zeros(len(reference_df), dtype=bool)
            for col in ['subject', 'raw_subject']:
                if col in reference_df.columns:
                    subject_mask |= reference_df[col].str.contains(self.reference_subject_filter, case=False, na=False)
            filter_mask &= subject_mask

        # Keep original reference data for later visualization
        self.reference_embeddings_full = self._l2_normalize(reference_embeddings)
        self.reference_df_full = reference_df.copy()

        # Apply filters
        self.reference_embeddings = reference_embeddings[filter_mask]
        self.reference_df = reference_df[filter_mask].reset_index(drop=True)

        if len(self.reference_embeddings) == 0:
            raise ValueError("No reference samples found after filtering.")

        # L2 normalize reference embeddings
        self.reference_embeddings = self._l2_normalize(self.reference_embeddings)

        print(f"Reference data: {len(self.reference_embeddings)} samples after filtering")

        # Load candidate embeddings
        print("Loading candidate embeddings...")
        candidate_embeddings = []
        candidate_dfs = []
        candidate_names = []

        for path in candidate_paths:
            cand_full_path = Path(output_dir) / path
            print(f"  Loading {cand_full_path}...")
            with open(cand_full_path, 'rb') as f:
                cand_data = pickle.load(f)
                embeddings = np.array(cand_data['embeddings'], dtype=np.float32)
                df = pd.DataFrame(cand_data['metadata'])

                # Add dataset identifier
                dataset_name = Path(path).stem.replace('_embeddings', '').upper()
                df['dataset'] = dataset_name

                candidate_embeddings.append(embeddings)
                candidate_dfs.append(df)
                candidate_names.append(dataset_name)

        # Combine all candidate data
        self.candidate_embeddings = np.vstack(candidate_embeddings)
        self.candidate_df = pd.concat(candidate_dfs, ignore_index=True)
        self.candidates_name = '_'.join(candidate_names)

        # L2 normalize candidate embeddings
        self.candidate_embeddings = self._l2_normalize(self.candidate_embeddings)

        # Build data_id mapping for DPO-specific handling
        self._build_data_id_mapping()

        print(f"Candidate data: {len(self.candidate_embeddings)} samples from {len(candidate_paths)} sources")
        print(f"Unique data_ids: {len(self.unique_data_ids)}")

    def load_raw_dpo_data(self, raw_data_paths: List[str]) -> None:
        """Load raw DPO data for output formatting"""
        print("Loading raw DPO data...")
        raw_dfs = []

        for path in raw_data_paths:
            try:
                if path.endswith('.parquet'):
                    df = pd.read_parquet(path)
                else:
                    # Assume it's a directory with parquet files
                    parquet_file = Path(path) / "data" / "train-00000-of-00001.parquet"
                    df = pd.read_parquet(parquet_file)

                raw_dfs.append(df)
                print(f"  Loaded {len(df)} raw DPO records from {path}")
            except Exception as e:
                print(f"  Warning: Could not load {path}: {e}")

        if raw_dfs:
            self.candidate_raw_df = pd.concat(raw_dfs, ignore_index=True)
            print(f"Total raw DPO records: {len(self.candidate_raw_df)}")
        else:
            print("Warning: No raw DPO data loaded. Output will use processed data only.")

    def validate_data_alignment(self) -> None:
        """Validate that candidate embeddings and raw data are properly aligned"""
        if self.candidate_raw_df is None:
            print("Warning: No raw data loaded, skipping alignment validation")
            return

        print("Validating data alignment between embeddings and raw data...")

        # Get data_ids from embedding data
        embedding_data_ids = set(self.candidate_df['data_id'].unique())

        # Get data_ids from raw data
        raw_data_ids = set(self.candidate_raw_df['data_id'].unique())

        # Check alignment
        only_in_embeddings = embedding_data_ids - raw_data_ids
        only_in_raw = raw_data_ids - embedding_data_ids
        common_data_ids = embedding_data_ids & raw_data_ids

        print(f"  Embedding data_ids: {len(embedding_data_ids)}")
        print(f"  Raw data_ids: {len(raw_data_ids)}")
        print(f"  Common data_ids: {len(common_data_ids)}")

        if only_in_embeddings:
            print(f"  WARNING: {len(only_in_embeddings)} data_ids only in embeddings:")
            for data_id in list(only_in_embeddings)[:3]:  # Show first 3 examples
                print(f"    {data_id}")
            if len(only_in_embeddings) > 3:
                print(f"    ... and {len(only_in_embeddings) - 3} more")

        if only_in_raw:
            print(f"  WARNING: {len(only_in_raw)} data_ids only in raw data:")
            for data_id in list(only_in_raw)[:3]:  # Show first 3 examples
                print(f"    {data_id}")
            if len(only_in_raw) > 3:
                print(f"    ... and {len(only_in_raw) - 3} more")

        # Check for excessive duplicates in combined data
        if only_in_raw:
            print("  ERROR: Misaligned data detected!")
            print("  This may cause duplicate records in output.")
            print("  Please ensure candidate_paths and raw_data_paths correspond to the same datasets.")

        # Validate record counts for common data_ids
        excessive_duplicates = []
        for data_id in list(common_data_ids)[:10]:  # Check first 10 common data_ids
            embedding_count = len(self.candidate_df[self.candidate_df['data_id'] == data_id])
            raw_count = len(self.candidate_raw_df[self.candidate_raw_df['data_id'] == data_id])

            if embedding_count != raw_count:
                excessive_duplicates.append((data_id, embedding_count, raw_count))

        if excessive_duplicates:
            print("  WARNING: Record count mismatches found:")
            for data_id, emb_count, raw_count in excessive_duplicates[:3]:
                print(f"    {data_id}: embedding={emb_count}, raw={raw_count}")

        if not only_in_embeddings and not only_in_raw and not excessive_duplicates:
            print("  ✓ Data alignment validation passed!")

    def _build_data_id_mapping(self) -> None:
        """Build mapping from data_id to candidate indices"""
        self.data_id_to_indices = {}

        for idx, row in self.candidate_df.iterrows():
            data_id = row.get('data_id', f'unknown_{idx}')
            if data_id not in self.data_id_to_indices:
                self.data_id_to_indices[data_id] = []
            self.data_id_to_indices[data_id].append(idx)

        self.unique_data_ids = list(self.data_id_to_indices.keys())
        print(f"Built data_id mapping: {len(self.unique_data_ids)} unique data_ids")

    def filter_by_token_length(self) -> None:
        """Filter candidates by token length if specified"""
        if self.max_token_length is None:
            print("No token length filtering specified")
            return

        print(f"Filtering candidates by max token length: {self.max_token_length}")

        # Check for cached token lengths
        cache_dir = Path("analysis_outputs/token_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Create cache key based on dataset characteristics
        dataset_hash = hash(str(sorted(self.candidate_df[['question', 'rationale']].values.tolist())))
        cache_file = cache_dir / f"token_lengths_dpo_{self.candidates_name}_{dataset_hash}.pkl"

        if cache_file.exists():
            print(f"Loading cached token lengths from {cache_file}")
            start_time = time.time()
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                self.token_lengths = cached_data['token_lengths']
                print(f"Loaded {len(self.token_lengths)} cached token lengths")
            self.timings['token_filtering'] = time.time() - start_time
        else:
            # Calculate token lengths for all candidates
            start_time = time.time()
            tokenizer = load_tokenizer()

            # Prepare questions for tokenization by creating a DataFrame
            # For DPO, we calculate based on question and rationale (preferred response)
            questions_df = self.candidate_df[['question', 'rationale']].copy()
            # Add empty answer column for compatibility with calculate_token_lengths
            questions_df['answer'] = ''

            # Calculate token lengths
            self.token_lengths = calculate_token_lengths(questions_df, tokenizer)

            # Cache the results
            print(f"Caching token lengths to {cache_file}")
            cache_data = {
                'token_lengths': self.token_lengths,
                'candidates_name': self.candidates_name,
                'dataset_hash': dataset_hash,
                'n_samples': len(self.token_lengths),
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)

            self.timings['token_filtering'] = time.time() - start_time

        # Apply filter
        length_mask = self.token_lengths <= self.max_token_length
        n_filtered = np.sum(~length_mask)

        # Update candidate data
        self.candidate_embeddings = self.candidate_embeddings[length_mask]
        self.candidate_df = self.candidate_df[length_mask].reset_index(drop=True)
        self.token_lengths = self.token_lengths[length_mask]

        # Rebuild data_id mapping after filtering
        self._build_data_id_mapping()

        print(f"Filtered out {n_filtered} samples exceeding token length limit")
        print(f"Remaining candidates: {len(self.candidate_embeddings)}")
        print(f"Remaining unique data_ids: {len(self.unique_data_ids)}")

        self.timings['token_filtering'] = time.time() - start_time

    def compute_quality_scores(self) -> None:
        """Compute quality scores based on cosine distance to reference set"""
        start_time = time.time()
        print("Computing quality scores based on cosine similarity to reference set...")

        # Compute cosine similarities in batches to manage memory
        batch_size = 1000
        n_candidates = len(self.candidate_embeddings)
        min_distances = np.full(n_candidates, np.inf, dtype=np.float32)
        nearest_ref_indices = np.zeros(n_candidates, dtype=int)

        # Create progress bar for quality computation
        pbar = tqdm(range(0, n_candidates, batch_size), desc="Computing quality scores")

        for i in pbar:
            end_idx = min(i + batch_size, n_candidates)
            batch_embeddings = self.candidate_embeddings[i:end_idx]

            # Compute cosine similarities with all reference points
            similarities = cosine_similarity(batch_embeddings, self.reference_embeddings)
            distances = 1 - similarities  # Convert to distances

            # Find minimum distance and corresponding reference index for each candidate
            batch_min_distances = np.min(distances, axis=1)
            batch_nearest_indices = np.argmin(distances, axis=1)

            min_distances[i:end_idx] = batch_min_distances
            nearest_ref_indices[i:end_idx] = batch_nearest_indices

            # Update progress bar
            pbar.set_postfix({'processed': f"{end_idx}/{n_candidates}"})

        self.distances = min_distances
        self.nearest_ref_indices = nearest_ref_indices

        # Compute quality scores using Gaussian kernel
        sigma = np.percentile(self.distances, self.sigma_percentile)
        print(f"Using sigma = {sigma:.4f} (percentile {self.sigma_percentile})")

        quality_scores = np.exp(-self.distances**2 / (2 * sigma**2))
        if self.alpha != 1.0:
            quality_scores = quality_scores**self.alpha

        self.quality_scores = quality_scores.astype(np.float32)

        # Leak detection: mark samples with cosine similarity > 0.90 as leaks
        max_similarities = 1 - self.distances
        self.leak_mask = max_similarities > 0.90
        n_leaks = np.sum(self.leak_mask)

        print("Quality score statistics:")
        print(
            f"  Distance - Min: {np.min(self.distances):.4f}, Median: {np.median(self.distances):.4f}, Max: {np.max(self.distances):.4f}"
        )
        print(
            f"  Quality - Min: {np.min(quality_scores):.4f}, Median: {np.median(quality_scores):.4f}, Max: {np.max(quality_scores):.4f}"
        )
        print(f"  Detected {n_leaks} potential leaks (similarity > 0.90)")

        self.timings['quality_computation'] = time.time() - start_time

    def _get_data_id_quality_score(self, data_id: str) -> float:
        """Get the best quality score for a data_id (max across all its records)"""
        indices = self.data_id_to_indices[data_id]
        return np.max([self.quality_scores[idx] for idx in indices])

    def _get_data_id_embedding(self, data_id: str) -> np.ndarray:
        """Get representative embedding for a data_id (mean of all its records)"""
        indices = self.data_id_to_indices[data_id]
        embeddings = [self.candidate_embeddings[idx] for idx in indices]
        return np.mean(embeddings, axis=0)

    def diversity_quality_greedy_sampling(self) -> List[str]:
        """
        Fast greedy sampling balancing diversity and quality for DPO data.
        Selects data_ids rather than individual records.
        """
        start_time = time.time()
        print(f"Starting diversity-quality greedy sampling for k={self.k} data_ids...")
        print(f"Diversity weight: {self.diversity_weight}")

        selected_data_ids = []

        # Exclude leaked samples at data_id level
        valid_data_ids = []
        for data_id in self.unique_data_ids:
            indices = self.data_id_to_indices[data_id]
            # Check if any record in this data_id is not a leak
            if not all(self.leak_mask[idx] for idx in indices):
                valid_data_ids.append(data_id)

        print(f"Valid data_ids (non-leak): {len(valid_data_ids)}")

        # Track minimum distances to selected data_ids
        min_distances_to_selected = np.ones(len(valid_data_ids), dtype=np.float32)

        # Create progress bar for greedy sampling
        pbar = tqdm(range(min(self.k, len(valid_data_ids))), desc="Greedy sampling")

        for iteration in pbar:
            # Compute scores for all valid data_ids
            scores = np.full(len(valid_data_ids), -np.inf, dtype=np.float32)

            for i, data_id in enumerate(valid_data_ids):
                if data_id in selected_data_ids:
                    continue

                # Quality component (best quality score for this data_id)
                quality_component = self._get_data_id_quality_score(data_id)

                # Diversity component (distance to nearest selected data_id)
                diversity_component = min_distances_to_selected[i]

                # Combined score
                scores[i] = (
                    1 - self.diversity_weight
                ) * quality_component + self.diversity_weight * diversity_component

            # Select best data_id
            if np.max(scores) <= -np.inf:
                pbar.set_description(f"Warning: No more valid data_ids. Selected {len(selected_data_ids)} data_ids.")
                break

            best_idx = np.argmax(scores)
            best_data_id = valid_data_ids[best_idx]
            selected_data_ids.append(best_data_id)

            # Update distances to selected data_ids for all remaining data_ids
            if iteration < min(self.k, len(valid_data_ids)) - 1:
                best_embedding = self._get_data_id_embedding(best_data_id)

                for i, data_id in enumerate(valid_data_ids):
                    if data_id not in selected_data_ids:
                        current_embedding = self._get_data_id_embedding(data_id)
                        similarity = cosine_similarity(current_embedding.reshape(1, -1), best_embedding.reshape(1, -1))[
                            0, 0
                        ]
                        distance_to_new = 1 - similarity
                        min_distances_to_selected[i] = min(min_distances_to_selected[i], distance_to_new)

            # Update progress bar with current selection info
            score = scores[best_idx]
            quality = self._get_data_id_quality_score(best_data_id)
            diversity = min_distances_to_selected[best_idx] if iteration > 0 else 1.0
            pbar.set_postfix(
                {
                    'selected': best_data_id[-20:],  # Show last 20 chars of data_id
                    'score': f'{score:.3f}',
                    'quality': f'{quality:.3f}',
                    'diversity': f'{diversity:.3f}',
                }
            )

        self.timings['sampling'] = time.time() - start_time
        print(f"Completed diversity-quality greedy sampling in {self.timings['sampling']:.2f}s")
        return selected_data_ids

    def stratified_quality_sampling(self) -> List[str]:
        """Stratified sampling based on quality bins with diversity within bins for DPO data."""
        start_time = time.time()
        print(f"Starting stratified quality sampling for k={self.k} data_ids...")

        # Get quality scores for each data_id
        data_id_qualities = []
        valid_data_ids = []

        for data_id in self.unique_data_ids:
            indices = self.data_id_to_indices[data_id]
            # Check if any record in this data_id is not a leak
            if not all(self.leak_mask[idx] for idx in indices):
                valid_data_ids.append(data_id)
                data_id_qualities.append(self._get_data_id_quality_score(data_id))

        if len(valid_data_ids) == 0:
            print("No valid data_ids found!")
            return []

        data_id_qualities = np.array(data_id_qualities)

        # Create quality bins
        n_bins = min(10, self.k)
        bin_edges = np.percentile(data_id_qualities, np.linspace(0, 100, n_bins + 1))
        bin_edges[-1] += 1e-8  # Ensure max value is included

        # Assign samples per bin (more samples to higher quality bins)
        samples_per_bin = []
        remaining_k = self.k

        for i in range(n_bins):
            if i == n_bins - 1:  # Last bin gets remaining samples
                samples_per_bin.append(remaining_k)
            else:
                # Higher quality bins get more samples (quadratic weighting)
                weight = (i + 1) ** 2
                n_samples = max(1, int(self.k * weight / sum([(j + 1) ** 2 for j in range(n_bins)])))
                n_samples = min(n_samples, remaining_k - (n_bins - i - 1))
                samples_per_bin.append(n_samples)
                remaining_k -= n_samples

        print(f"Quality bins: {n_bins}, samples per bin: {samples_per_bin}")

        selected_data_ids = []

        # Create progress bar for bin processing
        pbar = tqdm(range(n_bins), desc="Processing quality bins")

        for bin_idx in pbar:
            # Find data_ids in this quality bin
            bin_mask = (data_id_qualities >= bin_edges[bin_idx]) & (data_id_qualities < bin_edges[bin_idx + 1])
            bin_data_ids = [valid_data_ids[i] for i in range(len(valid_data_ids)) if bin_mask[i]]

            if len(bin_data_ids) == 0:
                pbar.set_postfix({'bin': bin_idx + 1, 'candidates': 0, 'selected': 0})
                continue

            n_samples_bin = samples_per_bin[bin_idx]
            if len(bin_data_ids) <= n_samples_bin:
                # Take all data_ids in this bin
                selected_data_ids.extend(bin_data_ids)
                n_selected = len(bin_data_ids)
            else:
                # Use diversity-based selection within the bin
                selected_data_ids.extend(self._diverse_data_id_selection(bin_data_ids, n_samples_bin))
                n_selected = n_samples_bin

            pbar.set_postfix({'bin': bin_idx + 1, 'candidates': len(bin_data_ids), 'selected': n_selected})

        self.timings['sampling'] = time.time() - start_time
        print(f"Completed stratified quality sampling in {self.timings['sampling']:.2f}s")
        return selected_data_ids[: self.k]

    def _diverse_data_id_selection(self, data_ids: List[str], n_samples: int) -> List[str]:
        """Select diverse data_ids from a group using farthest-first traversal"""
        if len(data_ids) <= n_samples:
            return data_ids

        selected = []
        remaining = data_ids.copy()

        # Start with a random data_id
        first_data_id = self.rng.choice(remaining)
        selected.append(first_data_id)
        remaining.remove(first_data_id)

        # Greedily select farthest data_ids
        for _ in range(n_samples - 1):
            if not remaining:
                break

            # Compute distances to all selected data_ids
            max_min_distance = -1
            best_data_id = None

            for candidate_data_id in remaining:
                candidate_embedding = self._get_data_id_embedding(candidate_data_id)

                # Find minimum distance to any selected data_id
                min_distance = float('inf')
                for selected_data_id in selected:
                    selected_embedding = self._get_data_id_embedding(selected_data_id)
                    similarity = cosine_similarity(
                        candidate_embedding.reshape(1, -1), selected_embedding.reshape(1, -1)
                    )[0, 0]
                    dist = 1 - similarity
                    min_distance = min(min_distance, dist)

                # Track data_id with maximum minimum distance
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_data_id = candidate_data_id

            if best_data_id is not None:
                selected.append(best_data_id)
                remaining.remove(best_data_id)

        return selected

    def top_k_diverse_sampling(self) -> List[str]:
        """Select top quality data_ids and then diversify using clustering."""
        start_time = time.time()
        print(f"Starting top-k diverse sampling for k={self.k} data_ids...")

        # Get quality scores for each data_id
        data_id_qualities = []
        valid_data_ids = []

        for data_id in self.unique_data_ids:
            indices = self.data_id_to_indices[data_id]
            # Check if any record in this data_id is not a leak
            if not all(self.leak_mask[idx] for idx in indices):
                valid_data_ids.append(data_id)
                data_id_qualities.append(self._get_data_id_quality_score(data_id))

        data_id_qualities = np.array(data_id_qualities)

        # Select top candidates by quality
        top_ratio = min(3.0, len(valid_data_ids) / self.k)
        n_top = min(int(self.k * top_ratio), len(valid_data_ids))

        top_indices = np.argsort(data_id_qualities)[-n_top:]
        top_data_ids = [valid_data_ids[i] for i in top_indices]

        print(f"Selected top {len(top_data_ids)} data_ids from {len(valid_data_ids)} valid data_ids")

        if len(top_data_ids) <= self.k:
            selected_data_ids = top_data_ids
        else:
            # Use clustering for diversification
            print(f"Clustering {len(top_data_ids)} data_ids into {self.n_clusters} clusters...")

            # Get embeddings for top data_ids
            top_embeddings = np.array([self._get_data_id_embedding(data_id) for data_id in top_data_ids])

            # Use KMeans clustering
            n_clusters = min(self.n_clusters, len(top_data_ids))
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init=10)
            cluster_labels = kmeans.fit_predict(top_embeddings)

            # Select data_ids from each cluster
            selected_data_ids = []
            samples_per_cluster = self.k // n_clusters
            remaining_samples = self.k % n_clusters

            # Create progress bar for cluster processing
            pbar = tqdm(range(n_clusters), desc="Processing clusters")

            for cluster_id in pbar:
                cluster_mask = cluster_labels == cluster_id
                cluster_data_ids = [top_data_ids[i] for i in range(len(top_data_ids)) if cluster_mask[i]]
                cluster_qualities = [
                    data_id_qualities[top_indices[i]] for i in range(len(top_data_ids)) if cluster_mask[i]
                ]

                # Number of samples for this cluster
                n_samples_cluster = samples_per_cluster
                if cluster_id < remaining_samples:
                    n_samples_cluster += 1

                # Select best quality data_ids from this cluster
                if len(cluster_data_ids) <= n_samples_cluster:
                    selected_data_ids.extend(cluster_data_ids)
                    n_selected = len(cluster_data_ids)
                else:
                    best_in_cluster = np.argsort(cluster_qualities)[-n_samples_cluster:]
                    selected_data_ids.extend([cluster_data_ids[i] for i in best_in_cluster])
                    n_selected = n_samples_cluster

                pbar.set_postfix({'cluster': cluster_id, 'candidates': len(cluster_data_ids), 'selected': n_selected})

        self.timings['sampling'] = time.time() - start_time
        print(f"Completed top-k diverse sampling in {self.timings['sampling']:.2f}s")
        return selected_data_ids[: self.k]

    def random_quality_sampling(self) -> List[str]:
        """Simple quality-weighted random sampling for DPO data."""
        start_time = time.time()
        print(f"Starting quality-weighted random sampling for k={self.k} data_ids...")

        # Get quality scores for each data_id
        data_id_qualities = []
        valid_data_ids = []

        for data_id in self.unique_data_ids:
            indices = self.data_id_to_indices[data_id]
            # Check if any record in this data_id is not a leak
            if not all(self.leak_mask[idx] for idx in indices):
                valid_data_ids.append(data_id)
                data_id_qualities.append(self._get_data_id_quality_score(data_id))

        if len(valid_data_ids) == 0:
            print("No valid data_ids found!")
            return []

        data_id_qualities = np.array(data_id_qualities)

        # Normalize quality scores to get probabilities
        min_quality = np.min(data_id_qualities)
        if min_quality < 0:
            data_id_qualities = data_id_qualities - min_quality

        if np.sum(data_id_qualities) == 0:
            # If all qualities are zero, use uniform sampling
            probabilities = np.ones(len(data_id_qualities)) / len(data_id_qualities)
        else:
            probabilities = data_id_qualities / np.sum(data_id_qualities)

        # Sample without replacement
        n_samples = min(self.k, len(valid_data_ids))
        selected_indices = self.rng.choice(len(valid_data_ids), size=n_samples, replace=False, p=probabilities)
        selected_data_ids = [valid_data_ids[i] for i in selected_indices]

        self.timings['sampling'] = time.time() - start_time
        print(f"Completed quality-weighted random sampling in {self.timings['sampling']:.2f}s")
        return selected_data_ids

    def run_sampling(self) -> List[str]:
        """Run the complete fast sampling pipeline for DPO data"""
        print("=" * 60)
        print("Starting DPO Fast Sampling Pipeline")
        print("=" * 60)
        print("Parameters:")
        print(f"  k={self.k}, alpha={self.alpha}")
        print(f"  sigma_percentile={self.sigma_percentile}")
        print(f"  seed={self.seed}, dry_run={self.dry_run}")
        print(f"  max_token_length={self.max_token_length}")
        print(f"  sampling_method={self.sampling_method}")
        print(f"  diversity_weight={self.diversity_weight}")
        print()

        # Step 1: Filter by token length if specified
        self.filter_by_token_length()

        # Step 2: Compute quality scores
        self.compute_quality_scores()

        # Step 2.5: Check if k is larger than available data_ids
        valid_data_ids = []
        for data_id in self.unique_data_ids:
            indices = self.data_id_to_indices[data_id]
            # Check if any record in this data_id is not a leak
            if not all(self.leak_mask[idx] for idx in indices):
                valid_data_ids.append(data_id)

        n_available = len(valid_data_ids)
        print(f"Available non-leak data_ids: {n_available}")

        if self.k >= n_available:
            print(f"Target k={self.k} is >= available data_ids ({n_available})")
            print("Selecting ALL available data_ids instead of sampling...")
            selected_data_ids = valid_data_ids

            # Convert selected data_ids to all corresponding indices
            self.selected_indices = []
            for data_id in selected_data_ids:
                self.selected_indices.extend(self.data_id_to_indices[data_id])

            self.selected_data_ids = selected_data_ids
            print(f"Selected all {len(selected_data_ids)} data_ids with {len(self.selected_indices)} total records.")
            return selected_data_ids

        # Step 3: Run selected sampling method
        if self.sampling_method == "diversity_quality_greedy":
            selected_data_ids = self.diversity_quality_greedy_sampling()
        elif self.sampling_method == "stratified_quality":
            selected_data_ids = self.stratified_quality_sampling()
        elif self.sampling_method == "top_k_diverse":
            selected_data_ids = self.top_k_diverse_sampling()
        elif self.sampling_method == "random_quality":
            selected_data_ids = self.random_quality_sampling()
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_method}")

        self.selected_data_ids = selected_data_ids

        # Convert selected data_ids to all corresponding indices
        self.selected_indices = []
        for data_id in selected_data_ids:
            self.selected_indices.extend(self.data_id_to_indices[data_id])

        print(
            f"\nSampling completed! Selected {len(selected_data_ids)} data_ids with {len(self.selected_indices)} total records using {self.sampling_method}."
        )

        return selected_data_ids

    def _prepare_output_data(self) -> List[Dict]:
        """Prepare output data in DPO format"""
        if self.selected_data_ids is None:
            raise ValueError("No data_ids selected. Run sampling first.")

        output_data = []

        # If we have raw DPO data, use it for output
        if self.candidate_raw_df is not None:
            print("Using raw DPO data for output formatting...")
            for data_id in self.selected_data_ids:
                # Find all records with this data_id in raw data
                matching_records = self.candidate_raw_df[self.candidate_raw_df['data_id'] == data_id]

                for _, record in matching_records.iterrows():
                    if not record['subject']:
                        print(f"\033[93mWarning: No subject found for data_id: {data_id}\033[0m")
                    output_row = {
                        'data_id': record['data_id'],
                        'question': record['question'],
                        'prompt': record['prompt'],
                        'messages': record['messages'],
                        'rejected_messages': record['rejected_messages'],
                        'subject': record['subject'],
                    }
                    output_data.append(output_row)
        else:
            print("\033[93mWarning: No raw DPO data available. Using processed data for output...\033[0m")
            # Fallback to processed data
            for data_id in self.selected_data_ids:
                indices = self.data_id_to_indices[data_id]
                for idx in indices:
                    row = self.candidate_df.iloc[idx]

                    # Reconstruct DPO format from processed data (best effort)
                    output_row = {
                        'data_id': row.get('data_id', data_id),
                        'question': row.get('question', ''),
                        'prompt': row.get('question', ''),  # Use question as prompt
                        'messages': [
                            {'role': 'user', 'content': row.get('question', '')},
                            {'role': 'assistant', 'content': row.get('rationale', '')},
                        ],
                        'rejected_messages': [
                            {'role': 'user', 'content': row.get('question', '')},
                            {'role': 'assistant', 'content': 'Alternative response not available in processed data'},
                        ],
                        'subject': row.get('subject', ''),
                    }
                    output_data.append(output_row)

        return output_data

    def save_selected_jsonl(self, output_path: str) -> None:
        """Save selected samples to JSONL file in DPO format"""

        def convert_numpy_types(obj):
            """Convert numpy types to native Python types"""
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj

        output_data = self._prepare_output_data()
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in output_data:
                converted_item = convert_numpy_types(item)
                f.write(json.dumps(converted_item, ensure_ascii=False) + '\n')
        print(f"Saved {len(output_data)} selected DPO records to {output_path}")

    def save_parameters_json(self, output_path: str) -> None:
        """Save sampling parameters to JSON file"""
        params = {
            'k': self.k,
            'alpha': self.alpha,
            'sigma_percentile': self.sigma_percentile,
            'seed': self.seed,
            'dry_run': self.dry_run,
            'reference_category_filter': self.reference_category_filter,
            'reference_subject_filter': self.reference_subject_filter,
            'max_token_length': self.max_token_length,
            'sampling_method': self.sampling_method,
            'diversity_weight': self.diversity_weight,
            'n_clusters': self.n_clusters,
            'dataset_ratios': self.dataset_ratios,
            'timings': self.timings,
            'n_reference_samples': len(self.reference_embeddings) if self.reference_embeddings is not None else 0,
            'n_candidate_samples': len(self.candidate_embeddings) if self.candidate_embeddings is not None else 0,
            'n_unique_data_ids': len(self.unique_data_ids),
            'n_selected_data_ids': len(self.selected_data_ids) if self.selected_data_ids is not None else 0,
            'n_selected_records': len(self.selected_indices) if self.selected_indices is not None else 0,
        }

        with open(output_path, 'w') as f:
            json.dump(params, f, indent=2)
        print(f"Saved parameters to {output_path}")

    def create_umap_visualization(
        self, output_path: str, color_by: str = 'dataset', use_full_reference: bool = True
    ) -> None:
        """Create UMAP visualization comparing reference and selected candidates"""
        if self.selected_indices is None:
            raise ValueError("No samples selected. Run sampling first.")

        print(f"Creating UMAP visualization at {output_path}...")
        print(f"Coloring by: {color_by}")
        print(f"Using {'full' if use_full_reference else 'filtered'} reference data")

        try:
            # Import UMAP visualization function
            sys.path.append(str(Path(__file__).parent))
            from interactive_umap_explorer import InteractiveUMAPExplorer

            # Prepare combined data for UMAP
            selected_embeddings = self.candidate_embeddings[self.selected_indices]

            # Choose reference data based on use_full_reference flag
            if use_full_reference and hasattr(self, 'reference_embeddings_full'):
                reference_embeddings = self.reference_embeddings_full
                ref_df_viz = self.reference_df_full.copy()
            else:
                reference_embeddings = self.reference_embeddings
                ref_df_viz = self.reference_df.copy()

            combined_embeddings = np.vstack([reference_embeddings, selected_embeddings])

            # Prepare combined metadata
            ref_df_viz['dataset_type'] = 'REF Dataset'
            ref_df_viz['visualization_group'] = ref_df_viz['category']

            selected_df = self.candidate_df.iloc[self.selected_indices].copy()
            selected_df['dataset_type'] = 'Selected DPO Samples'
            selected_df['visualization_group'] = 'DPO Candidates Selected'

            combined_df = pd.concat([ref_df_viz, selected_df], ignore_index=True)

            # Create UMAP explorer
            explorer = InteractiveUMAPExplorer(combined_embeddings, combined_df)

            # Generate UMAP projection
            print("Computing UMAP projection...")
            explorer.compute_umap_projection(n_neighbors=15, min_dist=0.1)

            # Use visualization_group for coloring
            color_by = 'visualization_group'

            # Validate color_by column exists
            if color_by not in combined_df.columns:
                print(f"Warning: Column '{color_by}' not found. Available columns: {list(combined_df.columns)}")
                print("Using 'dataset_type' as fallback.")
                color_by = 'dataset_type'

            # Create interactive visualization
            ref_type = "full reference" if use_full_reference else "filtered reference"
            title = f"UMAP: {ref_type} vs Selected DPO Samples ({self.candidates_name}) - {self.sampling_method}"

            explorer.create_interactive_umap(color_by=color_by, title=title, save_path=output_path, show_plot=False)

            print(f"UMAP visualization saved to {output_path}")

            # Print coloring statistics
            if color_by in combined_df.columns:
                value_counts = combined_df[color_by].value_counts()
                print(f"Color distribution ({color_by}): {dict(value_counts)}")

        except Exception as e:
            print(f"Warning: Could not create UMAP visualization: {e}")
            import traceback

            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Fast data sampling for DPO dataset analysis')

    parser.add_argument('--k', type=int, default=100, help='Number of unique data_ids to select')
    parser.add_argument('--alpha', type=float, default=1.0, help='Power of quality score (q_i^alpha)')
    parser.add_argument(
        '--sigma_percentile', type=float, default=50.0, help='Percentile of distances for quality scoring sigma'
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--dry_run', action='store_true', help='Use small parameters for testing')

    parser.add_argument(
        '--reference_category_filter',
        type=str,
        default=None,
        help='Filter reference data by category. Use "all" to include all categories.',
    )
    parser.add_argument(
        '--reference_subject_filter',
        type=str,
        default=None,
        help='Filter reference data by subject. Use "all" to include all subjects.',
    )
    parser.add_argument(
        '--max_token_length', type=int, default=None, help='Maximum token length for candidate filtering'
    )
    parser.add_argument(
        '--candidate_paths', type=str, nargs='+', default=None, help='List of candidate dataset paths (space-separated)'
    )
    parser.add_argument(
        '--raw_data_paths', type=str, nargs='+', default=None, help='List of raw DPO data paths for output formatting'
    )
    parser.add_argument(
        '--dataset_ratios',
        type=str,
        default=None,
        help='Dataset sampling ratios as JSON string (e.g., \'{"dataset1": 0.4, "dataset2": 0.6}\')',
    )

    # DPO fast sampling specific arguments
    parser.add_argument(
        '--sampling_method',
        type=str,
        default='diversity_quality_greedy',
        choices=['diversity_quality_greedy', 'stratified_quality', 'top_k_diverse', 'random_quality'],
        help='Sampling method to use',
    )
    parser.add_argument(
        '--diversity_weight',
        type=float,
        default=0.5,
        help='Weight for diversity vs quality (0=only quality, 1=only diversity)',
    )
    parser.add_argument(
        '--n_clusters', type=int, default=None, help='Number of clusters for clustering-based methods (default: k//4)'
    )
    parser.add_argument('--create_umap', action='store_true', help='Create UMAP visualization of selected samples')

    args = parser.parse_args()

    # Parse dataset ratios if provided
    dataset_ratios = None
    if args.dataset_ratios:
        try:
            dataset_ratios = json.loads(args.dataset_ratios)
            print(f"Dataset ratios: {dataset_ratios}")
        except json.JSONDecodeError as e:
            print(f"Error parsing dataset ratios: {e}")
            print("Please provide ratios as valid JSON, e.g., '{\"dataset1\": 0.4, \"dataset2\": 0.6}'")
            return

    # Default raw data paths if not provided
    if args.raw_data_paths is None:
        args.raw_data_paths = [
            "input_data/DPO_006_1",
            "input_data/DPO_ArxivData_widerange",
        ]

    # Initialize sampler
    sampler = DPOFastSampler(
        k=args.k,
        alpha=args.alpha,
        sigma_percentile=args.sigma_percentile,
        seed=args.seed,
        dry_run=args.dry_run,
        reference_category_filter=args.reference_category_filter,
        reference_subject_filter=args.reference_subject_filter,
        max_token_length=args.max_token_length,
        sampling_method=args.sampling_method,
        diversity_weight=args.diversity_weight,
        n_clusters=args.n_clusters,
        dataset_ratios=dataset_ratios,
    )

    # Load data
    sampler.load_embeddings_data(candidate_paths=args.candidate_paths)

    # Load raw DPO data for proper output formatting
    sampler.load_raw_dpo_data(args.raw_data_paths)

    # Validate data alignment to prevent duplication issues
    sampler.validate_data_alignment()

    # Run sampling
    selected_data_ids = sampler.run_sampling()

    # Create output directory
    candidates_name = sampler.candidates_name
    if len(candidates_name) > 50:
        # Shorten long names
        candidates_name = "combined_dpo_candidates"

    # Create filter-specific folder name
    if args.reference_category_filter:
        filter_name = args.reference_category_filter.lower().replace('/', '_').replace(' ', '_')
    elif args.reference_subject_filter:
        filter_name = args.reference_subject_filter.lower().replace('/', '_').replace(' ', '_')
    else:
        filter_name = "no_filter"

    output_dir = Path("fast_sampling_results") / f"dpo_{candidates_name}_{filter_name}_{args.sampling_method}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results in DPO format
    jsonl_path = output_dir / f"selected_dpo_{candidates_name}.jsonl"
    sampler.save_selected_jsonl(str(jsonl_path))

    params_path = output_dir / f"params_dpo_{candidates_name}.json"
    sampler.save_parameters_json(str(params_path))

    # Create UMAP visualization if requested
    if args.create_umap:
        umap_path = output_dir / f"umap_visualization_dpo_{candidates_name}.html"
        print("\nCreating UMAP visualization...")
        sampler.create_umap_visualization(str(umap_path), use_full_reference=True)

    print("\n" + "=" * 60)
    print("DPO fast sampling completed successfully!")
    print(f"Method: {args.sampling_method}")
    print(f"Selected {len(selected_data_ids)} unique data_ids from {candidates_name}")
    print(f"Total records: {len(sampler.selected_indices)}")
    print("Output files:")
    print(f"  JSONL: {jsonl_path}")
    print(f"  Parameters: {params_path}")
    if args.create_umap:
        print(f"  UMAP Visualization: {umap_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
