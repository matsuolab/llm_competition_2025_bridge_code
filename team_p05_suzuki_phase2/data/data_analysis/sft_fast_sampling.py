#!/usr/bin/env python3
"""
Fast Data Sampling Methods for SFT Dataset Analysis
=================================================

This script implements several lightweight sampling algorithms as alternatives to k-DPP
for selecting diverse, high-quality samples from candidate datasets.

Available methods:
1. Diversity-Quality Greedy: Fast greedy selection balancing diversity and quality
2. Stratified Quality: Stratified sampling based on quality bins with diversity within bins
3. Top-K Diverse: Select top quality samples and then diversify using clustering
4. Random Quality: Simple quality-weighted random sampling

Author: GitHub Copilot
"""

import argparse
import json
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Import token length utilities
from token_length_histogram_utils import calculate_token_lengths, load_tokenizer


class SFTFastSampler:
    """Fast sampler for selecting diverse, high-quality data points from candidate datasets"""

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
        Initialize the fast sampler

        Args:
            k: Number of samples to select
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
        self.candidates_name: str = ""

        # Results storage
        self.quality_scores: Optional[np.ndarray] = None
        self.distances: Optional[np.ndarray] = None
        self.selected_indices: Optional[List[int]] = None
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
        reference_path: str = "hle_question_analysis_output/embeddings/hle_embeddings.pkl",
        candidate_paths: List[str] = None,
        output_dir: str = "analysis_outputs",
    ) -> None:
        """Load reference and candidate embeddings from pickle files"""
        if candidate_paths is None:
            candidate_paths = [
                # "sft_006_origin_1_analysis_output/embeddings/sft_006_origin_1_embeddings.pkl",
                # "seed_openscience_analysis_output/embeddings/seed_openscience_embeddings.pkl",
                # "openmathreasoning_16k_analysis_output/embeddings/openmathreasoning_16k_embeddings.pkl",
                # "arxiv_data_analysis_output/embeddings/arxiv_data_embeddings.pkl",
                "sft_007_1_filtered_2_analysis_output/embeddings/sft_007_1_filtered_2_embeddings.pkl",
                "seed_000_openmath_16k_30bfiltered_analysis_output/embeddings/seed_000_openmath_16k_30bfiltered_embeddings.pkl",
                "seed_000_openscience_16k_30bfiltered_analysis_output/embeddings/seed_000_openscience_16k_30bfiltered_embeddings.pkl",
                "arxiv_data_30bfiltered_analysis_output/embeddings/arxiv_data_30bfiltered_embeddings.pkl",
            ]

        # self.dataset_ratios = {
        #     "SFT_007_1_FILTERED_2": 0.3,
        #     "SEED_000_OPENMATH_16K_30BFILTERED": 0.1,
        #     "SEED_000_OPENSCIENCE_16K_30BFILTERED": 0.3,
        #     "ARXIV_DATA_30BFILTERED": 0.3,
        # }

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
        original_ref_count = len(reference_df)
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

        print(f"Candidate data: {len(self.candidate_embeddings)} samples from {len(candidate_paths)} sources")

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
        dataset_hash = hash(str(sorted(self.candidate_df[['question', 'rationale', 'answer']].values.tolist())))
        cache_file = cache_dir / f"token_lengths_{self.candidates_name}_{dataset_hash}.pkl"

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
            # The calculate_token_lengths function expects a DataFrame with 'question', 'rationale', 'answer' columns
            questions_df = self.candidate_df[['question', 'rationale', 'answer']].copy()

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

        self.candidate_embeddings = self.candidate_embeddings[length_mask]
        self.candidate_df = self.candidate_df[length_mask].reset_index(drop=True)
        self.token_lengths = self.token_lengths[length_mask]

        print(f"Filtered out {n_filtered} samples exceeding token length limit")
        print(f"Remaining candidates: {len(self.candidate_embeddings)}")

        self.timings['token_filtering'] = time.time() - start_time

    def _simple_ratio_sampling(self) -> List[int]:
        """Simple ratio-based sampling: sample from each dataset separately and combine"""
        if self.dataset_ratios is None:
            return []

        print("Using simple ratio-based sampling...")
        start_time = time.time()

        # Calculate target samples per dataset
        target_samples = {}
        for dataset, ratio in self.dataset_ratios.items():
            target_samples[dataset] = int(self.k * ratio)

        # Ensure we don't exceed k
        total_target = sum(target_samples.values())
        if total_target > self.k:
            # Scale down proportionally
            scale_factor = self.k / total_target
            target_samples = {k: int(v * scale_factor) for k, v in target_samples.items()}

        # Add remaining samples to the largest dataset
        remaining = self.k - sum(target_samples.values())
        if remaining > 0:
            largest_dataset = max(target_samples.keys(), key=lambda k: target_samples[k])
            target_samples[largest_dataset] += remaining

        print(f"Target samples per dataset: {target_samples}")

        # Sample from each dataset separately
        all_selected_indices = []

        for dataset, target_count in target_samples.items():
            if target_count == 0:
                continue

            # Find candidates from this dataset
            dataset_mask = self.candidate_df['dataset'] == dataset
            valid_mask = ~self.leak_mask
            valid_dataset_mask = valid_mask & dataset_mask
            valid_dataset_indices = np.where(valid_dataset_mask)[0]

            if len(valid_dataset_indices) == 0:
                print(f"No valid candidates found for dataset {dataset}")
                continue

            # Create a temporary sampler for this dataset
            temp_sampler = SFTFastSampler(
                k=target_count,
                alpha=self.alpha,
                sigma_percentile=self.sigma_percentile,
                seed=self.seed,
                dry_run=self.dry_run,
                reference_category_filter=self.reference_category_filter,
                reference_subject_filter=self.reference_subject_filter,
                max_token_length=self.max_token_length,
                sampling_method=self.sampling_method,
                diversity_weight=self.diversity_weight,
                n_clusters=self.n_clusters,
                dataset_ratios=None,  # No ratios for individual dataset sampling
            )

            # Set up the temporary sampler with data from this dataset only
            temp_sampler.reference_embeddings = self.reference_embeddings
            temp_sampler.reference_df = self.reference_df
            temp_sampler.candidate_embeddings = self.candidate_embeddings[valid_dataset_indices]
            temp_sampler.candidate_df = self.candidate_df.iloc[valid_dataset_indices].reset_index(drop=True)
            temp_sampler.quality_scores = self.quality_scores[valid_dataset_indices]
            temp_sampler.leak_mask = self.leak_mask[valid_dataset_indices]
            temp_sampler.distances = self.distances[valid_dataset_indices]
            temp_sampler.nearest_ref_indices = self.nearest_ref_indices[valid_dataset_indices]

            # Run sampling for this dataset
            if self.sampling_method == "diversity_quality_greedy":
                dataset_selected = temp_sampler.diversity_quality_greedy_sampling()
            elif self.sampling_method == "stratified_quality":
                dataset_selected = temp_sampler.stratified_quality_sampling()
            elif self.sampling_method == "top_k_diverse":
                dataset_selected = temp_sampler.top_k_diverse_sampling()
            elif self.sampling_method == "random_quality":
                dataset_selected = temp_sampler.random_quality_sampling()
            else:
                raise ValueError(f"Unknown sampling method: {self.sampling_method}")

            # Convert local indices to global indices
            global_selected = [valid_dataset_indices[i] for i in dataset_selected]
            all_selected_indices.extend(global_selected)

            print(f"Dataset {dataset}: Selected {len(global_selected)} samples")

        self.timings['ratio_sampling'] = time.time() - start_time
        return all_selected_indices

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

    def diversity_quality_greedy_sampling(self) -> List[int]:
        """
        Fast greedy sampling balancing diversity and quality.

        At each step, select the candidate that maximizes:
        weighted_score = (1-diversity_weight) * quality + diversity_weight * min_distance_to_selected
        """
        start_time = time.time()
        print(f"Starting diversity-quality greedy sampling for k={self.k}...")
        print(f"Diversity weight: {self.diversity_weight}")

        # If dataset ratios are specified, use simple ratio-based sampling
        if self.dataset_ratios is not None:
            selected_indices = self._simple_ratio_sampling()
            self.timings['sampling'] = time.time() - start_time
            print(f"Completed diversity-quality greedy sampling in {self.timings['sampling']:.2f}s")
            return selected_indices

        n_candidates = len(self.candidate_embeddings)
        selected_indices = []

        # Exclude leaked samples
        valid_mask = ~self.leak_mask

        # Track minimum distances to selected samples
        min_distances_to_selected = np.ones(n_candidates, dtype=np.float32)  # Start with max distance

        # Create progress bar for greedy sampling
        pbar = tqdm(range(self.k), desc="Greedy sampling")

        for iteration in pbar:
            # Compute scores for all valid candidates
            scores = np.full(n_candidates, -np.inf, dtype=np.float32)

            for i in range(n_candidates):
                if not valid_mask[i]:
                    continue

                # Quality component (normalized)
                quality_component = self.quality_scores[i]

                # Diversity component (distance to nearest selected sample)
                diversity_component = min_distances_to_selected[i]

                # Combined score
                scores[i] = (
                    1 - self.diversity_weight
                ) * quality_component + self.diversity_weight * diversity_component

            # Select best candidate
            if np.max(scores) <= -np.inf:
                pbar.set_description(f"Warning: No more valid candidates. Selected {len(selected_indices)} samples.")
                break

            best_idx = np.argmax(scores)
            selected_indices.append(best_idx)
            valid_mask[best_idx] = False

            # Update distances to selected samples for all candidates
            if iteration < self.k - 1:  # No need to update on last iteration
                new_embedding = self.candidate_embeddings[best_idx : best_idx + 1]
                similarities = cosine_similarity(self.candidate_embeddings, new_embedding).flatten()
                distances_to_new = 1 - similarities
                min_distances_to_selected = np.minimum(min_distances_to_selected, distances_to_new)

            # Update progress bar with current selection info
            if iteration < len(selected_indices):
                score = scores[best_idx]
                quality = self.quality_scores[best_idx]
                diversity = min_distances_to_selected[best_idx] if iteration > 0 else 1.0
                pbar.set_postfix(
                    {
                        'selected': best_idx,
                        'score': f'{score:.3f}',
                        'quality': f'{quality:.3f}',
                        'diversity': f'{diversity:.3f}',
                    }
                )

        self.timings['sampling'] = time.time() - start_time
        print(f"Completed diversity-quality greedy sampling in {self.timings['sampling']:.2f}s")
        return selected_indices

    def stratified_quality_sampling(self) -> List[int]:
        """
        Stratified sampling based on quality bins with diversity within bins.
        """
        start_time = time.time()
        print(f"Starting stratified quality sampling for k={self.k}...")

        # If dataset ratios are specified, use simple ratio-based sampling
        if self.dataset_ratios is not None:
            selected_indices = self._simple_ratio_sampling()
            self.timings['sampling'] = time.time() - start_time
            print(f"Completed stratified quality sampling in {self.timings['sampling']:.2f}s")
            return selected_indices

        # Create quality bins
        n_bins = min(10, self.k)
        valid_mask = ~self.leak_mask
        valid_qualities = self.quality_scores[valid_mask]

        if len(valid_qualities) == 0:
            print("No valid candidates found!")
            return []

        # Define bin edges
        bin_edges = np.percentile(valid_qualities, np.linspace(0, 100, n_bins + 1))
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
                n_samples = min(n_samples, remaining_k - (n_bins - i - 1))  # Ensure at least 1 for remaining bins
                samples_per_bin.append(n_samples)
                remaining_k -= n_samples

        print(f"Quality bins: {n_bins}, samples per bin: {samples_per_bin}")

        selected_indices = []
        valid_indices = np.where(valid_mask)[0]

        # Create progress bar for bin processing
        pbar = tqdm(range(n_bins), desc="Processing quality bins")

        for bin_idx in pbar:
            # Find candidates in this quality bin
            bin_mask = (self.quality_scores >= bin_edges[bin_idx]) & (self.quality_scores < bin_edges[bin_idx + 1])
            bin_mask &= valid_mask
            bin_candidates = np.where(bin_mask)[0]

            if len(bin_candidates) == 0:
                pbar.set_postfix({'bin': bin_idx + 1, 'candidates': 0, 'selected': 0})
                continue

            n_samples_bin = samples_per_bin[bin_idx]
            if len(bin_candidates) <= n_samples_bin:
                # Take all candidates in this bin
                selected_indices.extend(bin_candidates.tolist())
                n_selected = len(bin_candidates)
            else:
                # Use diversity-based selection within the bin
                bin_embeddings = self.candidate_embeddings[bin_candidates]
                bin_selected = self._diverse_selection_within_group(bin_embeddings, n_samples_bin)
                selected_indices.extend([bin_candidates[i] for i in bin_selected])
                n_selected = n_samples_bin

            # Mark selected candidates as invalid
            for idx in bin_candidates:
                if idx in selected_indices:
                    valid_mask[idx] = False

            pbar.set_postfix({'bin': bin_idx + 1, 'candidates': len(bin_candidates), 'selected': n_selected})

        self.timings['sampling'] = time.time() - start_time
        print(f"Completed stratified quality sampling in {self.timings['sampling']:.2f}s")
        return selected_indices[: self.k]  # Ensure we don't exceed k

    def _diverse_selection_within_group(self, embeddings: np.ndarray, n_samples: int) -> List[int]:
        """Select diverse samples from a group using farthest-first traversal"""
        if len(embeddings) <= n_samples:
            return list(range(len(embeddings)))

        selected = []
        remaining = list(range(len(embeddings)))

        # Start with a random sample
        first_idx = self.rng.choice(remaining)
        selected.append(first_idx)
        remaining.remove(first_idx)

        # Greedily select farthest samples
        for _ in range(n_samples - 1):
            if not remaining:
                break

            # Compute distances to all selected samples
            max_min_distance = -1
            best_idx = None

            for candidate_idx in remaining:
                # Find minimum distance to any selected sample
                min_distance = float('inf')
                for selected_idx in selected:
                    dist = (
                        1
                        - cosine_similarity(
                            embeddings[candidate_idx : candidate_idx + 1], embeddings[selected_idx : selected_idx + 1]
                        )[0, 0]
                    )
                    min_distance = min(min_distance, dist)

                # Track candidate with maximum minimum distance
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_idx = candidate_idx

            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)

        return selected

    def top_k_diverse_sampling(self) -> List[int]:
        """
        Select top quality samples and then diversify using clustering.
        """
        start_time = time.time()
        print(f"Starting top-k diverse sampling for k={self.k}...")

        # If dataset ratios are specified, use simple ratio-based sampling
        if self.dataset_ratios is not None:
            selected_indices = self._simple_ratio_sampling()
            self.timings['sampling'] = time.time() - start_time
            print(f"Completed top-k diverse sampling in {self.timings['sampling']:.2f}s")
            return selected_indices

        # Select top candidates by quality (excluding leaks)
        valid_mask = ~self.leak_mask
        valid_indices = np.where(valid_mask)[0]
        valid_qualities = self.quality_scores[valid_mask]

        # Take top quality candidates (more than k to allow for diversification)
        top_ratio = min(3.0, len(valid_indices) / self.k)  # Take up to 3x more candidates
        n_top = min(int(self.k * top_ratio), len(valid_indices))

        top_indices_in_valid = np.argsort(valid_qualities)[-n_top:]
        top_candidate_indices = valid_indices[top_indices_in_valid]

        print(f"Selected top {len(top_candidate_indices)} candidates from {len(valid_indices)} valid candidates")

        if len(top_candidate_indices) <= self.k:
            selected_indices = top_candidate_indices.tolist()
        else:
            # Use clustering for diversification
            print(f"Clustering {len(top_candidate_indices)} candidates into {self.n_clusters} clusters...")

            top_embeddings = self.candidate_embeddings[top_candidate_indices]

            # Use KMeans clustering
            n_clusters = min(self.n_clusters, len(top_candidate_indices))
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init=10)
            cluster_labels = kmeans.fit_predict(top_embeddings)

            # Select samples from each cluster
            selected_indices = []
            samples_per_cluster = self.k // n_clusters
            remaining_samples = self.k % n_clusters

            # Create progress bar for cluster processing
            pbar = tqdm(range(n_clusters), desc="Processing clusters")

            for cluster_id in pbar:
                cluster_mask = cluster_labels == cluster_id
                cluster_indices = top_candidate_indices[cluster_mask]
                cluster_qualities = self.quality_scores[cluster_indices]

                # Number of samples for this cluster
                n_samples_cluster = samples_per_cluster
                if cluster_id < remaining_samples:
                    n_samples_cluster += 1

                # Select best quality samples from this cluster
                if len(cluster_indices) <= n_samples_cluster:
                    selected_indices.extend(cluster_indices.tolist())
                    n_selected = len(cluster_indices)
                else:
                    best_in_cluster = np.argsort(cluster_qualities)[-n_samples_cluster:]
                    selected_indices.extend(cluster_indices[best_in_cluster].tolist())
                    n_selected = n_samples_cluster

                pbar.set_postfix({'cluster': cluster_id, 'candidates': len(cluster_indices), 'selected': n_selected})

        self.timings['sampling'] = time.time() - start_time
        print(f"Completed top-k diverse sampling in {self.timings['sampling']:.2f}s")
        return selected_indices[: self.k]

    def random_quality_sampling(self) -> List[int]:
        """
        Simple quality-weighted random sampling.
        """
        start_time = time.time()
        print(f"Starting quality-weighted random sampling for k={self.k}...")

        # If dataset ratios are specified, use simple ratio-based sampling
        if self.dataset_ratios is not None:
            selected_indices = self._simple_ratio_sampling()
            self.timings['sampling'] = time.time() - start_time
            print(f"Completed quality-weighted random sampling in {self.timings['sampling']:.2f}s")
            return selected_indices

        # Get valid candidates (excluding leaks)
        valid_mask = ~self.leak_mask
        valid_indices = np.where(valid_mask)[0]
        valid_qualities = self.quality_scores[valid_mask]

        if len(valid_indices) == 0:
            print("No valid candidates found!")
            return []

        # Normalize quality scores to get probabilities
        min_quality = np.min(valid_qualities)
        if min_quality < 0:
            valid_qualities = valid_qualities - min_quality  # Ensure non-negative

        if np.sum(valid_qualities) == 0:
            # If all qualities are zero, use uniform sampling
            probabilities = np.ones(len(valid_qualities)) / len(valid_qualities)
        else:
            probabilities = valid_qualities / np.sum(valid_qualities)

        # Sample without replacement
        n_samples = min(self.k, len(valid_indices))
        selected_in_valid = self.rng.choice(len(valid_indices), size=n_samples, replace=False, p=probabilities)
        selected_indices = valid_indices[selected_in_valid].tolist()

        self.timings['sampling'] = time.time() - start_time
        print(f"Completed quality-weighted random sampling in {self.timings['sampling']:.2f}s")
        return selected_indices

    def run_sampling(self) -> List[int]:
        """Run the complete fast sampling pipeline"""
        print("=" * 60)
        print("Starting Fast Sampling Pipeline")
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

        # Step 3: Run selected sampling method
        if self.sampling_method == "diversity_quality_greedy":
            selected_indices = self.diversity_quality_greedy_sampling()
        elif self.sampling_method == "stratified_quality":
            selected_indices = self.stratified_quality_sampling()
        elif self.sampling_method == "top_k_diverse":
            selected_indices = self.top_k_diverse_sampling()
        elif self.sampling_method == "random_quality":
            selected_indices = self.random_quality_sampling()
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_method}")

        self.selected_indices = selected_indices
        print(f"\nSampling completed! Selected {len(selected_indices)} samples using {self.sampling_method}.")

        return selected_indices

    def _extract_answer_from_rationale(self, rationale: str) -> str:
        """
        Extract answer from rationale using \\boxed{answer} pattern.
        Returns the extracted answer or empty string if not found.
        Handles nested braces in LaTeX expressions.
        """
        if not rationale or pd.isna(rationale):
            return ""

        def extract_balanced_braces(text, start_pos):
            """Extract content within balanced braces starting from start_pos"""
            if start_pos >= len(text) or text[start_pos] != '{':
                return ""

            brace_count = 0
            content = ""
            i = start_pos

            while i < len(text):
                char = text[i]
                if char == '{':
                    brace_count += 1
                    if brace_count > 1:  # Don't include the opening brace
                        content += char
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:  # Found matching closing brace
                        return content
                    else:
                        content += char
                else:
                    if brace_count > 0:
                        content += char
                i += 1

            return ""  # Unmatched braces

        # Look for \\boxed{ patterns with different backslash counts
        patterns = [r'\\\\boxed\{', r'\\boxed\{', r'boxed\{']

        for pattern in patterns:
            matches = list(re.finditer(pattern, rationale))
            if matches:
                # Use the last match (usually the final answer)
                last_match = matches[-1]
                start_pos = last_match.end() - 1  # Position of the opening brace
                content = extract_balanced_braces(rationale, start_pos)
                if content:
                    return content.strip()

        return ""

    def _prepare_output_data(self) -> List[Dict]:
        """Prepare output data for both CSV and JSONL formats"""
        if self.selected_indices is None:
            raise ValueError("No samples selected. Run sampling first.")

        output_data = []
        for idx in self.selected_indices:
            row = self.candidate_df.iloc[idx].copy()

            # Add nearest reference information
            nearest_ref_idx = self.nearest_ref_indices[idx]
            nearest_ref_question = self.reference_df.iloc[nearest_ref_idx].get('question', 'N/A')
            nearest_cosine_sim = 1 - self.distances[idx]  # Convert distance back to similarity

            formatted_id = row.get('id', idx).replace('_', '-')

            # Get answer, extract from rationale if empty
            answer = row.get('answer', '')
            if not answer or pd.isna(answer) or answer.strip() == "":
                rationale = row.get('rationale', '')
                extracted_answer = self._extract_answer_from_rationale(rationale)
                answer = extracted_answer if extracted_answer else 'N/A'

            output_row = {
                'id': formatted_id,
                'question': row.get('question', 'N/A'),
                'think': row.get('rationale', 'N/A'),
                'answer': answer,
                'dataset_name': row.get('dataset', 'unknown'),
                'subject': row.get('category', 'N/A'),
                'category': row.get('category', 'N/A'),
                'nearest_bench_question': nearest_ref_question,
                'nearest_bench_cosine_similarity': nearest_cosine_sim,
            }

            # Add token length information if available
            if hasattr(self, 'token_lengths') and self.token_lengths is not None:
                if idx < len(self.token_lengths):
                    output_row['token_length'] = int(self.token_lengths[idx])

            output_data.append(output_row)

        return output_data

    def save_selected_csv(self, output_path: str) -> None:
        """Save selected samples to CSV file"""
        output_data = self._prepare_output_data()
        df = pd.DataFrame(output_data)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} selected samples to {output_path}")

    def save_selected_jsonl(self, output_path: str) -> None:
        """Save selected samples to JSONL file"""

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
        print(f"Saved {len(output_data)} selected samples to {output_path}")

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
            'n_selected_samples': len(self.selected_indices) if self.selected_indices is not None else 0,
        }

        with open(output_path, 'w') as f:
            json.dump(params, f, indent=2)
        print(f"Saved parameters to {output_path}")

    def create_umap_visualization(
        self, output_path: str, color_by: str = 'dataset', use_full_reference: bool = True
    ) -> None:
        """
        Create UMAP visualization comparing reference and selected candidates

        Args:
            output_path: Path to save the HTML visualization
            color_by: Column to color points by ('dataset', 'category', 'subject', 'raw_subject')
            use_full_reference: If True, use full reference data instead of filtered reference data
        """
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
            # For HLE data, use category for coloring
            ref_df_viz['dataset_type'] = 'HLE Dataset'
            ref_df_viz['visualization_group'] = ref_df_viz['category']  # Use category for coloring HLE data

            selected_df = self.candidate_df.iloc[self.selected_indices].copy()
            selected_df['dataset_type'] = 'Selected Samples'
            selected_df['visualization_group'] = 'Candidates Selected'  # All selected samples in one group

            combined_df = pd.concat([ref_df_viz, selected_df], ignore_index=True)

            # Create UMAP explorer
            explorer = InteractiveUMAPExplorer(combined_embeddings, combined_df)

            # Generate UMAP projection
            print("Computing UMAP projection...")
            explorer.compute_umap_projection(n_neighbors=15, min_dist=0.1)

            # Use visualization_group for coloring (HLE by category, selected samples as one group)
            color_by = 'visualization_group'

            # Validate color_by column exists
            if color_by not in combined_df.columns:
                print(f"Warning: Column '{color_by}' not found. Available columns: {list(combined_df.columns)}")
                print("Using 'dataset_type' as fallback.")
                color_by = 'dataset_type'

            # Create interactive visualization
            ref_type = "full reference" if use_full_reference else "filtered reference"
            title = f"UMAP: {ref_type} (colored by category) vs Selected Samples ({self.candidates_name}) - {self.sampling_method}"

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
    parser = argparse.ArgumentParser(description='Fast data sampling for HLE dataset analysis')

    parser.add_argument('--k', type=int, default=100, help='Number of samples to select')
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
        '--dataset_ratios',
        type=str,
        default=None,
        help='Dataset sampling ratios as JSON string (e.g., \'{"dataset1": 0.4, "dataset2": 0.6}\')',
    )

    # New fast sampling specific arguments
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

    # Initialize sampler
    sampler = SFTFastSampler(
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

    # Run sampling
    selected_indices = sampler.run_sampling()

    # Create output directory
    candidates_name = sampler.candidates_name
    if len(candidates_name) > 50:
        # Shorten long names
        candidates_name = "combined_candidates"

    # Create filter-specific folder name
    if args.reference_category_filter:
        filter_name = args.reference_category_filter.lower().replace('/', '_').replace(' ', '_')
    elif args.reference_subject_filter:
        filter_name = args.reference_subject_filter.lower().replace('/', '_').replace(' ', '_')
    else:
        filter_name = "no_filter"

    output_dir = Path("fast_sampling_results") / f"{candidates_name}_{filter_name}_{args.sampling_method}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    csv_path = output_dir / f"selected_{candidates_name}.csv"
    sampler.save_selected_csv(str(csv_path))

    jsonl_path = output_dir / f"selected_{candidates_name}.jsonl"
    sampler.save_selected_jsonl(str(jsonl_path))

    params_path = output_dir / f"params_{candidates_name}.json"
    sampler.save_parameters_json(str(params_path))

    # Create UMAP visualization if requested
    if args.create_umap:
        umap_path = output_dir / f"umap_visualization_{candidates_name}.html"
        print(f"\nCreating UMAP visualization...")
        sampler.create_umap_visualization(str(umap_path), use_full_reference=True)

    print("\n" + "=" * 60)
    print("Fast sampling completed successfully!")
    print(f"Method: {args.sampling_method}")
    print(f"Selected {len(selected_indices)} samples from {candidates_name}")
    print("Output files:")
    print(f"  CSV: {csv_path}")
    print(f"  JSONL: {jsonl_path}")
    print(f"  Parameters: {params_path}")
    if args.create_umap:
        print(f"  UMAP Visualization: {umap_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
