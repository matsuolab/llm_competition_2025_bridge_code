#!/usr/bin/env python3
"""
k-DPP Data Sampling with MAP Approximation for HLE Dataset Analysis
==================================================================

This script implements k-DPP (k-Determinantal Point Process) sampling with MAP approximation
using greedy log-det maximization to select diverse, high-quality samples from candidate
datasets based on their similarity to a reference benchmark dataset.

The algorithm uses:
- Quality scoring based on cosine similarity to reference dataset
- RBF kernel approximation via Random Fourier Features (RFF)
- Greedy MAP selection with incremental orthogonalization
- UMAP visualization and comprehensive output logging

Author: Generated with Claude Code
"""

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class HLESimilarDataSampler:
    """k-DPP sampler for selecting diverse, high-quality data points from candidate datasets"""

    def __init__(
        self,
        k: int = 100,
        alpha: float = 1.0,
        sim_gamma: Optional[float] = None,
        rff_dim: int = 512,
        sigma_percentile: float = 50.0,
        seed: int = 42,
        dry_run: bool = False,
        reference_category_filter: Optional[str] = None,
        reference_subject_filter: Optional[str] = None,
    ):
        """
        Initialize the k-DPP sampler

        Args:
            k: Number of samples to select
            alpha: Power of quality score (q_i^alpha)
            sim_gamma: RBF kernel gamma parameter (1/(2*l^2)). If None, estimated automatically
            rff_dim: Dimension of Random Fourier Features approximation
            sigma_percentile: Percentile of distances to use for quality scoring sigma
            seed: Random seed for reproducibility
            dry_run: If True, use small parameters for testing
            reference_category_filter: Filter reference data by category (e.g., 'chemistry')
            reference_subject_filter: Filter reference data by subject (e.g., 'chemistry')
        """
        self.k = k if not dry_run else min(k, 10)
        self.alpha = alpha
        self.sim_gamma = sim_gamma
        self.rff_dim = rff_dim if not dry_run else min(rff_dim, 128)
        self.sigma_percentile = sigma_percentile
        self.seed = seed
        self.dry_run = dry_run
        self.reference_category_filter = reference_category_filter
        self.reference_subject_filter = reference_subject_filter

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

        # Timing information
        self.timings: Dict[str, float] = {}

    def load_embeddings_data(
        self,
        reference_path: str = "hle_question_analysis_output/embeddings/hle_embeddings.pkl",
        candidate_paths: List[str] = None,
        output_dir: str = "analysis_outputs",
    ) -> None:
        """
        Load reference and candidate embeddings from pickle files

        Args:
            reference_path: Path to reference dataset embeddings
            candidate_paths: List of paths to candidate dataset embeddings
            output_dir: Base directory for analysis outputs
        """
        if candidate_paths is None:
            candidate_paths = [
                "gpqa_analysis_output/embeddings/gpqa_embeddings.pkl",
                "supergpqa_analysis_output/embeddings/supergpqa_embeddings.pkl",
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

        # Apply category and subject filters to reference data
        original_ref_count = len(reference_df)
        filter_mask = np.ones(len(reference_df), dtype=bool)

        if self.reference_category_filter:
            print(f"Filtering reference data by category: {self.reference_category_filter}")
            category_mask = reference_df['category'].str.contains(self.reference_category_filter, case=False, na=False)
            filter_mask &= category_mask

        if self.reference_subject_filter:
            print(f"Filtering reference data by subject: {self.reference_subject_filter}")
            # Check both 'subject' and 'raw_subject' columns
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

        # L2 normalize reference embeddings
        self.reference_embeddings = self._l2_normalize(self.reference_embeddings)

        if self.reference_category_filter or self.reference_subject_filter:
            print(f"Filtered reference data: {original_ref_count} -> {len(self.reference_embeddings)} samples")
        else:
            print(f"Loaded {len(self.reference_embeddings)} reference samples")

        # Load and combine candidate datasets
        all_candidate_embeddings = []
        all_candidate_dfs = []
        candidate_names = []

        for cand_path in candidate_paths:
            print(f"Loading candidate embeddings from {cand_path}...")
            cand_full_path = Path(output_dir) / cand_path

            with open(cand_full_path, 'rb') as f:
                cand_data = pickle.load(f)
                embeddings = np.array(cand_data['embeddings'], dtype=np.float32)
                df = pd.DataFrame(cand_data['metadata'])
                # Add dataset identifier
                dataset_name = Path(cand_path).stem.replace('_embeddings', '').upper()
                df['dataset'] = dataset_name

                # Add dataset name from path
                dataset_name = Path(cand_path).parent.parent.name.replace('_analysis_output', '')
                df['source_dataset'] = dataset_name
                candidate_names.append(dataset_name)

                all_candidate_embeddings.append(embeddings)
                all_candidate_dfs.append(df)

        # Combine all candidates
        self.candidate_embeddings = np.vstack(all_candidate_embeddings).astype(np.float32)
        self.candidate_df = pd.concat(all_candidate_dfs, ignore_index=True)
        self.candidates_name = "_".join(candidate_names)

        # L2 normalize candidate embeddings
        self.candidate_embeddings = self._l2_normalize(self.candidate_embeddings)

        print(f"Loaded {len(self.candidate_embeddings)} candidate samples from {len(candidate_paths)} datasets")

    def _l2_normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize embeddings for cosine similarity computation"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return embeddings / norms

    def compute_quality_scores(self) -> None:
        """
        Compute quality scores based on cosine distance to reference set
        Includes leak detection for highly similar samples
        """
        start_time = time.time()
        print("Computing quality scores based on cosine similarity to reference set...")

        # Compute cosine similarities in batches to manage memory
        batch_size = 1000
        n_candidates = len(self.candidate_embeddings)
        min_distances = np.full(n_candidates, np.inf, dtype=np.float32)
        nearest_ref_indices = np.zeros(n_candidates, dtype=int)

        for i in range(0, n_candidates, batch_size):
            end_idx = min(i + batch_size, n_candidates)
            batch_embeddings = self.candidate_embeddings[i:end_idx]

            # Compute cosine similarities with all reference points
            # cosine_similarity returns similarity, we want distance
            similarities = cosine_similarity(batch_embeddings, self.reference_embeddings)
            distances = 1 - similarities  # Convert to distances

            # Find minimum distance and corresponding reference index for each candidate
            batch_min_distances = np.min(distances, axis=1)
            batch_nearest_indices = np.argmin(distances, axis=1)

            min_distances[i:end_idx] = batch_min_distances
            nearest_ref_indices[i:end_idx] = batch_nearest_indices

            if (i // batch_size + 1) % 10 == 0:
                print(f"  Processed {end_idx}/{n_candidates} candidates...")

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

    def estimate_rbf_gamma(self) -> float:
        """
        Estimate RBF gamma parameter from median nearest-neighbor distance
        """
        print("Estimating RBF gamma from median nearest-neighbor distance...")

        # Use random subsample for efficiency
        n_subsample = min(2048, len(self.candidate_embeddings))
        indices = self.rng.choice(len(self.candidate_embeddings), n_subsample, replace=False)
        subsample = self.candidate_embeddings[indices]

        # Compute pairwise distances
        similarities = cosine_similarity(subsample, subsample)
        distances = 1 - similarities

        # Find nearest neighbor distances (excluding self)
        np.fill_diagonal(distances, np.inf)
        nn_distances = np.min(distances, axis=1)

        # Estimate length scale from median
        median_nn_distance = np.median(nn_distances)
        estimated_gamma = 1.0 / (2 * median_nn_distance**2)

        print(f"Estimated gamma = {estimated_gamma:.6f} (median NN distance = {median_nn_distance:.4f})")
        return estimated_gamma

    def generate_rff_features(self) -> np.ndarray:
        """
        Generate Random Fourier Features for RBF kernel approximation

        Returns:
            RFF feature matrix Φ(X) of shape (n_candidates, rff_dim)
        """
        start_time = time.time()
        print(f"Generating {self.rff_dim}-dimensional Random Fourier Features...")

        if self.sim_gamma is None:
            self.sim_gamma = self.estimate_rbf_gamma()

        # Generate random frequencies
        embed_dim = self.candidate_embeddings.shape[1]
        W = self.rng.normal(0, np.sqrt(2 * self.sim_gamma), (embed_dim, self.rff_dim))
        b = self.rng.uniform(0, 2 * np.pi, self.rff_dim)

        # Compute features in batches to manage memory
        batch_size = 1000
        n_candidates = len(self.candidate_embeddings)
        rff_features = np.zeros((n_candidates, self.rff_dim), dtype=np.float32)

        for i in range(0, n_candidates, batch_size):
            end_idx = min(i + batch_size, n_candidates)
            batch_embeddings = self.candidate_embeddings[i:end_idx]

            # Φ(x) = √(2/D) * cos(W^T x + b)
            projections = batch_embeddings @ W + b
            batch_features = np.sqrt(2.0 / self.rff_dim) * np.cos(projections)
            rff_features[i:end_idx] = batch_features.astype(np.float32)

        self.timings['rff_generation'] = time.time() - start_time
        print(f"Generated RFF features in {self.timings['rff_generation']:.2f}s")
        return rff_features

    def build_low_rank_matrix(self, rff_features: np.ndarray) -> np.ndarray:
        """
        Build low-rank matrix B where B_i = sqrt(q_i) * Φ(x_i)

        Args:
            rff_features: RFF feature matrix of shape (n_candidates, rff_dim)

        Returns:
            Low-rank matrix B of shape (n_candidates, rff_dim)
        """
        print("Building low-rank matrix B = Q^(1/2) * Φ...")

        # Multiply by square root of quality scores
        sqrt_quality = np.sqrt(self.quality_scores).reshape(-1, 1)
        B = sqrt_quality * rff_features

        # Clip norms for numerical stability
        norms = np.linalg.norm(B, axis=1)
        max_norm = np.percentile(norms, 95)  # Use 95th percentile as max
        clip_mask = norms > max_norm
        if np.any(clip_mask):
            B[clip_mask] = B[clip_mask] / norms[clip_mask, np.newaxis] * max_norm
            print(f"Clipped {np.sum(clip_mask)} vectors to max norm {max_norm:.4f}")

        return B.astype(np.float32)

    def greedy_map_selection(self, B: np.ndarray) -> List[int]:
        """
        Greedy MAP k-DPP selection using incremental orthogonalization

        Args:
            B: Low-rank matrix of shape (n_candidates, rff_dim)

        Returns:
            List of selected indices
        """
        start_time = time.time()
        print(f"Starting greedy MAP k-DPP selection for k={self.k}...")

        n_candidates, rff_dim = B.shape
        selected_indices = []

        # Initialize residual norms
        residual_norms = np.sum(B**2, axis=1).astype(np.float32)

        # Exclude leaked samples
        valid_mask = ~self.leak_mask
        residual_norms[self.leak_mask] = -1  # Mark as invalid

        # Keep track of orthogonalized vectors
        orthogonal_vectors = []

        for iteration in range(self.k):
            # Find candidate with maximum residual norm among valid candidates
            valid_residuals = residual_norms.copy()
            valid_residuals[~valid_mask] = -1

            if np.max(valid_residuals) <= 0:
                print(f"Warning: No more valid candidates. Selected {len(selected_indices)} samples.")
                break

            best_idx = np.argmax(valid_residuals)
            selected_indices.append(best_idx)

            # Get current vector and orthogonalize against previous selections
            current_vector = B[best_idx].copy()

            # Orthogonalize against all previous orthogonal vectors
            for orth_vec in orthogonal_vectors:
                projection = np.dot(current_vector, orth_vec)
                current_vector -= projection * orth_vec

            # Normalize to get new orthogonal vector
            current_norm = np.linalg.norm(current_vector)
            if current_norm > 1e-8:
                current_vector /= current_norm
                orthogonal_vectors.append(current_vector.copy())

                # Update residual norms by subtracting projections
                projections = B @ current_vector
                residual_norms -= projections**2

                # Ensure non-negative (numerical stability)
                residual_norms = np.maximum(residual_norms, 0)

            # Mark selected index as invalid for future selections
            valid_mask[best_idx] = False

            # Log progress
            if (iteration + 1) % max(1, self.k // 10) == 0 or iteration < 5:
                gain = valid_residuals[best_idx]
                median_r = np.median(residual_norms[residual_norms >= 0])
                print(f"  t={iteration+1:03d} sel={best_idx:04d} gain={gain:.3f} median_r={median_r:.3f}")

        self.timings['map_selection'] = time.time() - start_time
        print(f"Completed MAP selection in {self.timings['map_selection']:.2f}s")

        return selected_indices

    def run_sampling(self) -> List[int]:
        """
        Run the complete k-DPP sampling pipeline

        Returns:
            List of selected sample indices
        """
        print("=" * 60)
        print("Starting k-DPP Sampling Pipeline")
        print("=" * 60)
        print("Parameters:")
        print(f"  k={self.k}, alpha={self.alpha}, rff_dim={self.rff_dim}")
        print(f"  sim_gamma={self.sim_gamma}, sigma_percentile={self.sigma_percentile}")
        print(f"  seed={self.seed}, dry_run={self.dry_run}")
        print()

        # Step 1: Compute quality scores
        self.compute_quality_scores()

        # Step 2: Generate RFF features
        rff_features = self.generate_rff_features()

        # Step 3: Build low-rank matrix
        B = self.build_low_rank_matrix(rff_features)

        # Step 4: Greedy MAP selection
        selected_indices = self.greedy_map_selection(B)

        self.selected_indices = selected_indices
        print(f"\nSampling completed! Selected {len(selected_indices)} samples.")

        return selected_indices

    def save_selected_csv(self, output_path: str) -> None:
        """Save selected samples to CSV with required columns"""
        if self.selected_indices is None:
            raise ValueError("No samples selected. Run sampling first.")

        print(f"Saving selected samples to {output_path}...")

        # Prepare output data
        output_data = []
        for idx in self.selected_indices:
            row = self.candidate_df.iloc[idx].copy()

            # Add nearest reference information
            nearest_ref_idx = self.nearest_ref_indices[idx]
            nearest_ref_question = self.reference_df.iloc[nearest_ref_idx].get('question', 'N/A')
            nearest_cosine_sim = 1 - self.distances[idx]  # Convert distance back to similarity

            output_row = {
                'id': row.get('id', idx),
                'question': row.get('question', 'N/A'),
                'rationale': row.get('rationale', 'N/A'),
                'answer': row.get('answer', 'N/A'),
                'dataset_name': row.get('source_dataset', 'unknown'),
                'raw_subject': row.get('raw_subject', row.get('category', 'N/A')),
                'nearest_bench_question': nearest_ref_question,
                'nearest_bench_cosine_similarity': f"{nearest_cosine_sim:.4f}",
            }
            output_data.append(output_row)

        # Save to CSV
        output_df = pd.DataFrame(output_data)
        output_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Saved {len(output_df)} selected samples to CSV")

    def create_umap_visualization(
        self, output_path: str, color_by: str = 'dataset_type', use_full_reference: bool = True
    ) -> None:
        """
        Create UMAP visualization comparing reference and selected candidates

        Args:
            output_path: Path to save the HTML visualization
            color_by: Column to color points by ('dataset_type', 'category', 'subject', 'raw_subject')
            use_full_reference: If True, use full reference data instead of filtered reference data
        """
        if self.selected_indices is None:
            raise ValueError("No samples selected. Run sampling first.")

        print(f"Creating UMAP visualization at {output_path}...")
        print(f"Coloring by: {color_by}")
        print(f"Using {'full' if use_full_reference else 'filtered'} reference data")

        try:
            # Import UMAP visualization function
            import sys

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
            # ref_df_viz['dataset_type'] = 'Reference'

            selected_df = self.candidate_df.iloc[self.selected_indices].copy()
            # selected_df['dataset_type'] = 'Selected'

            combined_df = pd.concat([ref_df_viz, selected_df], ignore_index=True)

            # Create UMAP explorer
            explorer = InteractiveUMAPExplorer(combined_embeddings, combined_df)

            # Generate UMAP projection
            explorer.compute_umap_projection(n_neighbors=15, min_dist=0.1)

            # Validate color_by column exists
            if color_by not in combined_df.columns:
                print(f"Warning: Column '{color_by}' not found. Available columns: {list(combined_df.columns)}")
                print("Using 'dataset_type' as fallback.")
                color_by = 'dataset_type'

            # Create interactive visualization
            title_suffix = f"colored by {color_by}" if color_by != 'dataset_type' else ""
            ref_type = "full reference" if use_full_reference else "filtered reference"
            title = f"UMAP: {ref_type} vs Selected Samples ({self.candidates_name}) {title_suffix}".strip()

            explorer.create_interactive_umap(color_by=color_by, title=title, save_path=output_path, show_plot=False)

            print(f"UMAP visualization saved to {output_path}")

            # Print coloring statistics
            if color_by in combined_df.columns:
                value_counts = combined_df[color_by].value_counts()
                print(f"Color distribution ({color_by}): {dict(value_counts)}")

        except Exception as e:
            print(f"Warning: Could not create UMAP visualization: {e}")

    def save_parameters_json(self, output_path: str) -> None:
        """Save parameters and statistics to JSON file"""
        print(f"Saving parameters and statistics to {output_path}...")

        # Compile statistics
        stats = {
            'parameters': {
                'k': self.k,
                'alpha': self.alpha,
                'sim_gamma': float(self.sim_gamma) if self.sim_gamma else None,
                'rff_dim': self.rff_dim,
                'sigma_percentile': self.sigma_percentile,
                'seed': self.seed,
                'dry_run': self.dry_run,
                'reference_category_filter': self.reference_category_filter,
                'reference_subject_filter': self.reference_subject_filter,
            },
            'counts': {
                'reference_total': len(self.reference_embeddings),
                'candidates_total': len(self.candidate_embeddings),
                'selected': len(self.selected_indices) if self.selected_indices else 0,
                'leaks_excluded': int(np.sum(self.leak_mask)) if self.leak_mask is not None else 0,
            },
            'distance_statistics': {
                'min': float(np.min(self.distances)),
                'p10': float(np.percentile(self.distances, 10)),
                'median': float(np.median(self.distances)),
                'p90': float(np.percentile(self.distances, 90)),
                'max': float(np.max(self.distances)),
            },
            'timings': self.timings,
            'candidates_name': self.candidates_name,
        }

        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"Parameters and statistics saved to {output_path}")


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="k-DPP sampling for HLE dataset analysis", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--k', type=int, default=100, help='Number of samples to select')
    parser.add_argument('--alpha', type=float, default=1.0, help='Power of quality score')
    parser.add_argument(
        '--sim_gamma', type=float, default=None, help='RBF kernel gamma parameter (auto-estimated if None)'
    )
    parser.add_argument('--rff_dim', type=int, default=512, help='Dimension of Random Fourier Features')
    parser.add_argument(
        '--sigma_percentile', type=float, default=50.0, help='Percentile of distances for quality scoring sigma'
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--dry_run', action='store_true', help='Run with small parameters for testing')
    parser.add_argument(
        '--reference_category_filter',
        type=str,
        default=None,
        help='Filter reference data by category (case-insensitive substring match)',
    )
    parser.add_argument(
        '--reference_subject_filter',
        type=str,
        default=None,
        help='Filter reference data by subject (case-insensitive substring match)',
    )
    parser.add_argument(
        '--umap_color_by',
        type=str,
        default='dataset_type',
        choices=['dataset_type', 'category', 'subject', 'raw_subject'],
        help='Column to color UMAP visualization by',
    )

    args = parser.parse_args()

    # Initialize sampler
    sampler = HLESimilarDataSampler(
        k=args.k,
        alpha=args.alpha,
        sim_gamma=args.sim_gamma,
        rff_dim=args.rff_dim,
        sigma_percentile=args.sigma_percentile,
        seed=args.seed,
        dry_run=args.dry_run,
        reference_category_filter=args.reference_category_filter,
        reference_subject_filter=args.reference_subject_filter,
    )

    # Load data
    sampler.load_embeddings_data()

    # Run sampling
    selected_indices = sampler.run_sampling()

    # Create output directory
    output_dir = Path("data_sampling_results")
    output_dir.mkdir(exist_ok=True)

    # Save outputs
    candidates_name = sampler.candidates_name

    # Save CSV
    csv_path = output_dir / f"selected_{candidates_name}.csv"
    sampler.save_selected_csv(str(csv_path))

    # Create UMAP visualization
    umap_path = output_dir / f"umap_selected_{candidates_name}.html"
    sampler.create_umap_visualization(str(umap_path), color_by=args.umap_color_by)

    # Save parameters
    params_path = output_dir / f"params_{candidates_name}.json"
    sampler.save_parameters_json(str(params_path))

    print("\n" + "=" * 60)
    print("Sampling completed successfully!")
    print(f"Selected {len(selected_indices)} samples from {candidates_name}")
    print("Output files:")
    print(f"  CSV: {csv_path}")
    print(f"  UMAP: {umap_path}")
    print(f"  Parameters: {params_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
