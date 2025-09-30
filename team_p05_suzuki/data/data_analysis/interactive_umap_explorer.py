#!/usr/bin/env python3
"""
Interactive UMAP Explorer for HLE Dataset Analysis
=================================================

This module provides interactive visualization tools for exploring UMAP clusters,
allowing users to click on data points, hover for information, and extract specific clusters.

Features:
- Interactive UMAP plots with hover information
- Click-to-select data points
- Cluster extraction and analysis
- Export functionality for selected data
- Multiple visualization modes (category, cluster, dataset)
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import umap
import yaml
from sklearn.cluster import DBSCAN, KMeans


class InteractiveUMAPExplorer:
    """Interactive UMAP visualization and cluster exploration tool"""

    def __init__(self, embeddings_matrix=None, dataframe=None):
        """
        Initialize the explorer with embeddings and metadata

        Args:
            embeddings_matrix: numpy array of embeddings
            dataframe: pandas DataFrame with metadata
        """
        self.embedding_matrix = embeddings_matrix
        self.df = dataframe
        self.umap_2d = None
        self.selected_indices = set()
        self.output_dir = self._setup_output_directory()

    def _setup_output_directory(self):
        """Create umap_results output directory"""
        output_dir = Path("umap_results")
        output_dir.mkdir(exist_ok=True)
        return output_dir

    def _load_config(self, config_file):
        """
        Load configuration from YAML file

        Args:
            config_file: Path to YAML configuration file

        Returns:
            Dictionary containing configuration
        """
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # Validate required sections
            required_sections = ['datasets', 'filters']
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Configuration file missing required section: {section}")

            return config

        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration file: {e}")

    def _apply_filters_from_config(self, config):
        """
        Apply filters from configuration to the explorer instance

        Args:
            config: Configuration dictionary
        """
        filters = config.get('filters', {})

        # Apply include filters
        if 'includes' in filters and filters['includes']:
            self._filter_includes = filters['includes']
            print(f"Applied include filters: {self._filter_includes}")

        # Apply exclude filters
        if 'excludes' in filters and filters['excludes']:
            self._filter_excludes = filters['excludes']
            print(f"Applied exclude filters: {self._filter_excludes}")

        # Store image filtering preference
        self._exclude_images = filters.get('exclude_images', True)
        if self._exclude_images:
            print("Image filtering enabled: will exclude data with has_image=True")

    def _apply_data_filters(self):
        """
        Apply all configured filters to the loaded data
        """
        # Filter out data where has_image flag is True
        if hasattr(self, '_exclude_images') and self._exclude_images and 'has_image' in self.df.columns:
            print("Filtering out data with has_image flag set to True...")
            original_length = len(self.df)

            # Create mask for rows where has_image is NOT True (False or NaN)
            mask = self.df['has_image'] != True  # noqa: E712
            filtered_indices = self.df[mask].index.tolist()

            # Filter both dataframe and embedding matrix
            self.df = self.df[mask].reset_index(drop=True)
            self.embedding_matrix = self.embedding_matrix[filtered_indices]

            filtered_length = len(self.df)
            print(
                f"Filtered out images: from {original_length} to {filtered_length} samples ({filtered_length/original_length*100:.1f}% kept)"
            )

        # Filter data by category if includes parameter is provided
        if hasattr(self, '_filter_includes') and self._filter_includes:
            print(f"Filtering data by categories starting with: {self._filter_includes}")
            original_length = len(self.df)

            # Create mask for rows where category starts with any of the included strings
            mask = self.df['category'].str.startswith(tuple(self._filter_includes), na=False)
            filtered_indices = self.df[mask].index.tolist()

            # Filter both dataframe and embedding matrix
            self.df = self.df[mask].reset_index(drop=True)
            self.embedding_matrix = self.embedding_matrix[filtered_indices]

            filtered_length = len(self.df)
            print(
                f"Filtered from {original_length} to {filtered_length} samples ({filtered_length/original_length*100:.1f}% kept)"
            )

        # Filter data by category if excludes parameter is provided
        if hasattr(self, '_filter_excludes') and self._filter_excludes:
            print(f"Excluding data by categories starting with: {self._filter_excludes}")
            original_length = len(self.df)

            # Create mask for rows where category does NOT start with any of the excluded strings
            mask = ~self.df['category'].str.startswith(tuple(self._filter_excludes), na=False)
            filtered_indices = self.df[mask].index.tolist()

            # Filter both dataframe and embedding matrix
            self.df = self.df[mask].reset_index(drop=True)
            self.embedding_matrix = self.embedding_matrix[filtered_indices]

            filtered_length = len(self.df)
            print(
                f"Filtered from {original_length} to {filtered_length} samples ({filtered_length/original_length*100:.1f}% kept)"
            )

    def get_output_path(self, filename):
        """Get path for output file within umap_results directory"""
        return self.output_dir / filename

    def _ensure_question_is_string(self, question_field):
        """
        Helper method to ensure question field is a string, handling cases where it might be a list

        Args:
            question_field: The question field value (could be string, list, or other)

        Returns:
            String representation of the question
        """
        if isinstance(question_field, list):
            # If question is a list, join it into a string
            return ' '.join(str(item) for item in question_field)
        else:
            # If question is already a string or other type, convert to string
            return str(question_field)

    def _darken_color(self, color):
        """
        Darken a color by reducing its lightness
        """
        import re

        # Handle hex colors
        if color.startswith('#'):
            # Convert hex to RGB
            hex_color = color.lstrip('#')
            rgb = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
            # Darken by reducing each component by 30%
            darkened_rgb = tuple(max(0, int(c * 0.7)) for c in rgb)
            return f"rgb({darkened_rgb[0]}, {darkened_rgb[1]}, {darkened_rgb[2]})"

        # Handle named colors (convert to darker equivalents)
        color_map = {
            'lightblue': 'blue',
            'lightgreen': 'green',
            'lightcoral': 'red',
            'lightyellow': 'orange',
            'lightpink': 'hotpink',
            'lightgray': 'gray',
            'lightcyan': 'cyan',
            'lightsteelblue': 'steelblue',
        }

        # For plotly colors, try to darken
        if color.lower() in color_map:
            return color_map[color.lower()]

        # For RGB format
        if color.startswith('rgb'):
            # Extract RGB values
            rgb_match = re.findall(r'\d+', color)
            if len(rgb_match) >= 3:
                r, g, b = int(rgb_match[0]), int(rgb_match[1]), int(rgb_match[2])
                # Darken by reducing each component by 30%
                r_dark = max(0, int(r * 0.7))
                g_dark = max(0, int(g * 0.7))
                b_dark = max(0, int(b * 0.7))
                return f"rgb({r_dark}, {g_dark}, {b_dark})"

        # If we can't darken it, return the original color
        return color

    def load_from_combined_analysis(
        self, combined_csv_path="combined_analysis_output/combined_dataset_with_clusters.csv", embeddings_path=None
    ):
        """
        Load data from existing combined analysis results

        Args:
            combined_csv_path: Path to the combined dataset CSV with clusters
            embeddings_path: Optional path to embeddings pickle file
        """
        print("Loading combined analysis data...")

        # Load the CSV with cluster assignments
        self.df = pd.read_csv(combined_csv_path)
        print(f"Loaded {len(self.df)} samples from combined dataset")

        # Try to load embeddings if path provided
        if embeddings_path and Path(embeddings_path).exists():
            with open(embeddings_path, 'rb') as f:
                data = pickle.load(f)
                self.embedding_matrix = data['embeddings']
        else:
            print("Warning: No embeddings provided. Will need to compute UMAP from metadata only.")

        return self.df

    def load_separate_embeddings(self, config_file="dataset_config.yaml", output_dir="analysis_outputs"):
        """
        Load and combine embeddings from multiple pickle files based on YAML configuration

        Args:
            config_file: Path to YAML configuration file. Defaults to "dataset_config.yaml"
            output_dir: Optional directory to load embeddings from. Defaults to "analysis_outputs"
        """
        # Load configuration from YAML file
        config = self._load_config(config_file)

        # Get enabled datasets from configuration
        enabled_datasets = []
        for dataset_name, dataset_config in config['datasets'].items():
            if dataset_config.get('enabled', False):
                enabled_datasets.append(
                    {
                        'name': dataset_name,
                        'path': dataset_config['path'],
                        'description': dataset_config.get('description', ''),
                    }
                )

        if not enabled_datasets:
            raise ValueError("No datasets enabled in configuration file. Please enable at least one dataset.")

        print(f"Loading embeddings from {len(enabled_datasets)} enabled datasets...")
        for dataset in enabled_datasets:
            print(f"  ✓ {dataset['name']}: {dataset['description']}")

        # Apply filters from configuration
        self._apply_filters_from_config(config)

        # Extract paths for enabled datasets
        embeddings_paths = [dataset['path'] for dataset in enabled_datasets]
        dataset_names = [dataset['name'] for dataset in enabled_datasets]

        print(f"Loading embeddings from {len(embeddings_paths)} files...")

        all_embeddings = []
        all_dataframes = []

        for i, path in enumerate(embeddings_paths):
            print(f"Loading embeddings from {path}...")
            path = output_dir + "/" + path

            with open(path, 'rb') as f:
                data = pickle.load(f)
                embeddings = data['embeddings']
                df = pd.DataFrame(data['metadata'])

                # Add dataset identifier
                if dataset_names and i < len(dataset_names):
                    dataset_name = dataset_names[i]
                else:
                    # Use filename without extension as dataset name
                    dataset_name = Path(path).stem.replace('_embeddings', '').upper()

                df['dataset'] = dataset_name

                all_embeddings.append(embeddings)
                all_dataframes.append(df)

                print(f"Loaded {len(df)} samples from {dataset_name} dataset")

        # Combine all data
        self.df = pd.concat(all_dataframes, ignore_index=True)
        self.embedding_matrix = np.vstack(all_embeddings)

        # Apply configuration-based filtering
        self._apply_data_filters()

        # Update category names for GPQASUPER dataset to include subject
        if hasattr(self, 'df') and 'category' in self.df.columns:
            print("Updating SUPERGPQA category names to include subject...")

            # Find rows that are from GPQASUPER dataset
            supergpqa_mask = self.df['category'].str.startswith('SuperGPQA', na=False)
            supergpqa_rows = self.df[supergpqa_mask]

            if len(supergpqa_rows) > 0:
                print(f"Found {len(supergpqa_rows)} SUPERGPQA samples to update")

                # Update category names for SUPERGPQA rows
                for idx in supergpqa_rows.index:
                    current_category = self.df.loc[idx, 'category']

                    # Extract subject from appropriate column
                    if 'raw_subject' in self.df.columns and pd.notna(self.df.loc[idx, 'raw_subject']):
                        subject = self.df.loc[idx, 'raw_subject']
                    else:
                        subject = 'Unknown'

                    # Update category to include subject
                    new_category = f"{current_category}_{subject}"
                    self.df.loc[idx, 'category'] = new_category

                print(f"Updated category names for {len(supergpqa_rows)} SUPERGPQA samples")

        total_samples = sum(len(df) for df in all_dataframes)
        print(f"Combined {total_samples} total samples from {len(embeddings_paths)} datasets")

        # Add cluster assignments if not present
        if 'cluster' not in self.df.columns:
            self.add_cluster_assignments()

        return self.df

    def add_cluster_assignments(self, n_clusters=None, method=None):
        """
        Add cluster assignments to the dataframe

        Args:
            n_clusters: Number of clusters for KMeans (uses config if None)
            method: 'kmeans' or 'dbscan' (uses config if None)
        """
        if self.embedding_matrix is None:
            raise ValueError("No embeddings available for clustering")

        # Load configuration if parameters not provided
        if n_clusters is None or method is None:
            try:
                config = self._load_config("dataset_config.yaml")
                clustering_config = config.get('clustering', {})
                n_clusters = n_clusters or clustering_config.get('n_clusters', 8)
                method = method or clustering_config.get('method', 'kmeans')
            except Exception as e:
                print(f"Warning: Could not load clustering config, using defaults: {e}")
                n_clusters = n_clusters or 8
                method = method or 'kmeans'

        print(f"Adding cluster assignments using {method}...")

        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = clusterer.fit_predict(self.embedding_matrix)
        elif method == 'dbscan':
            # Load DBSCAN parameters from config if available
            try:
                config = self._load_config("dataset_config.yaml")
                clustering_config = config.get('clustering', {})
                eps = clustering_config.get('eps', 0.5)
                min_samples = clustering_config.get('min_samples', 5)
            except Exception:
                eps, min_samples = 0.5, 5

            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = clusterer.fit_predict(self.embedding_matrix)
        else:
            raise ValueError("Method must be 'kmeans' or 'dbscan'")

        self.df['cluster'] = cluster_labels
        print(f"Assigned {len(set(cluster_labels))} clusters")

    def compute_umap_projection(self, n_neighbors=None, min_dist=None, random_state=None):
        """
        Compute UMAP 2D projection from embeddings

        Args:
            n_neighbors: UMAP n_neighbors parameter (uses config if None)
            min_dist: UMAP min_dist parameter (uses config if None)
            random_state: Random state for reproducibility (uses config if None)
        """
        if self.embedding_matrix is None:
            raise ValueError("No embeddings available for UMAP projection")

        # Load configuration if parameters not provided
        if n_neighbors is None or min_dist is None or random_state is None:
            try:
                config = self._load_config("dataset_config.yaml")
                umap_config = config.get('umap', {})
                n_neighbors = n_neighbors or umap_config.get('n_neighbors', 15)
                min_dist = min_dist or umap_config.get('min_dist', 0.1)
                random_state = random_state or umap_config.get('random_state', 42)
            except Exception as e:
                print(f"Warning: Could not load UMAP config, using defaults: {e}")
                n_neighbors = n_neighbors or 15
                min_dist = min_dist or 0.1
                random_state = random_state or 42

        print("Computing UMAP 2D projection...")
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
        self.umap_2d = reducer.fit_transform(self.embedding_matrix)

        # Add UMAP coordinates to dataframe
        self.df['umap_x'] = self.umap_2d[:, 0]
        self.df['umap_y'] = self.umap_2d[:, 1]

        print("UMAP projection completed")
        return self.umap_2d

    def create_interactive_umap(
        self, color_by='category', size_by=None, title="Interactive UMAP Visualization", save_path=None, show_plot=True
    ):
        """
        Create interactive UMAP visualization with hover information and click selection

        Args:
            color_by: Column to color points by
            size_by: Column to size points by (optional)
            title: Plot title
            save_path: Path to save HTML file
            show_plot: Whether to display the plot
        """
        if self.umap_2d is None:
            self.compute_umap_projection()

        # Prepare hover text with detailed information
        hover_text = []
        for idx, row in self.df.iterrows():
            # Handle case where question might be a list or string
            question_full = self._ensure_question_is_string(row['question'])

            # Show at least 400 chars of the question, breaking lines every 100 chars
            question_text = question_full[:400]

            # Add line breaks every 100 characters
            formatted_question = ""
            for i in range(0, len(question_text), 100):
                formatted_question += question_text[i : i + 100]
                if i + 100 < len(question_text):
                    formatted_question += "<br>"

            # Add ellipsis if the original question was longer than 400 chars
            if len(question_full) > 400:
                formatted_question += "..."

            hover_info = f"""
            <b>ID:</b> {row['id']}<br>
            <b>Dataset:</b> {row.get('dataset', 'Unknown')}<br>
            <b>Category:</b> {row['category']}<br>
            <b>Subject:</b> {row['raw_subject']}<br>
            <b>Cluster:</b> {row.get('cluster', 'N/A')}<br>
            <b>Question Length:</b> {row.get('question_length', 'N/A')}<br>
            <b>Question:</b> {formatted_question}<br>
            <b>Index:</b> {idx}
            """
            hover_text.append(hover_info)

        # Create the scatter plot
        fig = go.Figure()

        # Handle different coloring schemes
        if color_by in self.df.columns:
            unique_values = sorted(self.df[color_by].unique())
            colors = px.colors.qualitative.Set3
            if len(unique_values) > len(colors):
                colors = px.colors.qualitative.Light24

            for i, value in enumerate(unique_values):
                mask = self.df[color_by] == value
                indices = self.df.index[mask].tolist()

                # Determine marker symbols and colors based on dataset for each data point
                marker_symbols = []
                marker_colors = []
                base_color = colors[i % len(colors)]

                if 'dataset' in self.df.columns:
                    for idx in indices:
                        dataset_name = self.df.loc[idx, 'dataset']
                        if str(dataset_name).startswith('HLE'):
                            marker_symbols.append('circle')
                            # Make HLE points darker by reducing the lightness
                            marker_colors.append(self._darken_color(base_color))
                        else:
                            marker_symbols.append('triangle-up')
                            marker_colors.append(base_color)
                else:
                    marker_symbols = ['triangle-up'] * len(indices)
                    marker_colors = [base_color] * len(indices)

                # Size handling
                size_array = None
                if size_by and size_by in self.df.columns:
                    size_array = self.df.loc[mask, size_by]
                    # Normalize sizes to reasonable range
                    size_array = 5 + (size_array - size_array.min()) / (size_array.max() - size_array.min()) * 15

                fig.add_trace(
                    go.Scatter(
                        x=self.umap_2d[mask, 0],
                        y=self.umap_2d[mask, 1],
                        mode='markers',
                        name=str(value),
                        text=[hover_text[i] for i in indices],
                        hovertemplate='%{text}<extra></extra>',
                        marker=dict(
                            color=marker_colors,  # Individual colors per point
                            size=size_array if size_array is not None else 8,
                            symbol=marker_symbols,  # Individual symbols per point
                            opacity=0.7,
                            line=dict(width=0.5, color='DarkSlateGrey'),
                        ),
                        customdata=indices,  # Store indices for selection
                    )
                )
        else:
            # Default visualization
            fig.add_trace(
                go.Scatter(
                    x=self.umap_2d[:, 0],
                    y=self.umap_2d[:, 1],
                    mode='markers',
                    text=hover_text,
                    hovertemplate='%{text}<extra></extra>',
                    marker=dict(size=8, opacity=0.7),
                    customdata=list(range(len(self.df))),
                )
            )

        # Add invisible traces for dataset type legend
        if 'dataset' in self.df.columns:
            # Check if we have HLE datasets
            has_hle = any(str(dataset).startswith('HLE') for dataset in self.df['dataset'])
            has_non_hle = any(not str(dataset).startswith('HLE') for dataset in self.df['dataset'])

            if has_hle:
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode='markers',
                        marker=dict(size=12, symbol='circle', color='gray'),
                        name='HLE Dataset',
                        showlegend=True,
                        legendgroup='dataset_type',
                    )
                )

            if has_non_hle:
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode='markers',
                        marker=dict(size=12, symbol='triangle-up', color='gray'),
                        name='SFT_SEED Dataset',
                        showlegend=True,
                        legendgroup='dataset_type',
                    )
                )

        # Load visualization settings from config
        try:
            config = self._load_config("dataset_config.yaml")
            viz_config = config.get('visualization', {})
            plot_width = viz_config.get('plot_width', 1500)
            plot_height = viz_config.get('plot_height', 1000)
        except Exception:
            plot_width, plot_height = 1500, 1000

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            width=plot_width,
            height=plot_height,
            hovermode='closest',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01),
        )

        # Save if requested
        if save_path:
            fig.write_html(save_path)
            print(f"Interactive plot saved to: {save_path}")

        if show_plot:
            fig.show()

        return fig

    def create_cluster_comparison_plot(self, save_path=None):
        """
        Create individual HTML files for different clustering approaches
        """
        if self.umap_2d is None:
            self.compute_umap_projection()

        # Define the different visualization configurations
        viz_configs = [
            {
                'color_by': 'category',
                'colors': px.colors.qualitative.Set1,
                'title': 'UMAP Visualization by Category',
                'filename': 'umap_by_category.html',
            },
            {
                'color_by': 'dataset',
                'colors': px.colors.qualitative.Set2,
                'title': 'UMAP Visualization by Dataset',
                'filename': 'umap_by_dataset.html',
            },
            {
                'color_by': 'cluster',
                'colors': px.colors.qualitative.Set3,
                'title': 'UMAP Visualization by Cluster',
                'filename': 'umap_by_cluster.html',
            },
            {
                'color_by': 'question_length',
                'colors': px.colors.sequential.Viridis,
                'title': 'UMAP Visualization by Question Length',
                'filename': 'umap_by_question_length.html',
            },
        ]

        # Create individual plots for each visualization
        for config in viz_configs:
            fig = go.Figure()

            if config['color_by'] == 'question_length':
                # Continuous color scale for question length
                fig.add_trace(
                    go.Scatter(
                        x=self.umap_2d[:, 0],
                        y=self.umap_2d[:, 1],
                        mode='markers',
                        marker=dict(
                            color=self.df['question_length'],
                            colorscale='Viridis',
                            size=8,
                            opacity=0.7,
                            colorbar=dict(title="Question Length"),
                            line=dict(width=0.5, color='DarkSlateGrey'),
                        ),
                        hovertemplate='Question Length: %{marker.color}<br>UMAP 1: %{x}<br>UMAP 2: %{y}<extra></extra>',
                    )
                )
            else:
                # Discrete color mapping
                if config['color_by'] in self.df.columns:
                    unique_values = sorted(self.df[config['color_by']].unique())
                    colors = config['colors']

                    for i, value in enumerate(unique_values):
                        mask = self.df[config['color_by']] == value
                        fig.add_trace(
                            go.Scatter(
                                x=self.umap_2d[mask, 0],
                                y=self.umap_2d[mask, 1],
                                mode='markers',
                                name=str(value),
                                marker=dict(
                                    color=colors[i % len(colors)],
                                    size=8,
                                    opacity=0.7,
                                    line=dict(width=0.5, color='DarkSlateGrey'),
                                ),
                                hovertemplate=f'{config["color_by"].title()}: {value}<br>UMAP 1: %{{x}}<br>UMAP 2: %{{y}}<extra></extra>',
                            )
                        )

            # Update layout for each individual plot
            fig.update_layout(
                title=config['title'],
                xaxis_title="UMAP 1",
                yaxis_title="UMAP 2",
                width=1200,
                height=800,
                hovermode='closest',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01),
            )

            # Save individual HTML file
            if save_path:
                # Extract directory from save_path if provided
                if hasattr(save_path, 'parent'):
                    output_dir = save_path.parent
                else:
                    output_dir = Path(save_path).parent

                individual_save_path = output_dir / config['filename']
            else:
                individual_save_path = self.output_dir / config['filename']

            fig.write_html(individual_save_path)
            print(f"Individual plot saved to: {individual_save_path}")

        print("All individual comparison plots created successfully!")
        return None

    def list_available_datasets(self, config_file="dataset_config.yaml"):
        """
        List all available datasets from configuration file

        Args:
            config_file: Path to YAML configuration file

        Returns:
            Dictionary with dataset information
        """
        try:
            config = self._load_config(config_file)
            datasets_info = {}

            for dataset_name, dataset_config in config['datasets'].items():
                datasets_info[dataset_name] = {
                    'enabled': dataset_config.get('enabled', False),
                    'description': dataset_config.get('description', ''),
                    'path': dataset_config.get('path', ''),
                }

            return datasets_info

        except Exception as e:
            print(f"Error loading dataset configuration: {e}")
            return {}

    def extract_cluster_data(self, cluster_id, output_path=None):
        """
        Extract data points from a specific cluster

        Args:
            cluster_id: ID of the cluster to extract
            output_path: Path to save extracted data (optional)

        Returns:
            DataFrame with cluster data
        """
        if 'cluster' not in self.df.columns:
            raise ValueError("No cluster assignments available")

        cluster_data = self.df[self.df['cluster'] == cluster_id].copy()

        print(f"Cluster {cluster_id} contains {len(cluster_data)} data points")
        print(f"Categories: {cluster_data['category'].value_counts().to_dict()}")
        print(f"Datasets: {cluster_data['dataset'].value_counts().to_dict()}")

        if output_path:
            cluster_data.to_csv(output_path, index=False)
            print(f"Cluster data saved to: {output_path}")

        return cluster_data

    def extract_region_data(self, x_range, y_range, output_path=None):
        """
        Extract data points from a specific UMAP coordinate region

        Args:
            x_range: Tuple of (min_x, max_x)
            y_range: Tuple of (min_y, max_y)
            output_path: Path to save extracted data (optional)

        Returns:
            DataFrame with region data
        """
        if self.umap_2d is None:
            raise ValueError("No UMAP projection available")

        # Find points in the specified region
        mask = (
            (self.df['umap_x'] >= x_range[0])
            & (self.df['umap_x'] <= x_range[1])
            & (self.df['umap_y'] >= y_range[0])
            & (self.df['umap_y'] <= y_range[1])
        )

        region_data = self.df[mask].copy()

        print(f"Region ({x_range}, {y_range}) contains {len(region_data)} data points")
        if len(region_data) > 0:
            print(f"Categories: {region_data['category'].value_counts().to_dict()}")
            print(f"Datasets: {region_data['dataset'].value_counts().to_dict()}")

        if output_path:
            region_data.to_csv(output_path, index=False)
            print(f"Region data saved to: {output_path}")

        return region_data

    def analyze_cluster_characteristics(self, cluster_id):
        """
        Provide detailed analysis of a specific cluster

        Args:
            cluster_id: ID of the cluster to analyze

        Returns:
            Dictionary with cluster analysis
        """
        cluster_data = self.df[self.df['cluster'] == cluster_id]

        if len(cluster_data) == 0:
            return {"error": f"No data found for cluster {cluster_id}"}

        analysis = {
            "cluster_id": cluster_id,
            "size": len(cluster_data),
            "percentage_of_total": len(cluster_data) / len(self.df) * 100,
            "categories": cluster_data['category'].value_counts().to_dict(),
            "datasets": cluster_data['dataset'].value_counts().to_dict(),
            "avg_question_length": cluster_data['question_length'].mean(),
            "avg_rationale_length": cluster_data['rationale_length'].mean(),
            "sample_questions": [self._ensure_question_is_string(q) for q in cluster_data['question'].head(5)],
            "sample_ids": cluster_data['id'].head(10).tolist(),
        }

        # Add statistical comparisons
        analysis["question_length_stats"] = {
            "mean": cluster_data['question_length'].mean(),
            "std": cluster_data['question_length'].std(),
            "median": cluster_data['question_length'].median(),
            "global_mean": self.df['question_length'].mean(),
        }

        return analysis

    def find_interesting_clusters(self, min_size=10):
        """
        Identify potentially interesting clusters based on various criteria

        Args:
            min_size: Minimum cluster size to consider

        Returns:
            List of cluster analyses sorted by "interestingness"
        """
        if 'cluster' not in self.df.columns:
            raise ValueError("No cluster assignments available")

        cluster_analyses = []

        for cluster_id in self.df['cluster'].unique():
            if cluster_id == -1:  # Skip noise cluster from DBSCAN
                continue

            analysis = self.analyze_cluster_characteristics(cluster_id)

            if analysis.get('size', 0) < min_size:
                continue

            # Calculate "interestingness" score
            size_score = min(analysis['size'] / len(self.df), 0.1) * 10  # Normalize size

            # Category diversity score
            categories = analysis['categories']
            category_diversity = len(categories) / len(self.df['category'].unique())

            # Dataset mixing score
            datasets = analysis['datasets']
            if len(datasets) > 1:
                dataset_mixing = min(datasets.values()) / max(datasets.values())
            else:
                dataset_mixing = 0

            # Length anomaly score
            length_stats = analysis['question_length_stats']
            length_diff = abs(length_stats['mean'] - length_stats['global_mean']) / length_stats['global_mean']

            interestingness = size_score + category_diversity + dataset_mixing + length_diff
            analysis['interestingness_score'] = interestingness

            cluster_analyses.append(analysis)

        # Sort by interestingness
        cluster_analyses.sort(key=lambda x: x['interestingness_score'], reverse=True)

        return cluster_analyses


def run_basic_exploration():
    """Run basic interactive exploration with default settings"""
    print("🚀 Starting Basic Interactive UMAP Exploration...")
    from datetime import datetime

    output_dir = Path("umap_results")
    output_dir.mkdir(exist_ok=True)

    explorer = InteractiveUMAPExplorer()

    try:
        print("📊 Loading combined analysis data...")
        explorer.load_separate_embeddings()

        print("🎨 Creating interactive visualizations...")

        # Create main interactive plot
        explorer.create_interactive_umap(
            color_by='category',
            title="🔍 Interactive UMAP - Hover for details, explore clusters!",
            save_path=explorer.get_output_path(
                f"interactive_umap_exploration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            ),
        )
        print("\n🎉 Interactive exploration setup complete!")
        print(f"📂 Files saved to: {explorer.output_dir}")

        # Create comparison view
        print("Creating cluster comparison plots...")
        explorer.create_cluster_comparison_plot(save_path=explorer.output_dir / "umap_cluster_comparison.html")

        return explorer

    except FileNotFoundError as e:
        print(f"❌ Error: Could not find required files - {e}")
        print("💡 Make sure you have run the embedding analysis first.")
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Interactive UMAP Explorer for HLE Dataset")
    parser.add_argument(
        '--config', default='dataset_config.yaml', help='Path to YAML configuration file (default: dataset_config.yaml)'
    )
    parser.add_argument('--list-datasets', action='store_true', help='List all available datasets from configuration')

    args = parser.parse_args()

    print("🎯 Interactive UMAP Explorer for HLE Dataset Analysis")
    print("=" * 55)

    if args.list_datasets:
        # Just list datasets and exit
        explorer = InteractiveUMAPExplorer()
        datasets_info = explorer.list_available_datasets(args.config)
        print(f"\n📊 Available datasets in {args.config}:")
        print("-" * 60)
        for name, info in datasets_info.items():
            status = "✓ Enabled" if info['enabled'] else "✗ Disabled"
            print(f"{status:<12} {name:<25} {info['description']}")
        print(f"\n💡 To enable/disable datasets, edit {args.config}")
        exit(0)

    # Run the basic exploration functionality
    explorer = run_basic_exploration()

    if explorer:
        print("\n🎉 Exploration completed successfully!")
        print("📂 Check the generated HTML files for interactive visualizations!")
    else:
        print("\n❌ Exploration failed. Please check your data files.")
