#!/usr/bin/env python3
"""
Complete SFT Dataset Analysis Pipeline
=====================================

This script provides a comprehensive analysis pipeline for various datasets including:
- Academic benchmark datasets
- OpenMathReasoning dataset (with DeepSeek-R1 CoT filtering)
- OpenScienceReasoning-2 dataset
- And many other supported datasets

It includes data loading, statistical analysis, visualization, and embedding-based analysis.

Usage:
    # Analyze reference dataset (default)
    python sft_complete_analysis.py [--samples 500] [--embeddings] [--visualizations] [--all]

    # Analyze OpenMathReasoning 16K dataset
    python sft_complete_analysis.py --dataset openmathreasoning --samples 1000 --all

        # Analyze standard OpenMathReasoning dataset
    python sft_complete_analysis.py --dataset openmathreasoning --samples 1000 --all

    # Analyze OpenMathReasoning 16K dataset
    python sft_complete_analysis.py --dataset openmathreasoning_16k --samples 1000 --all

    # Analyze Seed Mathematics OpenScience 16K dataset
    python sft_complete_analysis.py --dataset seed_mathematics_16k --samples 1000 --all

    # Analyze other supported datasets
    python sft_complete_analysis.py --dataset openscience_reasoning_2 --samples 500 --visualizations
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from dataset_loaders import get_dataset_loader, list_available_datasets


def setup_output_directory(dataset_name="sft", start_index=None, end_index=None):
    """Create output directory for analysis results"""
    if start_index is not None and end_index is not None:
        output_dir = Path(f"analysis_outputs/{dataset_name}_analysis_output_chunk_{start_index}_{end_index}")
    else:
        output_dir = Path(f"analysis_outputs/{dataset_name}_analysis_output")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (output_dir / "visualizations").mkdir(exist_ok=True)
    (output_dir / "embeddings").mkdir(exist_ok=True)
    (output_dir / "data").mkdir(exist_ok=True)

    return output_dir


def run_statistical_analysis(df, output_dir):
    """Run comprehensive statistical analysis"""

    print("\n" + "=" * 50)
    print("RUNNING STATISTICAL ANALYSIS")
    print("=" * 50)

    # Basic statistics
    stats = {
        'dataset_size': len(df),
        'categories': df['category'].value_counts().to_dict(),
        'subjects': df['raw_subject'].value_counts().head(20).to_dict(),
        'answer_types': df['answer_type'].value_counts().to_dict(),
        'question_stats': {
            'avg_length': float(df['question_length'].mean()),
            'min_length': int(df['question_length'].min()),
            'max_length': int(df['question_length'].max()),
            'std_length': float(df['question_length'].std()),
        },
        'rationale_stats': {
            'avg_length': float(df['rationale_length'].mean()),
            'min_length': int(df['rationale_length'].min()),
            'max_length': int(df['rationale_length'].max()),
            'std_length': float(df['rationale_length'].std()),
        },
        'multimodal_stats': {
            'questions_with_images': int(df['has_image'].sum()),
            'rationales_with_images': int(df['has_rationale_image'].sum()),
            'percentage_with_images': float(df['has_image'].mean() * 100),
        },
        'complexity_stats': {
            'avg_complexity_score': float(df['question_complexity_score'].mean()),
            'questions_with_math': int(df['has_mathematical_content'].sum()),
            'percentage_with_math': float(df['has_mathematical_content'].mean() * 100),
        },
    }

    # Advanced analysis
    print("\n1. Category Distribution Analysis:")
    category_analysis = (
        df.groupby('category')
        .agg(
            {
                'question_length': ['mean', 'std', 'count'],
                'rationale_length': ['mean', 'std'],
                'has_image': 'sum',
                'has_mathematical_content': 'sum',
            }
        )
        .round(2)
    )
    print(category_analysis)

    print("\n2. Subject Complexity Analysis:")
    subject_complexity = (
        df.groupby('raw_subject')
        .agg({'question_complexity_score': 'mean', 'question_length': 'mean', 'has_mathematical_content': 'mean'})
        .sort_values('question_complexity_score', ascending=False)
        .head(10)
        .round(2)
    )
    print(subject_complexity)

    print("\n3. Answer Type Analysis:")
    answer_type_analysis = (
        df.groupby('answer_type')
        .agg(
            {
                'question_length': 'mean',
                'rationale_length': 'mean',
                'has_image': 'mean',
                'question_complexity_score': 'mean',
            }
        )
        .round(2)
    )
    print(answer_type_analysis)

    # Save statistical analysis
    with open(output_dir / "data" / "statistical_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # Save detailed analysis
    category_analysis.to_csv(output_dir / "data" / "category_analysis.csv")
    subject_complexity.to_csv(output_dir / "data" / "subject_complexity.csv")
    answer_type_analysis.to_csv(output_dir / "data" / "answer_type_analysis.csv")

    return stats


def run_visualization_analysis(df, output_dir):
    """Run comprehensive visualization analysis"""

    print("\n" + "=" * 50)
    print("RUNNING VISUALIZATION ANALYSIS")
    print("=" * 50)

    viz_dir = output_dir / "visualizations"

    try:
        # Generate all visualizations with output directory
        difficulty_stats = generate_all_visualizations_with_path(df, viz_dir)

        print("Generated visualizations:")
        print(f"- {viz_dir}/analysis_dashboard.png")
        print(f"- {viz_dir}/subject_complexity_heatmap.png")
        print(f"- {viz_dir}/interactive_dashboard.html")
        print(f"- {viz_dir}/difficulty_analysis.png")
        print("- Word clouds for each category")

        return difficulty_stats

    except Exception as e:
        print(f"Error in visualization: {e}")
        return None


def run_embedding_analysis(df, output_dir, dataset_name="seed"):
    """Run comprehensive embedding analysis"""

    print("\n" + "=" * 50)
    print("RUNNING EMBEDDING ANALYSIS")
    print("=" * 50)

    embed_dir = output_dir / "embeddings"

    try:
        # Run embedding analysis with proper paths
        analyzer, analysis_results = run_comprehensive_embedding_analysis_with_path(
            df, embed_dir, save_embeddings=True, dataset_name=dataset_name
        )

        print("Generated embedding analysis:")
        print(f"- {embed_dir}/hle_embeddings.pkl")
        print(f"- {embed_dir}/hle_embeddings_umap.png")
        print(f"- {embed_dir}/hle_embeddings_tsne.png")
        print(f"- {embed_dir}/category_similarity_matrix.png")
        print(f"- {embed_dir}/hle_embedding_analysis.json")

        return analyzer, analysis_results

    except Exception as e:
        print(f"Error in embedding analysis: {e}")
        return None, None


def generate_summary_report(df, stats, output_dir):
    """Generate a comprehensive summary report"""

    report = f"""
# HLE Dataset Analysis Report

## Dataset Overview
- **Total Samples**: {stats['dataset_size']:,}
- **Categories**: {len(stats['categories'])}
- **Subjects**: {len(stats['subjects'])}
- **Questions with Images**: {stats['multimodal_stats']['questions_with_images']:,} ({stats['multimodal_stats']['percentage_with_images']:.1f}%)
- **Questions with Mathematical Content**: {stats['complexity_stats']['questions_with_math']:,} ({stats['complexity_stats']['percentage_with_math']:.1f}%)

## Category Distribution
"""

    for category, count in stats['categories'].items():
        percentage = (count / stats['dataset_size']) * 100
        report += f"- **{category}**: {count:,} ({percentage:.1f}%)\n"

    report += f"""
## Question Statistics
- **Average Length**: {stats['question_stats']['avg_length']:.0f} characters
- **Length Range**: {stats['question_stats']['min_length']} - {stats['question_stats']['max_length']:,} characters
- **Standard Deviation**: {stats['question_stats']['std_length']:.0f} characters

## Rationale Statistics
- **Average Length**: {stats['rationale_stats']['avg_length']:.0f} characters
- **Length Range**: {stats['rationale_stats']['min_length']} - {stats['rationale_stats']['max_length']:,} characters
- **Standard Deviation**: {stats['rationale_stats']['std_length']:.0f} characters

## Top 10 Subject Areas
"""

    for i, (subject, count) in enumerate(list(stats['subjects'].items())[:10], 1):
        report += f"{i}. **{subject}**: {count:,} questions\n"

    report += """
## Answer Types
"""
    for answer_type, count in stats['answer_types'].items():
        percentage = (count / stats['dataset_size']) * 100
        report += f"- **{answer_type}**: {count:,} ({percentage:.1f}%)\n"

    report += f"""
## Files Generated
### Data Files
- `statistical_analysis.json`: Complete statistical analysis
- `category_analysis.csv`: Analysis by category
- `subject_complexity.csv`: Subject complexity rankings
- `answer_type_analysis.csv`: Analysis by answer type

### Visualizations
- `hle_analysis_dashboard.png`: Main dashboard with 6 key visualizations
- `subject_complexity_heatmap.png`: Complexity distribution across subjects
- `hle_interactive_dashboard.html`: Interactive dashboard (open in browser)
- `difficulty_analysis.png`: Difficulty patterns by category
- Word clouds for each category

### Embeddings (if generated)
- `hle_embeddings.pkl`: Pre-computed embeddings for all questions
- `hle_embeddings_umap.png`: UMAP visualization colored by category
- `hle_embeddings_tsne.png`: t-SNE visualization colored by clusters
- `category_similarity_matrix.png`: Similarity heatmap between categories
- `hle_embedding_analysis.json`: Clustering and outlier analysis

## Usage Examples

### Load and Analyze Data
```python
# Load processed data
df = pd.read_csv('processed_data.csv')

# Basic data exploration
print(f"Dataset shape: {df.shape}")
print(f"Categories: {df['category'].unique()}")
print(f"Answer types: {df['answer_type'].unique()}")
```

### Filter by Category
```python
# Filter by category
math_questions = df[df['category'] == 'Math']
complex_questions = df[df['question_length'] > 1500]
```

### Embedding Analysis
```python
from sft_embeddings import SFTEmbeddingAnalyzer
analyzer = SFTEmbeddingAnalyzer(use_local_model=True)
analyzer.load_embeddings('hle_embeddings.pkl')
```

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    # Save report
    with open(output_dir / "HLE_Analysis_Report.md", 'w', encoding='utf-8') as f:
        f.write(report)

    # Also save processed data
    df.to_csv(output_dir / "data" / "processed_hle_data.csv", index=False)

    print(f"\nSummary report saved to: {output_dir / 'HLE_Analysis_Report.md'}")
    print(f"Processed data saved to: {output_dir / 'data' / 'processed_hle_data.csv'}")


def generate_all_visualizations_with_path(df, output_dir):
    """Generate all visualizations with specified output directory"""

    # Import specific functions to avoid wildcard import
    from sft_visualization import (
        analyze_difficulty_patterns_with_path,
        create_hle_visualizations_with_path,
        create_interactive_dashboard_with_path,
        create_subject_complexity_heatmap_with_path,
        create_wordcloud_by_category_with_path,
    )

    # Save current working directory
    original_cwd = os.getcwd()

    try:
        # Ensure output directory exists
        output_dir.mkdir(exist_ok=True)

        print("Generating HLE visualizations...")

        # Basic dashboard
        create_hle_visualizations_with_path(df, output_dir)

        # Subject complexity heatmap
        create_subject_complexity_heatmap_with_path(df, output_dir)

        # Interactive dashboard
        create_interactive_dashboard_with_path(df, output_dir)

        # Word clouds for each category
        for category in df['category'].unique():
            if df[df['category'] == category].shape[0] > 10:  # Only if enough samples
                create_wordcloud_by_category_with_path(df, category, output_dir)

        # Difficulty analysis
        difficulty_stats = analyze_difficulty_patterns_with_path(df, output_dir)

        print("All visualizations completed!")
        return difficulty_stats

    except Exception as e:
        print(f"Error in visualization generation: {e}")
        return None
    finally:
        # Restore working directory
        os.chdir(original_cwd)


def run_comprehensive_embedding_analysis_with_path(df, output_dir, save_embeddings=True, dataset_name="hle"):
    """Run complete embedding analysis pipeline with specified output directory

    Args:
        df: DataFrame containing the data to analyze
        output_dir: Path to output directory for saving results
        save_embeddings: Whether to save embeddings to disk
        dataset_name: Name of the dataset (used for file naming and output messages)
    """

    # Import here to avoid circular imports
    import json

    from sft_embeddings import SFTEmbeddingAnalyzer

    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)

    # Save current working directory
    original_cwd = os.getcwd()

    try:
        # Check if we should use local model (default to False for backward compatibility)
        use_local_model = os.environ.get("USE_LOCAL_MODEL", "true").lower() == "true"
        analyzer = SFTEmbeddingAnalyzer(use_local_model=use_local_model)

        # Compute embeddings with dataset-specific naming
        embeddings_path = output_dir / f'{dataset_name}_embeddings.pkl' if save_embeddings else None
        embeddings = analyzer.compute_hle_embeddings(  # NOQA F841
            df, save_path=str(embeddings_path) if embeddings_path else None
        )

        # Perform clustering with adaptive number of clusters
        n_clusters = min(8, len(df) // 2)  # Ensure n_clusters <= n_samples/2
        if n_clusters < 2:
            n_clusters = 2  # Minimum 2 clusters
        cluster_labels, cluster_analysis = analyzer.cluster_questions(n_clusters=n_clusters)

        # Visualize embeddings with dataset-specific naming
        analyzer.visualize_embeddings_2d(
            method='umap', color_by='category', save_path=str(output_dir / f'{dataset_name}_embeddings_umap.png')
        )

        analyzer.visualize_embeddings_2d(
            method='tsne', color_by='cluster', save_path=str(output_dir / f'{dataset_name}_embeddings_tsne.png')
        )

        # Save analysis results with dataset-specific naming
        analysis_results = {
            'cluster_analysis': cluster_analysis,
        }

        with open(output_dir / f'{dataset_name}_embedding_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2)

        print(f"Comprehensive embedding analysis for {dataset_name} dataset completed!")
        return analyzer, analysis_results

    except Exception as e:
        print(f"Error in embedding analysis: {e}")
        return None, None
    finally:
        # Restore working directory
        os.chdir(original_cwd)


def main():
    """Main function to run the complete HLE analysis pipeline"""

    from pathlib import Path

    parser = argparse.ArgumentParser(description='Complete Dataset Analysis Pipeline')
    parser.add_argument('--samples', type=int, default=500, help='Number of samples to process (default: 500)')
    parser.add_argument('--embeddings', action='store_true', help='Run embedding analysis (requires OpenAI API key)')
    parser.add_argument('--visualizations', action='store_true', help='Generate visualizations')
    parser.add_argument(
        '--all', action='store_true', help='Run complete analysis including embeddings and visualizations'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        help='Dataset to analyze: either a HuggingFace dataset name (e.g., hle, openmathreasoning, openscience_reasoning_2) or path to a JSONL file',
    )
    parser.add_argument(
        '--dataset-config', type=str, help='Configuration/subset name for HuggingFace dataset (if applicable)'
    )
    parser.add_argument(
        '--start-index', type=int, default=0, help='Starting index for dataset range selection (default: 0)'
    )
    parser.add_argument(
        '--end-index',
        type=int,
        help='Ending index for dataset range selection. If not provided, uses start-index + samples',
    )

    args = parser.parse_args()

    if args.all:
        args.embeddings = True
        args.visualizations = True

    print("🚀 Starting Dataset Analysis Pipeline")
    print("=" * 60)

    # Determine dataset type and name
    # Initialize default values
    dataset_type = "hle"
    dataset_name = "hle"
    data_source = None

    if args.dataset:
        if os.path.exists(args.dataset) and args.dataset.endswith('.jsonl'):
            # JSONL file path
            dataset_type = "jsonl"
            dataset_name = Path(args.dataset).stem
            data_source = args.dataset
        else:
            # Check if it's a supported dataset name
            try:
                # This will raise ValueError if dataset is not supported
                get_dataset_loader(args.dataset)
                dataset_type = args.dataset
                dataset_name = args.dataset
                data_source = None
            except ValueError:
                print(f"❌ Error: Dataset '{args.dataset}' is not supported.")
                print(f"Available datasets: {list_available_datasets()}")
                print("Or provide a path to a JSONL file.")
                return

    print(f"Dataset type: {dataset_type}")
    print(f"Dataset name: {dataset_name}")
    print(f"Data source: {data_source}")

    # Create output directory with range information if applicable
    if args.start_index is not None and args.end_index is not None:
        output_dir = setup_output_directory(dataset_name, args.start_index, args.end_index)
    else:
        output_dir = setup_output_directory(dataset_name)

    print(f"Dataset: {dataset_name}")
    print(f"Output directory: {output_dir}")

    # Load data based on dataset type
    if dataset_type == "jsonl":
        print(f"📊 Loading JSONL dataset from {data_source}...")
        # For JSONL files, we need to use the seed data loader with custom path
        from dataset_loaders import load_and_preprocess_seed_data

        df = load_and_preprocess_seed_data(jsonl_path=data_source, n_samples=args.samples)
        if df is None:
            print("❌ Failed to load JSONL data. Exiting.")
            return
    elif dataset_type == "hle":
        # Default HLE dataset
        print("📊 Loading HLE dataset...")
        df = get_dataset_loader("hle")(n_samples=args.samples)
    else:
        # Use the factory function for all other datasets
        try:
            print(f"📊 Loading {dataset_name} dataset...")
            loader_func = get_dataset_loader(dataset_name)

            # Special handling for datasets that support range parameters
            if dataset_name in ["openscience_reasoning_2"]:  # Only OpenScience supports range parameters
                df = loader_func(n_samples=args.samples, start_index=args.start_index, end_index=args.end_index)
            else:
                df = loader_func(n_samples=args.samples)

            if df is None or df.empty:
                print(f"❌ Failed to load {dataset_name} data. Exiting.")
                return
        except ValueError as e:
            print(f"❌ Dataset '{dataset_name}' not supported: {e}")
            print(f"Available datasets: {list_available_datasets()}")
            return
        except Exception as e:
            print(f"❌ Error loading {dataset_name} dataset: {e}")
            return

    # Statistical analysis (always run)
    stats = run_statistical_analysis(df, output_dir)

    # Visualization analysis
    if args.visualizations:
        viz_results = run_visualization_analysis(df, output_dir)  # NOQA F841

    # Embedding analysis
    if args.embeddings:
        if not os.environ.get("OPENAI_API_KEY"):
            print("⚠️  Warning: OPENAI_API_KEY not found. Skipping embedding analysis.")
        else:
            embed_analyzer, embed_results = run_embedding_analysis(
                df, output_dir, dataset_name=dataset_name
            )  # NOQA F841

    # Generate summary report
    generate_summary_report(df, stats, output_dir)

    print("\n" + "=" * 60)
    print("✅ Dataset Analysis Pipeline Completed!")
    print(f"📁 Results saved in: {output_dir}")
    print("📄 Check HLE_Analysis_Report.md for detailed summary")
    print("=" * 60)


if __name__ == "__main__":
    main()
