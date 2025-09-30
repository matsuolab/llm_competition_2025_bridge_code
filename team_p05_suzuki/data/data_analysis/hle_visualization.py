from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots
from wordcloud import WordCloud


def create_hle_visualizations(df):
    """Create comprehensive visualizations for HLE dataset"""

    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('HLE Dataset Analysis Dashboard', fontsize=16, fontweight='bold')

    # 1. Category Distribution
    category_counts = df['category'].value_counts()
    axes[0, 0].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
    axes[0, 0].set_title('Distribution by Category')

    # 2. Answer Type Distribution
    answer_type_counts = df['answer_type'].value_counts()
    axes[0, 1].bar(answer_type_counts.index, answer_type_counts.values, color=['#FF6B6B', '#4ECDC4'])
    axes[0, 1].set_title('Answer Type Distribution')
    axes[0, 1].set_ylabel('Count')

    # 3. Question Length Distribution
    axes[0, 2].hist(df['question_length'], bins=30, color='skyblue', alpha=0.7)
    axes[0, 2].set_title('Question Length Distribution')
    axes[0, 2].set_xlabel('Question Length (characters)')
    axes[0, 2].set_ylabel('Frequency')

    # 4. Top 10 Subjects
    top_subjects = df['raw_subject'].value_counts().head(10)
    axes[1, 0].barh(range(len(top_subjects)), top_subjects.values)
    axes[1, 0].set_yticks(range(len(top_subjects)))
    axes[1, 0].set_yticklabels(top_subjects.index, fontsize=8)
    axes[1, 0].set_title('Top 10 Subject Areas')
    axes[1, 0].set_xlabel('Count')

    # 5. Multimodal Content Analysis
    multimodal_data = {
        'Questions with Images': df['has_image'].sum(),
        'Questions without Images': len(df) - df['has_image'].sum(),
        'Rationales with Images': df['has_rationale_image'].sum(),
        'Rationales without Images': len(df) - df['has_rationale_image'].sum(),
    }

    values = list(multimodal_data.values())

    x = np.arange(2)
    width = 0.35

    axes[1, 1].bar(x - width / 2, [values[0], values[2]], width, label='With Images', color='#FF6B6B')
    axes[1, 1].bar(x + width / 2, [values[1], values[3]], width, label='Without Images', color='#4ECDC4')
    axes[1, 1].set_title('Multimodal Content Analysis')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(['Questions', 'Rationales'])
    axes[1, 1].legend()

    # 6. Question vs Rationale Length Scatter
    sample_df = df.sample(min(200, len(df)))  # Sample for readability
    axes[1, 2].scatter(sample_df['question_length'], sample_df['rationale_length'], alpha=0.6, color='purple')
    axes[1, 2].set_title('Question vs Rationale Length')
    axes[1, 2].set_xlabel('Question Length')
    axes[1, 2].set_ylabel('Rationale Length')

    plt.tight_layout()
    plt.savefig('hle_analysis_dashboard.png', dpi=300, bbox_inches='tight')
    # plt.show()


def create_hle_visualizations_with_path(df, output_dir):
    """Create comprehensive visualizations for HLE dataset with specified output path"""

    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('HLE Dataset Analysis Dashboard', fontsize=16, fontweight='bold')

    # 1. Category Distribution
    category_counts = df['category'].value_counts()
    axes[0, 0].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
    axes[0, 0].set_title('Distribution by Category')

    # 2. Answer Type Distribution
    answer_type_counts = df['answer_type'].value_counts()
    axes[0, 1].bar(answer_type_counts.index, answer_type_counts.values, color=['#FF6B6B', '#4ECDC4'])
    axes[0, 1].set_title('Answer Type Distribution')
    axes[0, 1].set_ylabel('Count')

    # 3. Question Length Distribution
    axes[0, 2].hist(df['question_length'], bins=30, color='skyblue', alpha=0.7)
    axes[0, 2].set_title('Question Length Distribution')
    axes[0, 2].set_xlabel('Question Length (characters)')
    axes[0, 2].set_ylabel('Frequency')

    # 4. Top 10 Subjects
    top_subjects = df['raw_subject'].value_counts().head(10)
    axes[1, 0].barh(range(len(top_subjects)), top_subjects.values)
    axes[1, 0].set_yticks(range(len(top_subjects)))
    axes[1, 0].set_yticklabels(top_subjects.index, fontsize=8)
    axes[1, 0].set_title('Top 10 Subject Areas')
    axes[1, 0].set_xlabel('Count')

    # 5. Multimodal Content Analysis
    multimodal_data = {
        'Questions with Images': df['has_image'].sum(),
        'Questions without Images': len(df) - df['has_image'].sum(),
        'Rationales with Images': df['has_rationale_image'].sum(),
        'Rationales without Images': len(df) - df['has_rationale_image'].sum(),
    }

    values = list(multimodal_data.values())
    x = np.arange(2)
    width = 0.35

    axes[1, 1].bar(x - width / 2, [values[0], values[2]], width, label='With Images', color='#FF6B6B')
    axes[1, 1].bar(x + width / 2, [values[1], values[3]], width, label='Without Images', color='#4ECDC4')
    axes[1, 1].set_title('Multimodal Content Analysis')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(['Questions', 'Rationales'])
    axes[1, 1].legend()

    # 6. Question vs Rationale Length Scatter
    sample_df = df.sample(min(200, len(df)))  # Sample for readability
    axes[1, 2].scatter(sample_df['question_length'], sample_df['rationale_length'], alpha=0.6, color='purple')
    axes[1, 2].set_title('Question vs Rationale Length')
    axes[1, 2].set_xlabel('Question Length')
    axes[1, 2].set_ylabel('Rationale Length')

    plt.tight_layout()
    save_path = Path(output_dir) / 'hle_analysis_dashboard.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()


def create_subject_complexity_heatmap(df):
    """Create heatmap showing complexity distribution across subjects"""

    # Create complexity bins
    df_copy = df.copy()
    df_copy['complexity_bin'] = pd.cut(
        df_copy['question_length'],
        bins=[0, 500, 1000, 2000, float('inf')],
        labels=['Simple', 'Medium', 'Complex', 'Very Complex'],
    )

    # Create heatmap data
    heatmap_data = pd.crosstab(df_copy['raw_subject'], df_copy['complexity_bin'])

    # Take top 15 subjects for readability
    top_subjects = df['raw_subject'].value_counts().head(15).index
    heatmap_data = heatmap_data.loc[top_subjects]

    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd')
    plt.title('Question Complexity Distribution by Subject')
    plt.xlabel('Complexity Level')
    plt.ylabel('Subject Area')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('subject_complexity_heatmap.png', dpi=300, bbox_inches='tight')
    # plt.show()


def create_subject_complexity_heatmap_with_path(df, output_dir):
    """Create heatmap showing complexity distribution across subjects with specified output path"""

    # Create complexity bins
    df_copy = df.copy()
    df_copy['complexity_bin'] = pd.cut(
        df_copy['question_length'],
        bins=[0, 500, 1000, 2000, float('inf')],
        labels=['Simple', 'Medium', 'Complex', 'Very Complex'],
    )

    # Create heatmap data
    heatmap_data = pd.crosstab(df_copy['raw_subject'], df_copy['complexity_bin'])

    # Take top 15 subjects for readability
    top_subjects = df['raw_subject'].value_counts().head(15).index
    heatmap_data = heatmap_data.loc[top_subjects]

    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd')
    plt.title('Question Complexity Distribution by Subject')
    plt.xlabel('Complexity Level')
    plt.ylabel('Subject Area')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_path = Path(output_dir) / 'subject_complexity_heatmap.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()


def create_interactive_dashboard(df):
    """Create interactive Plotly dashboard"""

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            'Category Distribution',
            'Question Length by Category',
            'Subject Distribution',
            'Complexity vs Category',
        ),
        specs=[[{"type": "pie"}, {"type": "box"}], [{"type": "bar"}, {"type": "scatter"}]],
    )

    # 1. Category pie chart
    category_counts = df['category'].value_counts()
    fig.add_trace(go.Pie(labels=category_counts.index, values=category_counts.values), row=1, col=1)

    # 2. Box plot of question length by category
    for i, category in enumerate(df['category'].unique()):
        category_data = df[df['category'] == category]['question_length']
        fig.add_trace(go.Box(y=category_data, name=category, showlegend=False), row=1, col=2)

    # 3. Top subjects bar chart
    top_subjects = df['raw_subject'].value_counts().head(10)
    fig.add_trace(go.Bar(x=top_subjects.values, y=top_subjects.index, orientation='h', showlegend=False), row=2, col=1)

    # 4. Complexity scatter plot
    fig.add_trace(
        go.Scatter(
            x=df['question_length'],
            y=df['rationale_length'],
            mode='markers',
            text=df['category'],
            marker=dict(color=df['category'].astype('category').cat.codes, colorscale='Viridis'),
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    fig.update_layout(height=800, title_text="HLE Dataset Interactive Dashboard")
    fig.write_html("hle_interactive_dashboard.html")
    # fig.show()


def create_interactive_dashboard_with_path(df, output_dir, show=False):
    """Create interactive Plotly dashboard with specified output path"""

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            'Category Distribution',
            'Question Length by Category',
            'Subject Distribution',
            'Complexity vs Category',
        ),
        specs=[[{"type": "pie"}, {"type": "box"}], [{"type": "bar"}, {"type": "scatter"}]],
    )

    # 1. Category pie chart
    category_counts = df['category'].value_counts()
    fig.add_trace(go.Pie(labels=category_counts.index, values=category_counts.values), row=1, col=1)

    # 2. Box plot of question length by category
    for i, category in enumerate(df['category'].unique()):
        category_data = df[df['category'] == category]['question_length']
        fig.add_trace(go.Box(y=category_data, name=category, showlegend=False), row=1, col=2)

    # 3. Top subjects bar chart
    top_subjects = df['raw_subject'].value_counts().head(10)
    fig.add_trace(go.Bar(x=top_subjects.values, y=top_subjects.index, orientation='h', showlegend=False), row=2, col=1)

    # 4. Complexity scatter plot
    fig.add_trace(
        go.Scatter(
            x=df['question_length'],
            y=df['rationale_length'],
            mode='markers',
            text=df['category'],
            marker=dict(color=df['category'].astype('category').cat.codes, colorscale='Viridis'),
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    fig.update_layout(height=800, title_text="HLE Dataset Interactive Dashboard")
    save_path = Path(output_dir) / "hle_interactive_dashboard.html"
    fig.write_html(save_path)
    if show:
        fig.show()


def create_wordcloud_by_category(df, category_name):
    """Create word cloud for questions in a specific category"""

    category_data = df[df['category'] == category_name]
    all_questions = ' '.join(category_data['question'].astype(str))

    # Remove common stop words and add domain-specific ones
    stopwords = {
        'the',
        'and',
        'or',
        'but',
        'in',
        'on',
        'at',
        'to',
        'for',
        'of',
        'with',
        'by',
        'what',
        'which',
        'how',
        'when',
        'where',
        'why',
        'who',
        'is',
        'are',
        'was',
        'were',
        'answer',
        'question',
        'following',
        'choices',
        'choice',
        'select',
        'find',
        'calculate',
    }

    wordcloud = WordCloud(
        width=800, height=400, background_color='white', stopwords=stopwords, max_words=100, colormap='viridis'
    ).generate(all_questions)

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {category_name} Questions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'wordcloud_{category_name.lower().replace("/", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_wordcloud_by_category_with_path(df, category_name, output_dir):
    """Create word cloud for questions in a specific category with specified output path"""

    category_data = df[df['category'] == category_name]
    all_questions = ' '.join(category_data['question'].astype(str))

    # Remove common stop words and add domain-specific ones
    stopwords = {
        'the',
        'and',
        'or',
        'but',
        'in',
        'on',
        'at',
        'to',
        'for',
        'of',
        'with',
        'by',
        'what',
        'which',
        'how',
        'when',
        'where',
        'why',
        'who',
        'is',
        'are',
        'was',
        'were',
        'answer',
        'question',
        'following',
        'choices',
        'choice',
        'select',
        'find',
        'calculate',
    }

    wordcloud = WordCloud(  # NOQA F841
        width=800, height=400, background_color='white', stopwords=stopwords, max_words=100, colormap='viridis'
    ).generate(all_questions)

    plt.figure(figsize=(12, 6))
    # plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {category_name} Questions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_path = Path(output_dir) / f'wordcloud_{category_name.lower().replace("/", "_")}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()


def analyze_difficulty_patterns(df):
    """Analyze patterns that might indicate question difficulty"""

    # Create difficulty proxy based on question length and rationale length
    df['difficulty_score'] = (df['question_length'] - df['question_length'].mean()) / df['question_length'].std() + (
        df['rationale_length'] - df['rationale_length'].mean()
    ) / df['rationale_length'].std()

    # Analyze by category
    difficulty_by_category = df.groupby('category')['difficulty_score'].agg(['mean', 'std', 'count'])

    plt.figure(figsize=(12, 6))
    x = range(len(difficulty_by_category))
    plt.bar(
        x, difficulty_by_category['mean'], yerr=difficulty_by_category['std'], capsize=5, alpha=0.7, color='steelblue'
    )
    plt.xlabel('Category')
    plt.ylabel('Average Difficulty Score')
    plt.title('Estimated Difficulty by Category')
    plt.xticks(x, difficulty_by_category.index, rotation=45)
    plt.tight_layout()
    plt.savefig('difficulty_analysis.png', dpi=300, bbox_inches='tight')
    # plt.show()

    return difficulty_by_category


def analyze_difficulty_patterns_with_path(df, output_dir):
    """Analyze patterns that might indicate question difficulty with specified output path"""

    # Create difficulty proxy based on question length and rationale length
    df_copy = df.copy()
    df_copy['difficulty_score'] = (df_copy['question_length'] - df_copy['question_length'].mean()) / df_copy[
        'question_length'
    ].std() + (df_copy['rationale_length'] - df_copy['rationale_length'].mean()) / df_copy['rationale_length'].std()

    # Analyze by category
    difficulty_by_category = df_copy.groupby('category')['difficulty_score'].agg(['mean', 'std', 'count'])

    plt.figure(figsize=(12, 6))
    x = range(len(difficulty_by_category))
    plt.bar(
        x, difficulty_by_category['mean'], yerr=difficulty_by_category['std'], capsize=5, alpha=0.7, color='steelblue'
    )
    plt.xlabel('Category')
    plt.ylabel('Average Difficulty Score')
    plt.title('Estimated Difficulty by Category')
    plt.xticks(x, difficulty_by_category.index, rotation=45)
    plt.tight_layout()
    save_path = Path(output_dir) / 'difficulty_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()

    return difficulty_by_category


# Example usage functions
def generate_all_visualizations(df):
    """Generate all visualizations for the HLE dataset"""

    print("Generating HLE visualizations...")

    # Basic dashboard
    create_hle_visualizations(df)

    # Subject complexity heatmap
    create_subject_complexity_heatmap(df)

    # Interactive dashboard
    create_interactive_dashboard(df)

    # Word clouds for each category
    for category in df['category'].unique():
        if df[df['category'] == category].shape[0] > 10:  # Only if enough samples
            create_wordcloud_by_category(df, category)

    # Difficulty analysis
    difficulty_stats = analyze_difficulty_patterns(df)

    print("All visualizations completed!")
    return difficulty_stats
