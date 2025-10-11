# v5 Knowledge-Based RAG Data Upgrading System

A production-ready system for upgrading educational datasets to doctoral level using knowledge-based RAG technology with local LLM inference.

## Overview

This system (v5) transforms existing educational datasets into doctoral-level materials through:
- **Knowledge-Based RAG**: FAISS-indexed ArXiv papers for theoretical grounding
- **Multi-Stage Pipeline**: Problem upgrade → Thinking enhancement → Answer validation → Quality assurance
- **Local LLM Inference**: vLLM-optimized Qwen3-235B-A22B model (no API calls)
- **Data Synthesis Flow**: Based on proven methodology from reference implementations

### Key Features

- 🎓 **Doctoral-Level Upgrade**: Transforms problems into PhD-level challenges
- 🧠 **Enhanced Reasoning**: Expands basic solutions into comprehensive theoretical analysis
- 📚 **Knowledge Integration**: Leverages specialized ArXiv papers for context
- ✅ **Multi-Stage Validation**: Automated quality checks and fixes
- 🤖 **vLLM Optimization**: 8-GPU tensor parallelism for efficient processing
- 🔄 **Fault Tolerance**: Dynamic task queue for distributed multi-node processing
- 📊 **Production Ready**: Comprehensive error handling and monitoring

### Architecture

```
Input (Seed Data)
    ↓
1. Problem Upgrade (with knowledge context)
    ↓
2. Thinking Enhancement (rigorous derivations)
    ↓
3. Answer Validation (format & correctness)
    ↓
4. Quality Assurance (comprehensive validation)
    ↓
Output (Doctoral-Level Data)
```

## Requirements

### System Requirements
- **GPUs**: 8x NVIDIA GPUs (A100/H100 recommended)
- **RAM**: 1TB+ system memory
- **Storage**: 500GB+ for models and indexes
- **OS**: Linux (tested on Ubuntu 20.04+)

### Software Dependencies
- Python 3.10+
- CUDA 11.8+
- Miniconda3
- See `requirements.txt` for Python packages

## Installation

### 1. Environment Setup

```bash
# Clone repository
cd /home/suzuki/projects/team_suzuki/data/data_generation/sft_data_generation/v5_knowledge_based_rag

# Create conda environment
conda create -n v5_datagen python=3.10
conda activate v5_datagen

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy configuration template
cp config/.env.example config/.env

# Edit configuration
vim config/.env
```

**Key Configuration Parameters:**

```bash
# Model Configuration
MODEL_PATH=/path/to/Qwen3-235B-A22B-Thinking-2507
TENSOR_PARALLEL_SIZE=8
GPU_MEMORY_UTILIZATION=0.95

# Knowledge RAG
USE_KNOWLEDGE_RAG=true
KNOWLEDGE_INDEX_PATH=/path/to/knowledge_index.faiss
KNOWLEDGE_METADATA_PATH=/path/to/knowledge_metadata.json

# Data Sources
SEED_DATASET=your-org/your-seed-dataset
HF_TOKEN=your_huggingface_token

# Generation Settings
DEFAULT_DIFFICULTY=doctoral
BATCH_SIZE=3
```

### 3. Knowledge Index Setup

The system requires a pre-built FAISS knowledge index:

```bash
# Option 1: Use pre-built index (recommended)
# Place index files in configured KNOWLEDGE_INDEX_PATH

# Option 2: Build from HuggingFace dataset
python scripts/build_knowledge_index.py \
    --dataset your-org/RAG-dataset \
    --output-dir /path/to/knowledge_indexes
```

## Usage

### Basic Execution

```bash
# Single-node processing
sbatch scripts/run_generate_data.sh \
    --dataset your-org/seed-data \
    --limit 1000 \
    --batch-size 3

# Full dataset processing
sbatch scripts/run_generate_data.sh \
    --dataset your-org/seed-data

# Custom output directory
sbatch scripts/run_generate_data.sh \
    --dataset your-org/seed-data \
    --output-dir /custom/output/path
```

### Advanced Options

```bash
# Multi-node distributed processing
sbatch scripts/run_generate_data.sh \
    --dataset your-org/seed-data \
    --use-dynamic-queue

# Adjust batch size for memory optimization
sbatch scripts/run_generate_data.sh \
    --dataset your-org/seed-data \
    --batch-size 2  # Reduce if OOM occurs

# Disable knowledge RAG (not recommended)
sbatch scripts/run_generate_data.sh \
    --dataset your-org/seed-data \
    --no-knowledge-rag
```

## Data Format

### Input Format

Input datasets must contain:

```json
{
  "question": "Original question text",
  "think": "Original thinking process/solution",
  "answer": "Original answer",
  "subject": "Mathematics|Physics|Chemistry|etc",
  "data_id": "unique_identifier"
}
```

### Output Format

Generated doctoral-level data:

```json
{
  "question": "Upgraded doctoral-level question with theoretical depth",
  "think": "Enhanced reasoning with rigorous mathematical derivations",
  "answer": "Validated answer (MCQ letter or concise result)",
  "subject": "Subject area",
  "data_id": "original_id",
  "answer_type": "multipleChoice|exactMatch",
  "original_question": "Reference to original problem",
  "original_thinking": "Reference to original solution",
  "original_answer": "Reference to original answer",
  "used_knowledge": true,
  "knowledge_docs_count": 5,
  "upgraded_at": "2025-01-19T12:00:00"
}
```

## Data Synthesis Flow

The system implements a proven data synthesis methodology:

1. **Topic Extraction**: Extract key concepts from problem statement
2. **Knowledge Search**: Retrieve relevant context from FAISS index
3. **Prompt Generation**: Build structured prompts with strict format requirements
4. **Response Parsing**: Extract problem, thinking, and answer using regex patterns
5. **Validation & Fixing**: Multi-stage quality checks with automated repairs

This flow is based on the reference implementation in `/home/suzuki/projects/data_generation/generate_data.py`.

## Monitoring

### Progress Tracking

```bash
# Monitor job logs
tail -f /path/to/logs/upgrade_data_v5_doctoral-<job_id>.out

# Check GPU usage
watch -n 1 nvidia-smi

# View generation statistics
cat <output_dir>/generation_stats.json
```

### Output Statistics

```json
{
  "total_seeds": 1000,
  "total_problems": 1000,
  "problems_per_seed": 1,
  "used_knowledge_rag": true,
  "timestamp": "2025-01-19T12:00:00"
}
```

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size
--batch-size 2

# Reduce GPU memory utilization
# In config/.env: GPU_MEMORY_UTILIZATION=0.90

# Reduce max model length
# In config/.env: MAX_MODEL_LEN=65536
```

### Knowledge Index Not Found

```bash
# Verify index paths
ls -la $KNOWLEDGE_INDEX_PATH
ls -la $KNOWLEDGE_METADATA_PATH

# Rebuild index if needed
python scripts/build_knowledge_index.py \
    --dataset your-org/RAG-dataset \
    --output-dir /path/to/knowledge_indexes
```

### Slow Processing

```bash
# Check GPU utilization
nvidia-smi

# Verify tensor parallelism
# In config/.env: TENSOR_PARALLEL_SIZE=8

# Enable dynamic queue for multi-node scaling
--use-dynamic-queue
```

## Project Structure

```
v5_knowledge_based_rag/
├── README.md                      # This file
├── config/
│   ├── .env.example              # Configuration template
│   └── .env                      # User configuration (create from .env.example)
├── data_generation/
│   ├── generate_data.py          # Main generation script
│   ├── dynamic_task_queue.py     # Distributed task management
│   └── prompts/
│       └── prompt_templates.py   # Doctoral-level prompts
├── rag/
│   └── knowledge_rag_store.py    # Knowledge retrieval system
├── scripts/
│   ├── run_generate_data.sh      # Main execution script (SLURM)
│   ├── build_knowledge_index.py  # Index builder
│   ├── monitor_and_merge.py      # Result monitoring
│   └── auto_submit_v5.sh         # Automatic job submission
└── arxiv/
    ├── arxiv_scraper.py          # ArXiv paper scraper
    └── build_knowledge_from_hf.py # HF dataset index builder
```

## Performance Optimization

### Memory Management
- **Batch Size**: Adjust based on available GPU memory (default: 3)
- **GPU Memory Utilization**: 0.90-0.95 for optimal performance
- **Swap Space**: 8GB for overflow handling

### Parallel Processing
- **Tensor Parallelism**: 8 GPUs for model sharding
- **Batch Parallelism**: Process multiple items concurrently
- **Multi-Node**: Dynamic queue for distributed scaling

### Best Practices
1. Start with small batches (--limit 10) to verify configuration
2. Monitor GPU memory with `nvidia-smi`
3. Use dynamic queue for large-scale processing (1000+ items)
4. Ensure knowledge index is on fast storage (NVMe SSD recommended)

## Citation

If you use this system in your research, please cite:

```bibtex
@software{v5_knowledge_rag_2025,
  title = {v5 Knowledge-Based RAG Data Upgrading System},
  author = {Team Suzuki},
  year = {2025},
  url = {https://github.com/team-suzuki/v5_knowledge_based_rag}
}
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Support

For issues or questions:
- Create an issue in the repository
- Check existing documentation in `/docs`
- Review troubleshooting section above

## Acknowledgments

- Built on vLLM inference framework
- Uses Qwen3-235B-A22B-Thinking-2507 model
- FAISS vector search by Facebook AI Research
- Based on proven data synthesis methodologies
