# SFT Data Generation Tool

Production-grade SFT (Supervised Fine-Tuning) data generation toolkit powered by vLLM and local LLM inference. Generate high-quality educational datasets with knowledge-based RAG and multi-stage validation.

## 🚀 Key Features

- **vLLM-Optimized**: High-speed inference with memory-efficient processing
- **Multi-Version Pipeline**: 5 generation modes for different use cases
- **Knowledge-Based RAG**: FAISS-indexed ArXiv papers for theoretical grounding
- **Multi-CoT Reasoning**: Generate and select optimal solutions via majority voting
- **Automated Quality Control**: Answer validation, majority voting, and GenSelect
- **Single-Node Execution**: 8-GPU parallel processing with optimized memory usage
- **Resume Capability**: Continue from interruptions seamlessly
- **Local LLM Inference**: No external API calls required

## 📁 Directory Structure

```
sft_data_generation/
├── v1_no_seed/                    # Generate problems from scratch
│   ├── generate_data.py           # Main generation script
│   ├── mp_spawn_wrapper.py        # Multiprocessing wrapper
│   └── run_generate_data.sh       # SLURM execution script
│
├── v2_seed_no_rag/                # Seed-based generation (basic)
│   ├── generate_data.py           # Seed-based script
│   ├── mp_spawn_wrapper.py        # Multiprocessing wrapper
│   └── run_generate_data.sh       # SLURM execution script
│
├── v3_seed_with_rag/              # Seed-based with problem RAG
│   ├── generate_data.py           # RAG-enhanced script
│   ├── mp_spawn_wrapper.py        # Multiprocessing wrapper
│   ├── rag_vector_store.py        # Vector store manager
│   ├── run_generate_data.sh       # SLURM execution script
│   └── vector_store/              # FAISS index storage
│
├── v4_knowledge_based_rag/        # Knowledge-based RAG (ArXiv)
│   ├── data_generation/           # Data generation core
│   │   ├── generate_data.py       # Main generation script
│   │   └── prompts/               # Prompt templates
│   ├── rag/                       # RAG infrastructure
│   │   └── knowledge_rag_store.py # Knowledge store implementation
│   ├── arxiv/                     # ArXiv paper processing
│   │   ├── arxiv_to_sheets.py     # ArXiv → Google Sheets
│   │   └── build_knowledge_from_arxiv.py # Index builder
│   ├── scripts/                   # Execution scripts
│   │   ├── run_scraping.sh        # ArXiv paper collection
│   │   ├── run_build_index.sh     # Index construction
│   │   ├── run_generate_data.sh   # Data generation
│   │   └── run_arxiv_sheets_pipeline.sh # Integrated pipeline
│   ├── config/                    # Configuration
│   │   └── .env.example           # Configuration template
│   └── docs/                      # Documentation
│       ├── QUICK_START.md         # Quick start guide
│       ├── USAGE_GUIDE.md         # Detailed usage
│       └── DEPLOYMENT_MODES.md    # Deployment modes
│
├── v5_knowledge_based_rag/        # ⭐ Latest: Doctoral-level upgrade ⭐
│   ├── data_generation/           # Data generation core
│   │   ├── generate_data.py       # Main generation script
│   │   ├── dynamic_task_queue.py  # Distributed task management
│   │   └── prompts/               # Doctoral-level prompt templates
│   ├── rag/                       # RAG infrastructure
│   │   └── knowledge_rag_store.py # Knowledge retrieval system
│   ├── arxiv/                     # ArXiv paper processing
│   │   ├── arxiv_scraper.py       # ArXiv paper scraper
│   │   └── build_knowledge_from_hf.py # HF dataset index builder
│   ├── scripts/                   # Execution scripts
│   │   ├── run_generate_data.sh   # Main execution script
│   │   ├── build_knowledge_index.py # Index builder
│   │   └── monitor_and_merge.py   # Result monitoring
│   ├── config/                    # Configuration
│   │   └── .env.example           # Configuration template
│   └── README.md                  # v5-specific documentation
│
├── common/                        # Shared modules
│   ├── utils.py                   # vLLM initialization, utilities
│   └── data_cleaning.py           # Data cleaning processing
│
├── setup_shared_environment.sh    # Shared environment setup
└── environment.yaml               # Conda environment definition
```

## 🛠️ Setup

### ⚡ One-Click Environment Setup (Recommended)

```bash
# Automated shared environment setup
bash setup_shared_environment.sh
```

This script automatically:
- Creates base directories (`/home/Competition2025/P05/shareP05/data_generation/`)
- Builds Conda environment (including v4/v5 packages)
- Prepares data output destinations
- Sets permissions (777 - accessible to all users)

### 🔄 Using the Environment

```bash
# Activate shared environment
conda activate /home/Competition2025/P05/shareP05/data_generation/data_generation_env
```

### Required Dependencies

- **Python 3.11**
- **CUDA 12.8**
- **vLLM 0.10.0**
- **PyTorch 2.7.1+cu128**
- **Multiprocessing**: No Ray, uses vLLM's native multiprocessing
- **faiss-gpu 1.8.0** (for RAG)
- **sentence-transformers 3.0.1** (for knowledge embeddings)

## 📊 Version Comparison and Use Cases

| Version | Purpose | Features | Recommended Scenario |
|---------|---------|----------|---------------------|
| **v1_no_seed** | Fresh generation | Generate diverse problems from scratch | Initial dataset creation |
| **v2_seed_no_rag** | Seed-based | Derive problems from existing data | Fast dataset expansion |
| **v3_seed_with_rag** | Problem RAG | Reference similar problems for quality | Improving existing problems |
| **v4_knowledge_based_rag** | Knowledge RAG | Leverage ArXiv papers via Google Sheets | Production data generation |
| **v5_knowledge_based_rag** ⭐ | Doctoral upgrade | Upgrade existing datasets to PhD level | **Latest & Recommended** |

## 🎓 v5_knowledge_based_rag (Latest & Recommended)

### Overview

v5 transforms existing educational datasets into doctoral-level materials using:
- **Multi-Stage Pipeline**: Problem upgrade → Thinking enhancement → Answer validation → Quality assurance
- **Knowledge Integration**: FAISS-indexed ArXiv papers for theoretical context
- **Proven Methodology**: Based on reference data synthesis flow
- **Local LLM**: vLLM-optimized Qwen3-235B-A22B (no API calls)

### Quick Start

#### Step 1: Knowledge Index Setup

```bash
cd v5_knowledge_based_rag

# Option 1: Use pre-built index (recommended)
# Place FAISS index files in configured path

# Option 2: Build from HuggingFace dataset
python scripts/build_knowledge_index.py \
    --dataset your-org/RAG-dataset \
    --output-dir /path/to/knowledge_indexes
```

#### Step 2: Configuration

```bash
# Copy configuration template
cp config/.env.example config/.env

# Edit configuration (defaults work out of the box)
vim config/.env
```

#### Step 3: Data Generation

```bash
# Basic execution
sbatch scripts/run_generate_data.sh \
    --dataset your-org/seed-data \
    --limit 1000 \
    --batch-size 3

# Full dataset processing
sbatch scripts/run_generate_data.sh \
    --dataset your-org/seed-data

# Multi-node distributed processing
sbatch scripts/run_generate_data.sh \
    --dataset your-org/seed-data \
    --use-dynamic-queue
```

#### Parameters

- `--dataset`: Input seed dataset (HuggingFace)
- `--limit`: Number of items to process
- `--batch-size`: Parallel processing batch size (default: 3)
- `--output-dir`: Custom output directory
- `--use-dynamic-queue`: Enable multi-node distributed processing
- `--no-knowledge-rag`: Disable knowledge RAG (debug only)

### Key Features

1. **Doctoral-Level Upgrade**: Transforms problems into PhD-level challenges
2. **Enhanced Reasoning**: Expands basic solutions into comprehensive theoretical analysis
3. **Knowledge Integration**: Leverages specialized ArXiv papers for context
4. **Multi-Stage Validation**: Automated quality checks and fixes
5. **Fault Tolerance**: Dynamic task queue for distributed processing
6. **Production Ready**: Comprehensive error handling and monitoring

### Data Format

**Input (Seed Data):**
```json
{
  "question": "Original question text",
  "think": "Original thinking process/solution",
  "answer": "Original answer",
  "subject": "Mathematics|Physics|Chemistry|etc",
  "data_id": "unique_identifier"
}
```

**Output (Doctoral-Level):**
```json
{
  "question": "Upgraded doctoral-level question",
  "think": "Enhanced reasoning with rigorous derivations",
  "answer": "Validated answer (MCQ letter or concise result)",
  "subject": "Subject area",
  "data_id": "original_id",
  "answer_type": "multipleChoice|exactMatch",
  "original_question": "Reference to original",
  "original_thinking": "Reference to original",
  "original_answer": "Reference to original",
  "used_knowledge": true,
  "knowledge_docs_count": 5,
  "upgraded_at": "2025-01-19T12:00:00"
}
```

For detailed v5 documentation, see: [v5_knowledge_based_rag/README.md](v5_knowledge_based_rag/README.md)

## 🚀 Execution Examples

### v4_knowledge_based_rag (ArXiv-based)

#### Step 1: ArXiv Collection and Index Building

```bash
# Integrated pipeline (collection + indexing)
./v4_knowledge_based_rag/scripts/run_arxiv_sheets_pipeline.sh \
  "https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/edit?usp=sharing" \
  both

# Individual execution (each component runs independently)
# Collect ArXiv papers to Google Sheets
MAX_PAPERS=100 ./v4_knowledge_based_rag/scripts/run_scraping.sh \
  "https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/edit?usp=sharing"

# Build FAISS index from Google Sheets
./v4_knowledge_based_rag/scripts/run_build_index.sh \
  "https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/edit?usp=sharing"
```

#### Step 2: Data Generation

```bash
# Basic execution
sbatch v4_knowledge_based_rag/scripts/run_generate_data.sh

# With parameters
sbatch v4_knowledge_based_rag/scripts/run_generate_data.sh \
  --limit 100 \
  --problems-per-seed 5

# Specific node
sbatch --nodelist=osk-gpu63 v4_knowledge_based_rag/scripts/run_generate_data.sh
```

### v3_seed_with_rag (Problem-based RAG)

```bash
# Basic execution
sbatch v3_seed_with_rag/run_generate_data.sh

# Custom parameters
sbatch v3_seed_with_rag/run_generate_data.sh \
  --limit 100 \
  --problems-per-seed 5
```

### v2_seed_no_rag (Fast generation)

```bash
# Basic execution
sbatch v2_seed_no_rag/run_generate_data.sh

# Large-scale generation
sbatch v2_seed_no_rag/run_generate_data.sh \
  --limit 1000 \
  --problems-per-seed 10
```

### v1_no_seed (From scratch)

```bash
# Basic execution
sbatch v1_no_seed/run_generate_data.sh

# Specify number of problems
sbatch v1_no_seed/run_generate_data.sh --num-iterations 100
```

## 📄 Output Files

**📁 Shared Data Storage**: `/home/Competition2025/P05/shareP05/data_generation/data_generation_output/`

| File | Content | Location |
|------|---------|----------|
| `instruction_dataset.jsonl` | Main dataset | `/data_generation_output/v*/YYYYMMDD_HHMMSS/` |
| `generation_log.jsonl` | Generation logs | Same directory |
| `generation_stats.json` | Statistics | Same directory |
| `generation_progress.json` | Progress info | Same directory |
| SLURM logs | Execution logs/errors | `/data_generation_output/logs/` |
| Knowledge indexes | FAISS indexes | `/shareP05/knowledge_indexes/` |

## 🔍 Troubleshooting

### GPU/CUDA Issues

```bash
# Check GPU status
nvidia-smi

# Check CUDA
echo $CUDA_VISIBLE_DEVICES

# Test vLLM
python -c "from vllm import LLM; print('vLLM OK')"
```

### Out of Memory

```bash
# Adjust settings
export GPU_MEMORY_UTILIZATION=0.8  # 0.95 → 0.8
export MAX_MODEL_LEN=8192          # 16384 → 8192
export NUM_COT_CANDIDATES=3        # 5 → 3 (v4)
export BATCH_SIZE=2                # 3 → 2 (v5)
```

### Process Management

```bash
# Check running jobs
squeue -u $USER

# Cancel job
scancel <job_id>

# Monitor logs
tail -f /home/Competition2025/P05/shareP05/data_generation/data_generation_output/logs/generate_data-*.out
```

## 📈 Performance Benchmarks

### Processing Speed (Qwen3-30B-A3B, 8 GPUs, single node for v1-v4)

| Dataset Size | v1_no_seed | v2_seed_no_rag | v3_seed_with_rag | v4_knowledge_based_rag |
|--------------|-----------|---------------|------------------|------------------------|
| 100 problems | 20 min | 30 min | 40 min | 35 min |
| 1000 problems | 3 hours | 5 hours | 6 hours | 5.5 hours |
| 5000 problems | 15 hours | 24 hours | 30 hours | 28 hours |

### Processing Speed (Qwen3-235B-A22B, 8 GPUs for v5)

| Dataset Size | v5_knowledge_based_rag (Doctoral Upgrade) |
|--------------|------------------------------------------|
| 100 items | ~2 hours |
| 1000 items | ~18 hours |
| 5000 items | ~90 hours |

*Note: v4 times exclude initial index building (~1-2 hours). v5 uses larger model.*

### Memory Management

#### Qwen3-30B-A3B (v1-v4)
```python
TENSOR_PARALLEL_SIZE=8      # 8-GPU parallel
GPU_MEMORY_UTILIZATION=0.95 # 95% utilization
DTYPE=bfloat16              # BF16 precision
MAX_MODEL_LEN=16384         # Context length
```

#### Qwen3-235B-A22B (v5)
```python
TENSOR_PARALLEL_SIZE=8      # 8-GPU parallel
GPU_MEMORY_UTILIZATION=0.95 # 95% utilization
DTYPE=bfloat16              # BF16 precision
MAX_MODEL_LEN=131072        # Longer context
BATCH_SIZE=3                # Parallel items
```

## 📊 Log Monitoring

```bash
# SLURM job logs
tail -f /home/Competition2025/P05/shareP05/data_generation/data_generation_output/logs/generate_data-*.out

# Generation logs (v4)
tail -f /home/Competition2025/P05/shareP05/data_generation/data_generation_output/v4_knowledge_based/*/generation_log.jsonl | jq '.'

# Generation logs (v5)
tail -f /home/Competition2025/P05/shareP05/data_generation/data_generation_output/v5_doctoral_upgrade/*/instruction_dataset.jsonl | jq '.'

# Progress monitoring
watch -n 5 'cat /path/to/output/generation_progress.json | jq'
```

## ⚠️ System Requirements

1. **GPU Requirements**: NVIDIA H100 (8 GPUs recommended)
2. **Memory Requirements**: System RAM 512GB+ (v1-v4), 1TB+ (v5)
3. **Disk Space**: Sufficient space for output (~1GB per 1000 problems)
4. **CUDA**: 12.8 or higher

## 🆕 Latest Features and Changes

### v5 Knowledge-Based RAG (Latest)
- **Doctoral-Level Upgrade**: Transform existing datasets to PhD quality
- **Multi-Stage Validation**: Comprehensive quality assurance pipeline
- **Reference-Based Flow**: Proven data synthesis methodology
- **Distributed Processing**: Dynamic task queue for multi-node scaling
- **Production Ready**: Comprehensive error handling and monitoring

### Execution Mode Improvements (All Versions)
- **Multiprocessing Execution**: Complete Ray removal, stable vLLM native multiprocessing
- **Simplified Configuration**: No complex node setup required, single-node stable execution
- **Memory Efficiency**: Proper resource cleanup on process termination

### v4 Architecture Improvements
- **Unified Management**: Consolidated from 3 separate spreadsheets to single Google Sheets
- **Direct ArXiv Collection**: Scrape directly from ArXiv HTML pages
- **Independent Execution**: Each component (scraping, indexing, RAG, generation) runs independently
- **CSV Elimination**: No local CSV files, Google Sheets access only
- **Organized Structure**: Folders organized as `data_generation/`, `rag/`, `arxiv/`, `scripts/`, `docs/`
