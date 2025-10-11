#!/bin/bash
#SBATCH --job-name=build_knowledge_index
#SBATCH --output=/home/Competition2025/P05/shareP05/data_generation/data_generation_output/logs/build_index_%j.out
#SBATCH --error=/home/Competition2025/P05/shareP05/data_generation/data_generation_output/logs/build_index_%j.err
#SBATCH --partition=P05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# Set up environment
echo "Starting knowledge index building at $(date)"
echo "Running on node: $(hostname)"

# Activate conda environment
source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh
conda activate /home/Competition2025/P05/shareP05/data_generation/data_generation_env

# Set environment variables for GPU usage
export CUDA_VISIBLE_DEVICES=0
export DEVICE=cuda

# Change to script directory
cd /home/Competition2025/P05/P05U001/team_suzuki/data/data_generation/sft_data_generation/v5_knowledge_based_rag

# Run the index builder
echo "Building knowledge index from team-suzuki/RAG_0915..."
python scripts/build_knowledge_index.py --config config/.env

# Check if successful
if [ $? -eq 0 ]; then
    echo "Knowledge index built successfully!"

    # Check the output
    echo "Checking generated files:"
    ls -lh /home/Competition2025/P05/shareP05/data_generation/knowledge_indexes/

    # Show statistics
    echo "Extracting statistics from metadata..."
    python -c "
import json
with open('/home/Competition2025/P05/shareP05/data_generation/knowledge_indexes/knowledge_metadata.json', 'r') as f:
    data = json.load(f)
    stats = data.get('stats', {})
    print(f'Total documents: {stats.get(\"total_documents\", 0)}')
    print(f'Math documents: {stats.get(\"math_documents\", 0)}')
    print(f'Subject distribution:')
    for subject, count in sorted(stats.get('subjects_distribution', {}).items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f'  {subject}: {count}')
"
else
    echo "Knowledge index building failed!"
    exit 1
fi

echo "Completed at $(date)"