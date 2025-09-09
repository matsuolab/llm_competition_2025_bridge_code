import argparse
import sys
from pathlib import Path
from typing import List, Optional
import logging

from ..core.multi_dataset_processor import MultiDatasetProcessor
from ..utils.config import Config
from ..utils.file_utils import FileUtils


def main():
    parser = argparse.ArgumentParser(description='Filter solvable questions using LLM evaluation')
    
    # Config and datasets
    parser.add_argument('--config', '-c', type=str, help='Path to YAML configuration file')
    parser.add_argument('--datasets', nargs='+', help='Dataset repository names to process')
    parser.add_argument('dataset_repo', nargs='?', help='Single dataset repository (for backward compatibility)')
    
    # Model and evaluation options
    parser.add_argument('--models', nargs='+', help='List of model names to evaluate')
    parser.add_argument('--limit', type=int, help='Limit number of questions to process')
    
    # Output options
    parser.add_argument('--output-dir', '-o', default='', 
                       help='Output directory for results')
    
    # vLLM options
    parser.add_argument('--tensor-parallel-size', type=int, help='Tensor parallel size for vLLM')
    parser.add_argument('--gpu-memory-utilization', type=float, help='GPU memory utilization (0-1)')
    parser.add_argument('--batch-size', type=int, help='Batch size for processing')
    
    # Other options
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    # Legacy compatibility
    parser.add_argument('--multi-model', action='store_true', 
                       help='Legacy option (now always enabled)')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        if args.config:
            config_path = Path(args.config)
            if not config_path.exists():
                logger.error(f"Config file not found: {config_path}")
                sys.exit(1)
            config = Config.from_yaml(config_path)
            logger.info(f"Loaded config from: {config_path}")
        else:
            config = Config.load_default()
            logger.info("Using default configuration")
        
        # Apply command-line overrides
        overrides = {}
        
        # Dataset configuration
        # Support both new and legacy dataset configuration
        datasets = args.datasets or ([args.dataset_repo] if args.dataset_repo else config.get_dataset_repositories())
        if not datasets:
            logger.error("No datasets specified. Use --datasets or provide dataset_repo argument.")
            sys.exit(1)
        if args.datasets:
            # For CLI override, use legacy structure for backward compatibility
            overrides['dataset_repositories'] = datasets
        
        # Model configuration
        if args.models:
            overrides['models'] = args.models
        
        # Evaluation configuration
        eval_overrides = {}
        if args.limit:
            eval_overrides['max_questions'] = args.limit
        if args.batch_size:
            eval_overrides['batch_size'] = args.batch_size
        if eval_overrides:
            overrides['evaluation'] = eval_overrides
        
        # vLLM configuration
        vllm_overrides = {}
        if args.tensor_parallel_size:
            vllm_overrides['tensor_parallel_size'] = args.tensor_parallel_size
        if args.gpu_memory_utilization:
            vllm_overrides['gpu_memory_utilization'] = args.gpu_memory_utilization
        if vllm_overrides:
            overrides['vllm'] = vllm_overrides
        
        if args.output_dir:
            overrides['output'] = {'base_dir': args.output_dir}
        # Merge overrides
        if overrides:
                    # Output configuration
            config = config.merge(overrides)
        
        # Setup logging from config
        config.setup_logging()
        
        logger.info(f"Processing {len(datasets)} datasets with {len(config.models)} models")
        logger.info(f"Models: {config.models}")
        logger.info(f"Datasets: {datasets}")
        logger.info(f"Output directory: {config.output.base_dir}")
        
        # Process datasets
        processor = MultiDatasetProcessor(config)
        all_results = processor.process_multiple_datasets(datasets)
        
        if not all_results:
            logger.error("No results generated")
            sys.exit(1)
        
        # Save results
        saved_paths = processor.save_all_results(all_results)
        
        # Display summary statistics
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        total_questions = 0
        total_success_rate = 0.0
        
        for result in all_results:
            dataset_name = result['dataset_name']
            stats = result['results']['overall_statistics']
            
            print(f"\n{dataset_name}:")
            print(f"  Questions: {stats['total_questions']}")
            print(f"  Success Rate: {stats['average_success_rate']:.1%}")
            print(f"  Fully Solved: {stats['fully_solved_questions']}")
            print(f"  Unsolved: {stats['unsolved_questions']}")
            
            total_questions += stats['total_questions']
            total_success_rate += stats['average_success_rate'] * stats['total_questions']
        
        if total_questions > 0:
            overall_success_rate = total_success_rate / total_questions
            print(f"\nOVERALL:")
            print(f"  Total Questions: {total_questions}")
            print(f"  Overall Success Rate: {overall_success_rate:.1%}")
        
        print(f"\nResults saved to:")
        for path in saved_paths:
            print(f"  {path}")
        
        print(f"\nTo aggregate results, run:")
        print(f"  aggregate {config.output.base_dir}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
