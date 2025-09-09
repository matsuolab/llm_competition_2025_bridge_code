import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any
import json
import logging
from datetime import datetime

from ..utils.file_utils import FileUtils

logger = logging.getLogger(__name__)


class ResultAggregator:
    """Aggregate and analyze results from multiple dataset evaluations."""
    
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        
    def load_all_results(self) -> List[Dict[str, Any]]:
        """Load all result files from the results directory with path information."""
        result_files = FileUtils.list_result_files(self.results_dir)
        
        if not result_files:
            logger.warning(f"No result files found in {self.results_dir}")
            return []
        
        logger.info(f"Found {len(result_files)} result files")
        results = []
        
        for result_file in result_files:
            try:
                data = FileUtils.load_results_from_file(result_file)
                
                # Parse path information
                path_info = FileUtils.parse_result_file_path(result_file, self.results_dir)
                data['path_info'] = path_info
                
                results.append(data)
                logger.debug(f"Loaded results from {result_file}")
            except Exception as e:
                logger.error(f"Failed to load {result_file}: {e}")
                continue
        
        logger.info(f"Successfully loaded {len(results)} result files")
        return results
    
    def aggregate_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate statistics across all datasets with model-level analysis."""
        if not results:
            return {}
        
        # Group results by dataset and model
        dataset_model_results = {}
        all_models = set()
        all_datasets = set()
        
        for result in results:
            path_info = result.get('path_info', {})
            dataset_name = path_info.get('dataset_name', result['metadata'].get('dataset_name', 'unknown'))
            model_name = path_info.get('model_name', result['metadata'].get('model_name', 'unknown'))
            
            all_models.add(model_name)
            all_datasets.add(dataset_name)
            
            if dataset_name not in dataset_model_results:
                dataset_model_results[dataset_name] = {}
            
            dataset_model_results[dataset_name][model_name] = result
        
        # Calculate statistics by model
        model_performance = {}
        for model_name in all_models:
            model_performance[model_name] = {
                'total_questions': 0,
                'total_correct': 0,
                'datasets': [],
                'overall_accuracy': 0.0
            }
            
            for dataset_name in all_datasets:
                if model_name in dataset_model_results.get(dataset_name, {}):
                    result = dataset_model_results[dataset_name][model_name]
                    overall_stats = result['results']['overall_statistics']
                    
                    questions = overall_stats['total_questions']
                    success_rate = overall_stats['average_success_rate']
                    correct = overall_stats.get('fully_solved_questions', int(questions * success_rate))
                    
                    model_performance[model_name]['total_questions'] += questions
                    model_performance[model_name]['total_correct'] += correct
                    model_performance[model_name]['datasets'].append({
                        'dataset_name': dataset_name,
                        'accuracy': success_rate,
                        'questions': questions,
                        'correct': correct
                    })
            
            # Calculate overall accuracy for this model
            if model_performance[model_name]['total_questions'] > 0:
                model_performance[model_name]['overall_accuracy'] = (
                    model_performance[model_name]['total_correct'] / 
                    model_performance[model_name]['total_questions']
                )
        
        # Calculate dataset summaries
        dataset_summaries = []
        for dataset_name in all_datasets:
            dataset_models = dataset_model_results.get(dataset_name, {})
            if not dataset_models:
                continue
                
            # Use first available model's data for basic stats
            first_result = next(iter(dataset_models.values()))
            overall_stats = first_result['results']['overall_statistics']
            
            # Calculate average performance across models for this dataset
            model_accuracies = []
            for model_name, result in dataset_models.items():
                model_stats = result['results']['overall_statistics']
                model_accuracies.append(model_stats['average_success_rate'])
            
            avg_success_rate = sum(model_accuracies) / len(model_accuracies) if model_accuracies else 0.0
            
            dataset_summaries.append({
                'dataset_name': dataset_name,
                'total_questions': overall_stats['total_questions'],
                'avg_success_rate': avg_success_rate,
                'fully_solved': overall_stats.get('fully_solved_questions', 0),
                'unsolved': overall_stats.get('unsolved_questions', 0),
                'models': list(dataset_models.keys())
            })
        
        # Calculate overall metrics
        total_questions = sum(perf['total_questions'] for perf in model_performance.values())
        total_correct = sum(perf['total_correct'] for perf in model_performance.values())
        overall_success_rate = total_correct / total_questions if total_questions > 0 else 0.0
        
        return {
            'total_datasets': len(all_datasets),
            'total_models': len(all_models),
            'total_questions': total_questions,
            'overall_success_rate': overall_success_rate,
            'dataset_summaries': dataset_summaries,
            'model_performance': model_performance,
            'aggregation_timestamp': datetime.now().isoformat()
        }
    
    def generate_summary_report(self, aggregated_stats: Dict[str, Any]) -> str:
        """Generate a human-readable summary report with model-level analysis."""
        if not aggregated_stats:
            return "No results to aggregate."
        
        report_lines = [
            "# Evaluation Results Summary",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Overall Statistics",
            f"- Total Datasets: {aggregated_stats['total_datasets']}",
            f"- Total Models: {aggregated_stats['total_models']}",
            f"- Total Questions: {aggregated_stats['total_questions']}",
            f"- Overall Success Rate: {aggregated_stats['overall_success_rate']:.1%}",
            ""
        ]
        
        # Model performance overview
        model_perf = aggregated_stats.get('model_performance', {})
        if model_perf:
            report_lines.extend([
                "## Model Performance Overview",
                "| Model | Datasets | Total Questions | Overall Accuracy |",
                "|-------|----------|-----------------|------------------|"
            ])
            
            # Sort models by accuracy
            sorted_models = sorted(model_perf.items(), key=lambda x: x[1]['overall_accuracy'], reverse=True)
            
            for model_name, perf in sorted_models:
                report_lines.append(
                    f"| {model_name} | {len(perf['datasets'])} | {perf['total_questions']} | {perf['overall_accuracy']:.1%} |"
                )
        
        # Dataset performance by model
        report_lines.extend([
            "",
            "## Dataset Performance by Model"
        ])
        
        for dataset in aggregated_stats['dataset_summaries']:
            dataset_name = dataset['dataset_name']
            report_lines.extend([
                "",
                f"### {dataset_name}",
                f"- Total Questions: {dataset['total_questions']}",
                f"- Average Success Rate: {dataset['avg_success_rate']:.1%}",
                "",
                "| Model | Accuracy | Correct | Total |",
                "|-------|----------|---------|-------|"
            ])
            
            # Find performance of each model on this dataset
            for model_name, perf in model_perf.items():
                dataset_perf = next((d for d in perf['datasets'] if d['dataset_name'] == dataset_name), None)
                if dataset_perf:
                    report_lines.append(
                        f"| {model_name} | {dataset_perf['accuracy']:.1%} | "
                        f"{dataset_perf['correct']} | {dataset_perf['questions']} |"
                    )
        
        # Performance highlights
        if model_perf:
            best_model = max(model_perf.items(), key=lambda x: x[1]['overall_accuracy'])
            worst_model = min(model_perf.items(), key=lambda x: x[1]['overall_accuracy'])
            
            report_lines.extend([
                "",
                "## Performance Highlights",
                f"- Best performing model: {best_model[0]} ({best_model[1]['overall_accuracy']:.1%})",
                f"- Worst performing model: {worst_model[0]} ({worst_model[1]['overall_accuracy']:.1%})",
            ])
        
        if aggregated_stats['dataset_summaries']:
            sorted_datasets = sorted(
                aggregated_stats['dataset_summaries'],
                key=lambda x: x['avg_success_rate'],
                reverse=True
            )
            
            report_lines.extend([
                f"- Best performing dataset: {sorted_datasets[0]['dataset_name']} ({sorted_datasets[0]['avg_success_rate']:.1%} avg)",
                f"- Worst performing dataset: {sorted_datasets[-1]['dataset_name']} ({sorted_datasets[-1]['avg_success_rate']:.1%} avg)",
            ])
        
        return "\n".join(report_lines)
    
    def save_aggregated_results(self, aggregated_stats: Dict[str, Any], output_path: Path) -> None:
        """Save aggregated results to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(aggregated_stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved aggregated results to: {output_path}")
    
    def save_summary_report(self, report: str, output_path: Path) -> None:
        """Save summary report to text file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Saved summary report to: {output_path}")


def main():
    """Main entry point for the aggregate command."""
    parser = argparse.ArgumentParser(description='Aggregate evaluation results from multiple datasets')
    parser.add_argument('results_dir', type=str, help='Directory containing result files')
    parser.add_argument('--output', '-o', type=str, default='./aggregated_results',
                       help='Output directory for aggregated results (default: ./aggregated_results)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        results_dir = Path(args.results_dir)
        output_dir = Path(args.output)
        
        if not results_dir.exists():
            logger.error(f"Results directory does not exist: {results_dir}")
            sys.exit(1)
        
        logger.info(f"Aggregating results from: {results_dir}")
        
        # Create aggregator and load results
        aggregator = ResultAggregator(results_dir)
        results = aggregator.load_all_results()
        
        if not results:
            logger.error("No results found to aggregate")
            sys.exit(1)
        
        # Aggregate statistics
        aggregated_stats = aggregator.aggregate_statistics(results)
        
        # Generate summary report
        summary_report = aggregator.generate_summary_report(aggregated_stats)
        
        # Save outputs
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        json_output = output_dir / 'aggregated_results.json'
        aggregator.save_aggregated_results(aggregated_stats, json_output)
        
        # Save text summary
        summary_output = output_dir / 'summary_report.md'
        aggregator.save_summary_report(summary_report, summary_output)
        
        # Print summary to console
        print("\n" + summary_report)
        
        logger.info(f"Aggregation complete. Results saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Error during aggregation: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()