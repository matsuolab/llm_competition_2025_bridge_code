from typing import List, Dict, Any
from pathlib import Path
import logging
from tqdm import tqdm

from .dataset_loader import DatasetLoader
from .llm_solver import LLMSolver
from .answer_evaluator import AnswerEvaluator
from .result_classifier import ResultClassifier
from ..utils.config import Config
from ..utils.file_utils import FileUtils
from ..utils.response_parser import ResponseParser

logger = logging.getLogger(__name__)


class MultiDatasetProcessor:
    """Process multiple datasets with multiple models."""
    
    def __init__(self, config: Config):
        self.config = config
        self.evaluators = {}  # Cache for model evaluators
        self.answer_evaluator = AnswerEvaluator()
        self.result_classifier = ResultClassifier()
        self.response_parser = ResponseParser()
    
    def get_or_create_evaluator(self, model_name: str) -> LLMSolver:
        """Get or create LLM evaluator for a model."""
        if model_name not in self.evaluators:
            logger.info(f"Creating evaluator for model: {model_name}")
            evaluator = LLMSolver(model_name, self.config.vllm)
            evaluator.initialize_model()
            self.evaluators[model_name] = evaluator
        return self.evaluators[model_name]
    
    def process_single_dataset(self, dataset_repo: str) -> Dict[str, Any]:
        """
        Process a single dataset with all configured models.
        
        Args:
            dataset_repo: Repository name for the dataset
            
        Returns:
            Dictionary containing results and metadata
        """
        # Extract category from base_dir to remove prefix from dataset name
        base_dir_path = Path(self.config.output.base_dir)
        base_category = base_dir_path.name if base_dir_path.name != 'results' else None
        
        dataset_name = FileUtils.extract_dataset_name(dataset_repo, base_category)
        logger.info(f"Processing dataset: {dataset_repo} -> {dataset_name}")
        
        # Load dataset with field mapping
        field_mapping = self.config.get_field_mapping(dataset_name)
        loader = DatasetLoader(dataset_repo, field_mapping)
        dataset = loader.load_dataset()
        
        # Apply question limit if configured
        if self.config.evaluation.max_questions:
            dataset = dataset[:self.config.evaluation.max_questions]
            logger.info(f"Limited to {len(dataset)} questions")
        
        # Extract questions and correct answers
        questions = [item['question'] for item in dataset]
        answer_texts = [item['answer'] for item in dataset]
        correct_answers = [loader.extract_correct_answer(answer) for answer in answer_texts]
        
        # Process with all models
        model_results = {}
        question_evaluations = {i: [] for i in range(len(questions))}
        
        for model_name in tqdm(self.config.models, desc="Processing models"):
            logger.info(f"Processing with model: {model_name}")
            
            # Get evaluator
            evaluator = self.get_or_create_evaluator(model_name)
            
            # Process in batches
            batch_size = self.config.evaluation.batch_size
            llm_responses = []
            
            # for i in tqdm(range(0, len(questions), batch_size), desc="Processing batches"):
            #     batch_questions = questions[i:i+batch_size]
            #     batch_responses = evaluator.solve_questions_batch(batch_questions)
            #     llm_responses.extend(batch_responses)

            llm_responses = evaluator.solve_questions_batch(questions)
            
            # Parse responses into structured format
            parsed_responses = self.response_parser.parse_batch(llm_responses)
            
            # Evaluate responses
            evaluations = self.answer_evaluator.evaluate_batch(llm_responses, correct_answers)
            model_results[model_name] = {
                'evaluations': evaluations,
                'responses': parsed_responses  # Always save parsed responses
            }
            
            # Update question evaluations
            for i, evaluation in enumerate(evaluations):
                question_evaluations[i].append(evaluation)
        
        # Calculate statistics
        results = self._calculate_dataset_statistics(
            questions, correct_answers, question_evaluations, model_results
        )
        
        # Add metadata
        metadata = {
            'dataset_repo': dataset_repo,
            'dataset_name': dataset_name,
            'models': self.config.models,
            'total_questions': len(questions),
            'config': self.config.dict()
        }
        
        return {
            'dataset_name': dataset_name,
            'results': results,
            'metadata': metadata
        }
    
    def _calculate_dataset_statistics(
        self,
        questions: List[str],
        correct_answers: List[str],
        question_evaluations: Dict[int, List[bool]],
        model_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate comprehensive statistics for a dataset."""
        
        # Question-level statistics
        question_stats = []
        for i, (question, correct_answer) in enumerate(zip(questions, correct_answers)):
            evaluations = question_evaluations[i]
            success_count = sum(evaluations)
            total_models = len(evaluations)
            success_rate = success_count / total_models if total_models > 0 else 0.0
            
            # Collect model responses for this question
            model_responses = {}
            for model_name, model_data in model_results.items():
                if model_data.get('responses') and i < len(model_data['responses']):
                    model_responses[model_name] = model_data['responses'][i]
            
            question_stats.append({
                'question_index': i,
                'question': question,
                'correct_answer': correct_answer,
                'success_count': success_count,
                'total_models': total_models,
                'success_rate': success_rate,
                'model_evaluations': evaluations,
                'model_responses': model_responses
            })
        
        # Model-level statistics
        model_stats = {}
        for model_name, model_data in model_results.items():
            evaluations = model_data['evaluations']
            total = len(evaluations)
            correct = sum(evaluations)
            accuracy = correct / total if total > 0 else 0.0
            
            model_stats[model_name] = {
                'total_questions': total,
                'correct_answers': correct,
                'accuracy': accuracy
            }
        
        # Overall statistics
        success_rates = [q['success_rate'] for q in question_stats]
        overall_stats = {
            'total_questions': len(questions),
            'total_models': len(self.config.models),
            'average_success_rate': sum(success_rates) / len(success_rates) if success_rates else 0.0,
            'fully_solved_questions': sum(1 for rate in success_rates if rate == 1.0),
            'unsolved_questions': sum(1 for rate in success_rates if rate == 0.0),
            'partially_solved_questions': sum(1 for rate in success_rates if 0 < rate < 1.0)
        }
        
        return {
            'question_statistics': question_stats,
            'model_statistics': model_stats,
            'overall_statistics': overall_stats
        }
    
    def process_multiple_datasets(self, datasets: List[str] = None) -> List[Dict[str, Any]]:
        """
        Process multiple datasets with optional individual saving.
        
        Args:
            datasets: List of dataset repositories. If None, use config datasets
            
        Returns:
            List of results for each dataset
        """
        # Get datasets from new or legacy config structure
        if datasets is None:
            datasets = self.config.get_dataset_repositories()
            
        if not datasets:
            raise ValueError("No datasets specified in config or arguments")
        
        logger.info(f"Processing {len(datasets)} datasets")
        results = []
        
        for dataset_repo in tqdm(datasets, desc="Processing datasets"):
            try:
                result = self.process_single_dataset(dataset_repo)
                results.append(result)
                
                # Save immediately if per-dataset saving is enabled
                if self.config.output.save_per_dataset:
                    self._save_single_dataset_result(result)
                    logger.info(f"Saved result for dataset: {result['dataset_name']}")
                    
            except Exception as e:
                logger.error(f"Failed to process dataset {dataset_repo}: {e}")
                # Continue with other datasets
                continue
        
        logger.info(f"Successfully processed {len(results)} datasets")
        return results
    
    def _save_single_dataset_result(self, result: Dict[str, Any]) -> List[Path]:
        """
        Save a single dataset result immediately.
        
        Args:
            result: Dataset result dictionary
            
        Returns:
            List of saved file paths
        """
        base_dir = Path(self.config.output.base_dir)
        saved_paths = []
        
        dataset_name = result['dataset_name']
        dataset_results = result['results']
        metadata = result['metadata']
        models = metadata.get('models', [])
        
        # Save results for each model separately
        for model_name in models:
            try:
                # Extract model-specific results
                model_results = self._extract_model_results(dataset_results, model_name)
                model_metadata = metadata.copy()
                model_metadata['model_name'] = model_name
                model_metadata['models'] = [model_name]  # Single model for this file
                
                # Create model-specific output directory
                output_dir = FileUtils.create_dataset_output_dir(base_dir, dataset_name, model_name)
                
                # Save results with metadata
                saved_path = FileUtils.save_results_with_metadata(
                    output_dir, dataset_name, model_results, model_metadata
                )
                saved_paths.append(saved_path)
                
            except Exception as e:
                logger.error(f"Failed to save results for {dataset_name}/{model_name}: {e}")
                continue
        
        return saved_paths
    
    def save_all_results(self, results: List[Dict[str, Any]]) -> List[Path]:
        """
        Save all results to individual model directories within dataset directories.
        
        Args:
            results: List of dataset results
            
        Returns:
            List of saved file paths
        """
        base_dir = Path(self.config.output.base_dir)
        saved_paths = []
        
        for result in results:
            dataset_name = result['dataset_name']
            dataset_results = result['results']
            metadata = result['metadata']
            models = metadata.get('models', [])
            
            # Save results for each model separately
            for model_name in models:
                # Extract model-specific results
                model_results = self._extract_model_results(dataset_results, model_name)
                model_metadata = metadata.copy()
                model_metadata['model_name'] = model_name
                model_metadata['models'] = [model_name]  # Single model for this file
                
                # Create model-specific output directory
                output_dir = FileUtils.create_dataset_output_dir(base_dir, dataset_name, model_name)
                
                # Save results with metadata
                saved_path = FileUtils.save_results_with_metadata(
                    output_dir, dataset_name, model_results, model_metadata
                )
                saved_paths.append(saved_path)
        
        return saved_paths
    
    def _extract_model_results(self, dataset_results: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """
        Extract results for a specific model from dataset results.
        
        Args:
            dataset_results: Full dataset results
            model_name: Name of the model to extract
            
        Returns:
            Results filtered for the specific model
        """
        model_results = dataset_results.copy()
        
        # Extract model-specific statistics
        if 'model_statistics' in model_results:
            if model_name in model_results['model_statistics']:
                model_results['model_statistics'] = {
                    model_name: model_results['model_statistics'][model_name]
                }
            else:
                model_results['model_statistics'] = {}
        
        # Update question statistics to include only this model's results
        if 'question_statistics' in model_results:
            updated_questions = []
            for q_stat in model_results['question_statistics']:
                # Find the model's evaluation for this question
                model_evaluations = q_stat.get('model_evaluations', [])
                if len(model_evaluations) >= len(self.config.models):
                    model_index = self.config.models.index(model_name) if model_name in self.config.models else 0
                    if model_index < len(model_evaluations):
                        q_stat_copy = q_stat.copy()
                        q_stat_copy['model_evaluations'] = [model_evaluations[model_index]]
                        q_stat_copy['success_count'] = 1 if model_evaluations[model_index] else 0
                        q_stat_copy['total_models'] = 1
                        q_stat_copy['success_rate'] = 1.0 if model_evaluations[model_index] else 0.0
                        updated_questions.append(q_stat_copy)
            model_results['question_statistics'] = updated_questions
        
        # Update overall statistics for single model
        if 'overall_statistics' in model_results and model_name in dataset_results.get('model_statistics', {}):
            model_stats = dataset_results['model_statistics'][model_name]
            model_results['overall_statistics'] = {
                'total_questions': model_stats['total_questions'],
                'average_success_rate': model_stats['accuracy'],
                'fully_solved_questions': model_stats['correct_answers'],
                'unsolved_questions': model_stats['total_questions'] - model_stats['correct_answers'],
                'partially_solved_questions': 0
            }
        
        return model_results