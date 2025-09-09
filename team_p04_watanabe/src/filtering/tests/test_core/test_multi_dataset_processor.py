import pytest
from unittest.mock import patch, MagicMock
from filter_solvable_question.core.multi_dataset_processor import MultiDatasetProcessor
from filter_solvable_question.utils.config import Config


class TestMultiDatasetProcessor:
    def test_init_with_config(self):
        config = Config.load_default()
        processor = MultiDatasetProcessor(config)
        
        assert processor.config == config
        assert len(processor.evaluators) == 0
    
    @patch('filter_solvable_question.core.multi_dataset_processor.DatasetLoader')
    @patch('filter_solvable_question.core.multi_dataset_processor.LLMSolver')
    def test_process_single_dataset(self, mock_solver, mock_loader):
        config = Config.load_default()
        processor = MultiDatasetProcessor(config)
        
        # Mock dataset loader
        mock_loader_instance = MagicMock()
        mock_loader_instance.load_dataset.return_value = [
            {'question': 'Q1', 'answer': 'A1'},
            {'question': 'Q2', 'answer': 'A2'}
        ]
        mock_loader_instance.extract_correct_answer.side_effect = ['A1', 'A2']
        mock_loader.return_value = mock_loader_instance
        
        # Mock LLM solver
        mock_solver_instance = MagicMock()
        mock_solver_instance.solve_questions_batch.return_value = ['R1', 'R2']
        mock_solver.return_value = mock_solver_instance
        
        dataset_repo = "test/dataset"
        results = processor.process_single_dataset(dataset_repo)
        
        assert 'dataset_name' in results
        assert 'results' in results
        assert 'metadata' in results
        assert results['dataset_name'] == 'dataset'
    
    def test_process_multiple_datasets(self):
        config = Config.load_default()
        config.datasets = ['dataset1/repo', 'dataset2/repo']
        processor = MultiDatasetProcessor(config)
        
        with patch.object(processor, 'process_single_dataset') as mock_process:
            mock_process.side_effect = [
                {'dataset_name': 'dataset1', 'results': {'test': 1}},
                {'dataset_name': 'dataset2', 'results': {'test': 2}}
            ]
            
            results = processor.process_multiple_datasets()
            
            assert len(results) == 2
            assert results[0]['dataset_name'] == 'dataset1'
            assert results[1]['dataset_name'] == 'dataset2'
    
    def test_save_all_results(self):
        config = Config.load_default()
        processor = MultiDatasetProcessor(config)
        
        results = [
            {'dataset_name': 'dataset1', 'results': {'test': 1}, 'metadata': {"models": ["model1"]}},
            {'dataset_name': 'dataset2', 'results': {'test': 2}, 'metadata': {"models": ["model1"]}}
        ]
        
        with patch('filter_solvable_question.core.multi_dataset_processor.FileUtils') as mock_utils:
            mock_utils.create_dataset_output_dir.return_value = MagicMock()
            mock_utils.save_results_with_metadata.return_value = MagicMock()
            
            saved_paths = processor.save_all_results(results)
            
            assert len(saved_paths) == 2
            assert mock_utils.create_dataset_output_dir.call_count == 2
            assert mock_utils.save_results_with_metadata.call_count == 2