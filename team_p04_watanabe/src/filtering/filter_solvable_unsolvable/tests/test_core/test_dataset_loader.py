import pytest
from unittest.mock import patch, MagicMock
from filter_solvable_question.core.dataset_loader import DatasetLoader
from filter_solvable_question.utils.config import DatasetMapping


MODULE_UNDER_TEST = "filter_solvable_question.core.dataset_loader"

@patch(f'{MODULE_UNDER_TEST}.load_dataset')
class TestDatasetLoader:
    target_dataset = "test/dataset"
    
    def test_init_with_repository_name(self, mock_load):
        loader = DatasetLoader(self.target_dataset)
        assert loader.repository_name == self.target_dataset
    
    def test_load_dataset_returns_dataset_with_question_answer_columns(self, mock_load):
        # Create actual list instead of MagicMock
        train_data = [
            {'question': 'What is 2+2?', 'answer': 'think>...\n#### 4'},
            {'question': 'What is 3+3?', 'answer': 'think>...\n#### 6'}
        ]
        mock_dataset = {'train': train_data}
        mock_load.return_value = mock_dataset
        
        loader = DatasetLoader(self.target_dataset)
        result = loader.load_dataset()
        
        mock_load.assert_called_once_with(self.target_dataset)
        assert len(result) == 2
        assert result[0]['question'] == 'What is 2+2?'
        assert result[0]['answer'] == 'think>...\n#### 4'
        assert 'original_item' in result[0]
    
    def test_extract_correct_answer_from_answer_field(self, mock_load):
        loader = DatasetLoader(self.target_dataset)
        answer_text = "think>\nSome reasoning here\n#### 4"
        
        result = loader.extract_correct_answer(answer_text)
        
        assert result == "4"
    
    def test_extract_correct_answer_with_complex_answer(self, mock_load):
        loader = DatasetLoader(self.target_dataset)
        answer_text = "think>\nComplex reasoning\n#### \\(\\frac{C_{n_1}^{a_1} \\cdot C_{n_2}^{a_2}}{C_N^A}\\)"
        
        result = loader.extract_correct_answer(answer_text)
        
        assert result == "\\(\\frac{C_{n_1}^{a_1} \\cdot C_{n_2}^{a_2}}{C_N^A}\\)"
    
    def test_extract_correct_answer_no_separator_returns_empty(self, mock_load):
        loader = DatasetLoader(self.target_dataset)
        answer_text = "No separator in this text"
        
        result = loader.extract_correct_answer(answer_text)
        
        assert result == ""


@patch(f'{MODULE_UNDER_TEST}.load_dataset')
class TestDatasetLoaderWithFieldMapping:
    target_dataset = "test/dataset"
    
    def test_init_with_field_mapping(self, mock_load):
        mapping = DatasetMapping(question_field="problem", answer_field="solution")
        loader = DatasetLoader(self.target_dataset, mapping)
        assert loader.repository_name == self.target_dataset
        assert loader.field_mapping.question_field == "problem"
        assert loader.field_mapping.answer_field == "solution"
    
    def test_load_dataset_with_custom_fields(self, mock_load):
        train_data = [
            {'problem': 'What is 2+2?', 'solution': 'think>...\n#### 4', 'extra': 'data'},
            {'problem': 'What is 3+3?', 'solution': 'think>...\n#### 6', 'extra': 'more'}
        ]
        mock_dataset = {'train': train_data}
        mock_load.return_value = mock_dataset
        
        mapping = DatasetMapping(question_field="problem", answer_field="solution")
        loader = DatasetLoader(self.target_dataset, mapping)
        result = loader.load_dataset()
        
        assert len(result) == 2
        assert result[0]['question'] == 'What is 2+2?'  # Normalized
        assert result[0]['answer'] == 'think>...\n#### 4'  # Normalized
        assert result[0]['original_item']['extra'] == 'data'  # Original preserved
    
    def test_load_dataset_missing_required_field(self, mock_load):
        train_data = [
            {'problem': 'What is 2+2?'},  # Missing solution field
            {'solution': 'answer'},       # Missing problem field
            {'problem': 'What is 3+3?', 'solution': 'think>...\n#### 6'}  # Complete
        ]
        mock_dataset = {'train': train_data}
        mock_load.return_value = mock_dataset
        
        mapping = DatasetMapping(question_field="problem", answer_field="solution")
        loader = DatasetLoader(self.target_dataset, mapping)
        result = loader.load_dataset()
        
        # Only the complete item should be included
        assert len(result) == 1
        assert result[0]['question'] == 'What is 3+3?'
    
    def test_get_available_fields(self, mock_load):
        # Create actual data for get_available_fields test
        train_data = [
            {'problem': 'P1', 'solution': 'S1', 'difficulty': 'easy'}
        ]
        mock_dataset = {'train': train_data}
        mock_load.return_value = mock_dataset
        
        loader = DatasetLoader(self.target_dataset)
        fields = loader.get_available_fields()
        
        assert set(fields) == {'problem', 'solution', 'difficulty'}