import pytest
import tempfile
from pathlib import Path
from filter_solvable_question.utils.config import Config, DatasetMapping, DatasetConfig


class TestExtendedConfig:
    def test_dataset_mapping_defaults(self):
        """Test DatasetMapping default values."""
        mapping = DatasetMapping()
        assert mapping.question_field == "question"
        assert mapping.answer_field == "answer"
    
    def test_dataset_mapping_custom(self):
        """Test DatasetMapping with custom values."""
        mapping = DatasetMapping(question_field="problem", answer_field="solution")
        assert mapping.question_field == "problem"
        assert mapping.answer_field == "solution"
    
    def test_dataset_config_defaults(self):
        """Test DatasetConfig default values."""
        config = DatasetConfig()
        assert config.default_mappings.question_field == "question"
        assert config.default_mappings.answer_field == "answer"
        assert len(config.dataset_specific) == 0
        assert len(config.repositories) == 0
    
    def test_config_get_field_mapping_default(self):
        """Test getting default field mapping."""
        config = Config()
        mapping = config.get_field_mapping("any_dataset")
        assert mapping.question_field == "question"
        assert mapping.answer_field == "answer"
    
    def test_config_get_field_mapping_specific(self):
        """Test getting dataset-specific field mapping."""
        config_dict = {
            'datasets': {
                'default_mappings': {
                    'question_field': 'question',
                    'answer_field': 'answer'
                },
                'dataset_specific': {
                    'custom_dataset': {
                        'question_field': 'problem',
                        'answer_field': 'solution'
                    }
                }
            }
        }
        config = Config.from_dict(config_dict)
        
        # Test default mapping
        default_mapping = config.get_field_mapping("other_dataset")
        assert default_mapping.question_field == "question"
        assert default_mapping.answer_field == "answer"
        
        # Test specific mapping
        specific_mapping = config.get_field_mapping("custom_dataset")
        assert specific_mapping.question_field == "problem"
        assert specific_mapping.answer_field == "solution"
    
    def test_config_get_dataset_repositories_new(self):
        """Test getting dataset repositories from new config structure."""
        config_dict = {
            'datasets': {
                'repositories': ['repo1', 'repo2']
            }
        }
        config = Config.from_dict(config_dict)
        repos = config.get_dataset_repositories()
        assert repos == ['repo1', 'repo2']
    
    def test_config_get_dataset_repositories_legacy(self):
        """Test getting dataset repositories from legacy config structure."""
        config_dict = {
            'dataset_repositories': ['legacy_repo1', 'legacy_repo2']
        }
        config = Config.from_dict(config_dict)
        repos = config.get_dataset_repositories()
        assert repos == ['legacy_repo1', 'legacy_repo2']
    
    def test_config_get_dataset_repositories_priority(self):
        """Test that new structure takes priority over legacy."""
        config_dict = {
            'datasets': {
                'repositories': ['new_repo']
            },
            'dataset_repositories': ['legacy_repo']
        }
        config = Config.from_dict(config_dict)
        repos = config.get_dataset_repositories()
        assert repos == ['new_repo']
    
    def test_output_config_save_per_dataset(self):
        """Test output config with save_per_dataset option."""
        config_dict = {
            'output': {
                'save_per_dataset': True
            }
        }
        config = Config.from_dict(config_dict)
        assert config.output.save_per_dataset is True
    
    def test_config_from_yaml_with_new_structure(self):
        """Test loading config with new dataset structure from YAML."""
        yaml_content = """
        datasets:
          default_mappings:
            question_field: "question"
            answer_field: "answer"
          dataset_specific:
            math_dataset:
              question_field: "problem"
              answer_field: "solution"
          repositories:
            - "repo1"
            - "repo2"
        output:
          save_per_dataset: true
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            yaml_path = Path(f.name)
        
        try:
            config = Config.from_yaml(yaml_path)
            assert config.datasets.repositories == ["repo1", "repo2"]
            assert config.output.save_per_dataset is True
            
            # Test field mappings
            default_mapping = config.get_field_mapping("other")
            assert default_mapping.question_field == "question"
            
            specific_mapping = config.get_field_mapping("math_dataset")
            assert specific_mapping.question_field == "problem"
            assert specific_mapping.answer_field == "solution"
            
        finally:
            yaml_path.unlink()  # Clean up