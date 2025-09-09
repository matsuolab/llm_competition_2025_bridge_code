import pytest
import tempfile
import json
from pathlib import Path
from filter_solvable_question.utils.file_utils import FileUtils


class TestFileUtils:
    def test_extract_dataset_name_from_repo(self):
        repo_name = "org/dataset-name"
        expected = "dataset-name"
        
        result = FileUtils.extract_dataset_name(repo_name)
        
        assert result == expected
    
    def test_extract_dataset_name_simple(self):
        repo_name = "simple-dataset"
        expected = "simple-dataset"
        
        result = FileUtils.extract_dataset_name(repo_name)
        
        assert result == expected
        
    def test_extract_dataset_name_with_category_prefix(self):
        # Test removing category prefix when specified
        assert FileUtils.extract_dataset_name("math_s1K-1.1_preprocess", "math") == "s1K-1.1_preprocess"
        assert FileUtils.extract_dataset_name("org/math_dataset", "math") == "dataset"
        
        # Test when no category prefix to remove
        assert FileUtils.extract_dataset_name("science_dataset", "math") == "science_dataset"
        
        # Test when no base_category specified
        assert FileUtils.extract_dataset_name("math_dataset") == "math_dataset"
    
    def test_create_dataset_output_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            dataset_name = "test-dataset"
            
            output_dir = FileUtils.create_dataset_output_dir(base_dir, dataset_name)
            
            assert output_dir.exists()
            assert output_dir.is_dir()
            assert output_dir.name == dataset_name
            assert output_dir.parent == base_dir
    
    def test_save_results_with_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            dataset_name = "test-dataset"
            results = {"test": "data"}
            metadata = {"model": "test-model", "timestamp": "2024-01-01"}
            
            saved_path = FileUtils.save_results_with_metadata(
                output_dir, dataset_name, results, metadata
            )
            
            assert saved_path.exists()
            
            # Check saved content
            with open(saved_path, 'r') as f:
                saved_data = json.load(f)
            
            assert saved_data['results'] == results
            assert saved_data['metadata'] == metadata
    
    def test_list_result_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            
            # Create test files
            (base_dir / "dataset1").mkdir()
            (base_dir / "dataset1" / "results.json").write_text('{"test": 1}')
            (base_dir / "dataset2").mkdir() 
            (base_dir / "dataset2" / "results.json").write_text('{"test": 2}')
            
            result_files = FileUtils.list_result_files(base_dir)
            
            assert len(result_files) == 2
            assert any("dataset1" in str(f) for f in result_files)
            assert any("dataset2" in str(f) for f in result_files)
    
    def test_load_results_from_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.json"
            test_data = {"results": {"test": "data"}, "metadata": {"model": "test"}}
            
            with open(test_file, 'w') as f:
                json.dump(test_data, f)
            
            loaded_data = FileUtils.load_results_from_file(test_file)
            
            assert loaded_data == test_data