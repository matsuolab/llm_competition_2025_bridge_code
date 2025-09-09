import pytest
import tempfile
import yaml
from pathlib import Path
from filter_solvable_question.utils.config import Config, VLLMConfig, OutputConfig, EvaluationConfig


class TestConfig:
    def test_config_from_dict(self):
        config_dict = {
            'vllm': {
                'tensor_parallel_size': 2,
                'max_model_len': 1024
            },
            'models': ['model1', 'model2'],
            'datasets': ['dataset1'],
            'output': {
                'base_dir': './test_results',
                'format': 'json'
            }
        }
        
        config = Config.from_dict(config_dict)
        
        assert config.vllm.tensor_parallel_size == 2
        assert config.models == ['model1', 'model2']
        assert config.datasets == ['dataset1']
        assert config.output.base_dir == './test_results'
    
    def test_config_from_yaml_file(self):
        yaml_content = """
        vllm:
          tensor_parallel_size: 4
          gpu_memory_utilization: 0.8
        models:
          - "test/model"
        datasets:
          - "test/dataset"
        output:
          base_dir: "./results"
          format: "json"
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            
            config = Config.from_yaml(f.name)
            
            assert config.vllm.tensor_parallel_size == 4
            assert config.vllm.gpu_memory_utilization == 0.8
            assert config.models == ["test/model"]
            assert config.datasets == ["test/dataset"]
        
        Path(f.name).unlink()
    
    def test_config_load_default(self):
        config = Config.load_default()
        
        assert isinstance(config.vllm, VLLMConfig)
        assert isinstance(config.output, OutputConfig)
        assert isinstance(config.evaluation, EvaluationConfig)
        assert config.models == ["Qwen/Qwen3-0.6B"]
    
    def test_config_validation_invalid_tensor_parallel(self):
        with pytest.raises(ValueError, match="tensor_parallel_size must be positive"):
            VLLMConfig(
                tensor_parallel_size=0,
                max_model_len=2048
            )
    
    def test_config_validation_invalid_gpu_memory(self):
        with pytest.raises(ValueError, match="gpu_memory_utilization must be between 0 and 1"):
            VLLMConfig(
                tensor_parallel_size=1,
                max_model_len=2048,
                gpu_memory_utilization=1.5
            )
    
    def test_config_merge_with_overrides(self):
        base_config = Config.load_default()
        overrides = {
            'models': ['new/model'],
            'vllm': {'tensor_parallel_size': 2}
        }
        
        merged_config = base_config.merge(overrides)
        
        assert merged_config.models == ['new/model']
        assert merged_config.vllm.tensor_parallel_size == 2