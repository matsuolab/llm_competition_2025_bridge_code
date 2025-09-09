import yaml
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator
import logging

logger = logging.getLogger(__name__)


class VLLMConfig(BaseModel):
    tensor_parallel_size: int = Field(default=1)
    max_model_len: int = Field(default=2048, gt=0)
    gpu_memory_utilization: float = Field(default=0.9)
    dtype: str = Field(default="auto")
    quantization: Optional[str] = Field(default=None)
    trust_remote_code: bool = Field(default=False)
    enable_chunked_prefill: bool = Field(default=True)
    batch_size: int = Field(default=8, gt=0)
    max_num_batched_tokens: int = Field(default=16384, gt=0)
    quantization: Optional[str] = Field(default=None)

    @field_validator('gpu_memory_utilization')
    @classmethod
    def validate_gpu_memory(cls, v):
        if not 0 < v <= 1.0:
            raise ValueError("gpu_memory_utilization must be between 0 and 1")
        return v

    @field_validator('tensor_parallel_size')
    @classmethod
    def validate_tensor_parallel_size(cls, v):
        if not v > 0:
            raise ValueError("tensor_parallel_size must be positive")
        return v


class DatasetMapping(BaseModel):
    """Field mapping for a specific dataset."""
    question_field: str = Field(default="question")
    answer_field: str = Field(default="answer")


class DatasetConfig(BaseModel):
    """Configuration for dataset field mappings."""
    default_mappings: DatasetMapping = Field(default_factory=DatasetMapping)
    dataset_specific: Dict[str, DatasetMapping] = Field(default_factory=dict)
    repositories: List[str] = Field(default_factory=list)


class OutputConfig(BaseModel):
    base_dir: str = Field(default="./results")
    format: str = Field(default="json")
    include_metadata: bool = Field(default=True)
    save_raw_responses: bool = Field(default=True)
    save_per_dataset: bool = Field(default=True)


class EvaluationConfig(BaseModel):
    batch_size: int = Field(default=8, gt=0)
    max_questions: Optional[int] = Field(default=None)


class LoggingConfig(BaseModel):
    level: str = Field(default="INFO")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class Config(BaseModel):
    vllm: VLLMConfig = Field(default_factory=VLLMConfig)
    models: List[str] = Field(default=["Qwen/Qwen3-0.6B"])
    datasets: DatasetConfig = Field(default_factory=DatasetConfig)
    # Legacy support for backward compatibility
    dataset_repositories: List[str] = Field(default_factory=list)
    output: OutputConfig = Field(default_factory=OutputConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    def get_dataset_repositories(self) -> List[str]:
        """Get dataset repositories from either new or legacy config."""
        if self.datasets.repositories:
            return self.datasets.repositories
        return self.dataset_repositories
    
    def get_field_mapping(self, dataset_name: str) -> DatasetMapping:
        """Get field mapping for a specific dataset."""
        # Check if there's a dataset-specific mapping
        if dataset_name in self.datasets.dataset_specific:
            return self.datasets.dataset_specific[dataset_name]
        # Fall back to default mappings
        return self.datasets.default_mappings
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create config from dictionary."""
        # Handle None values for dataset_specific
        if 'datasets' in config_dict and isinstance(config_dict['datasets'], dict):
            if config_dict['datasets'].get('dataset_specific') is None:
                config_dict['datasets']['dataset_specific'] = {}
        
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'Config':
        """Load config from YAML file."""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        if not config_dict:
            raise ValueError(f"Empty or invalid YAML file: {yaml_path}")
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def load_default(cls) -> 'Config':
        """Load default configuration."""
        default_config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
        return cls.from_yaml(default_config_path)
    
    def merge(self, overrides: Dict[str, Any]) -> 'Config':
        """Merge configuration with overrides."""
        config_dict = self.dict()
        
        def deep_update(base_dict: Dict, update_dict: Dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(config_dict, overrides)
        return self.__class__.from_dict(config_dict)
    
    def save_to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.dict(), f, default_flow_style=False, allow_unicode=True)
    
    def setup_logging(self) -> None:
        """Setup logging based on configuration."""
        logging.basicConfig(
            level=getattr(logging, self.logging.level.upper()),
            format=self.logging.format
        )