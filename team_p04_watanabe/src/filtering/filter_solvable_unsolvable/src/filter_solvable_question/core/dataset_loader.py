from datasets import load_dataset
from typing import List, Dict, Any, Optional
from ..utils.config import DatasetMapping
import logging

logger = logging.getLogger(__name__)


class DatasetLoader:
    def __init__(self, repository_name: str, field_mapping: Optional[DatasetMapping] = None):
        self.repository_name = repository_name
        self.field_mapping = field_mapping or DatasetMapping()
        
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load dataset and normalize field names."""
        dataset = load_dataset(self.repository_name)
        raw_data = dataset['train']
        
        # Normalize field names based on mapping
        normalized_data = []
        for item in raw_data:
            normalized_item = self._normalize_item(item)
            if normalized_item:
                normalized_data.append(normalized_item)
        
        logger.info(f"Loaded {len(normalized_data)} items from {self.repository_name}")
        return normalized_data
    
    def _normalize_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize a single dataset item based on field mapping."""
        question_field = self.field_mapping.question_field
        answer_field = self.field_mapping.answer_field
        
        # Check if required fields exist
        if question_field not in item:
            logger.warning(f"Question field '{question_field}' not found in item: {list(item.keys())}")
            return None
            
        if answer_field not in item:
            logger.warning(f"Answer field '{answer_field}' not found in item: {list(item.keys())}")
            return None
        
        # Create normalized item with standard field names
        normalized = {
            'question': item[question_field],
            'answer': item[answer_field],
            'original_item': item  # Keep original for reference
        }
        
        return normalized
    
    def extract_correct_answer(self, answer_text: str) -> str:
        """Extract correct answer from answer text."""
        if "####" not in answer_text:
            return ""
        
        parts = answer_text.split("####")
        if len(parts) < 2:
            return ""
        
        return parts[-1].strip()
    
    def get_available_fields(self) -> List[str]:
        """Get list of available fields in the dataset (for debugging)."""
        try:
            dataset = load_dataset(self.repository_name)
            sample_item = dataset['train'][0]
            return list(sample_item.keys())
        except Exception as e:
            logger.error(f"Failed to get available fields: {e}")
            return []