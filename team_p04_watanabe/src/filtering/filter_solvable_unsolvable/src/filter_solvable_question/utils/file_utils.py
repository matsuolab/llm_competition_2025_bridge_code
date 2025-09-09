import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class FileUtils:
    """Utility functions for file and directory operations."""
    
    @staticmethod
    def extract_dataset_name(repo_name: str, base_category: str = None) -> str:
        """
        Extract dataset name from repository name, optionally removing category prefix.
        
        Args:
            repo_name: Repository name (e.g., "org/dataset-name" or "dataset-name")
            base_category: Base category to remove from dataset name (e.g., "math")
            
        Returns:
            Dataset name for use in file paths
        """
        if '/' in repo_name:
            dataset_name = repo_name.split('/')[-1]
        else:
            dataset_name = repo_name
            
        # Remove category prefix if present and base_category specified
        if base_category and dataset_name.startswith(f"{base_category}_"):
            dataset_name = dataset_name[len(f"{base_category}_"):]
            
        return dataset_name
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize filename by removing invalid characters.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename safe for filesystem use
        """
        # Replace invalid characters with underscores
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_{2,}', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        return sanitized
    
    @staticmethod
    def create_dataset_output_dir(base_dir: Path, dataset_name: str, model_name: str = None) -> Path:
        """
        Create output directory for a dataset and optionally model.
        
        Args:
            base_dir: Base output directory
            dataset_name: Name of the dataset
            model_name: Name of the model (optional)
            
        Returns:
            Path to the created directory
        """
        sanitized_dataset = FileUtils.sanitize_filename(dataset_name)
        
        if model_name:
            sanitized_model = FileUtils.sanitize_filename(model_name)
            output_dir = base_dir / sanitized_dataset / sanitized_model
        else:
            output_dir = base_dir / sanitized_dataset
            
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
        return output_dir
    
    @staticmethod
    def save_results_with_metadata(
        output_dir: Path,
        dataset_name: str,
        results: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save results with metadata to JSON file.
        
        Args:
            output_dir: Directory to save results
            dataset_name: Name of the dataset
            results: Results data to save
            metadata: Optional metadata to include
            
        Returns:
            Path to the saved file
        """
        if metadata is None:
            metadata = {}
        
        # Add default metadata
        metadata.update({
            'dataset_name': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'version': '0.1.0'
        })
        
        # Create complete data structure
        data = {
            'results': results,
            'metadata': metadata
        }
        
        # Save to file
        output_file = output_dir / 'results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved results to: {output_file}")
        return output_file
    
    @staticmethod
    def list_result_files(base_dir: Path, pattern: str = "**/results.json") -> List[Path]:
        """
        List all result files in base directory.
        
        Args:
            base_dir: Base directory to search
            pattern: Glob pattern for result files (supports nested directories)
            
        Returns:
            List of result file paths
        """
        if not base_dir.exists():
            logger.warning(f"Base directory does not exist: {base_dir}")
            return []
        
        result_files = list(base_dir.glob(pattern))
        logger.info(f"Found {len(result_files)} result files")
        return result_files
    
    @staticmethod
    def parse_result_file_path(result_file: Path, base_dir: Path) -> Dict[str, str]:
        """
        Parse result file path to extract dataset and model names.
        
        Args:
            result_file: Path to result file
            base_dir: Base directory
            
        Returns:
            Dictionary with dataset_name and model_name
        """
        relative_path = result_file.relative_to(base_dir)
        parts = relative_path.parts[:-1]  # Remove 'results.json'
        
        if len(parts) >= 2:
            return {
                'dataset_name': parts[0],
                'model_name': parts[1]
            }
        elif len(parts) == 1:
            return {
                'dataset_name': parts[0],
                'model_name': 'unknown'
            }
        else:
            return {
                'dataset_name': 'unknown',
                'model_name': 'unknown'
            }
    
    @staticmethod
    def load_results_from_file(file_path: Path) -> Dict[str, Any]:
        """
        Load results from JSON file.
        
        Args:
            file_path: Path to the results file
            
        Returns:
            Loaded results data
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Results file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.debug(f"Loaded results from: {file_path}")
        return data
    
    @staticmethod
    def ensure_directory(directory: Path) -> None:
        """
        Ensure directory exists, create if necessary.
        
        Args:
            directory: Directory path to ensure
        """
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")
    
    @staticmethod
    def get_file_size_mb(file_path: Path) -> float:
        """
        Get file size in megabytes.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File size in megabytes
        """
        if not file_path.exists():
            return 0.0
        
        size_bytes = file_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        return round(size_mb, 2)