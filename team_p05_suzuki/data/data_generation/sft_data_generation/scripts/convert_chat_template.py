#!/usr/bin/env python3
"""CLI tool for converting datasets to model-specific chat templates."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chat_template_adapter import TemplateAdapterFactory


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_json(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both list and single object formats
    if isinstance(data, dict):
        return [data]
    return data


def save_jsonl(data: List[Any], file_path: str):
    """Save data to JSONL file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def save_json(data: List[Any], file_path: str):
    """Save data to JSON file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_text(data: List[Any], file_path: str):
    """Save data as plain text file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            if isinstance(item, str):
                f.write(item + '\n\n')
            else:
                # Handle dict/list items
                f.write(json.dumps(item, ensure_ascii=False) + '\n\n')


def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load data from file based on extension"""
    ext = Path(file_path).suffix.lower()
    
    if ext == '.jsonl':
        return load_jsonl(file_path)
    elif ext == '.json':
        return load_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def save_data(data: List[Any], file_path: str, format: str):
    """Save data to file in specified format"""
    if format == 'jsonl':
        save_jsonl(data, file_path)
    elif format == 'json':
        save_json(data, file_path)
    elif format == 'text':
        save_text(data, file_path)
    else:
        raise ValueError(f"Unsupported output format: {format}")


def validate_data(data: List[Dict[str, Any]], logger: logging.Logger) -> List[Dict[str, Any]]:
    """Validate and clean input data."""
    valid_data = []
    
    for i, entry in enumerate(data):
        # Support both input/output and question/answer formats
        has_input_output = 'input' in entry and 'output' in entry
        has_question_answer = 'question' in entry and 'answer' in entry
        
        if not has_input_output and not has_question_answer:
            logger.warning(f"Entry {i} missing required fields (question/answer), skipping")
            continue
            
        # Normalize to question/answer format
        if has_input_output and not has_question_answer:
            entry['question'] = entry['input']
            entry['answer'] = entry['output']
            
        if 'think' not in entry:
            entry['think'] = ''
            
        valid_data.append(entry)
    
    logger.info(f"Validated {len(valid_data)} out of {len(data)} entries")
    return valid_data


def convert_dataset(
    input_file: str,
    output_file: str,
    model_type: str,
    output_format: str,
    include_metadata: bool = False,
    logger: Optional[logging.Logger] = None
) -> int:
    """Convert dataset to model-specific chat template."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Loading data from {input_file}")
    data = load_data(input_file)
    
    valid_data = validate_data(data, logger)
    
    if not valid_data:
        logger.error("No valid data to process")
        return 0
    
    logger.info(f"Converting {len(valid_data)} entries to {model_type} format")
    
    adapter = TemplateAdapterFactory.get_adapter(model_type)
    formatted_data = []
    
    for entry in tqdm(valid_data, desc="Converting"):
        # Use question/answer format
        dataset_item = {
            'question': entry.get('question', ''),
            'think': entry.get('think', ''),
            'answer': entry.get('answer', '')
        }
        
        # Format using the adapter
        formatted_result = adapter.format_dataset_item(dataset_item)
        
        if output_format == 'text':
            # For text format, extract the text field if available
            if 'text' in formatted_result:
                formatted_data.append(formatted_result['text'])
            else:
                # Fallback: serialize messages
                formatted_data.append(json.dumps(formatted_result.get('messages', formatted_result), ensure_ascii=False))
        else:
            # For JSON/JSONL format
            formatted_entry = formatted_result
            
            if include_metadata:
                formatted_entry['metadata'] = {
                    'original_question': entry.get('question', ''),
                    'original_answer': entry.get('answer', ''),
                    'has_thinking': bool(entry.get('think', ''))
                }
                
                for key, value in entry.items():
                    if key not in ['question', 'answer', 'think']:
                        formatted_entry['metadata'][key] = value
            
            formatted_data.append(formatted_entry)
    
    logger.info(f"Saving converted data to {output_file}")
    save_data(formatted_data, output_file, output_format)
    
    logger.info(f"Successfully converted {len(formatted_data)} entries")
    return len(formatted_data)


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description='Convert question/think/answer datasets to model-specific chat templates',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python convert_chat_template.py data.jsonl output.jsonl --model deepseek-r1
    python convert_chat_template.py data.json output.json --model qwen3 --include-metadata
    python convert_chat_template.py data.jsonl output.txt --model qwen3 --format text
    python convert_chat_template.py data/*.jsonl --output-dir converted/ --model deepseek-r1
        """
    )
    
    parser.add_argument(
        'input',
        nargs='+',
        help='Input file(s) in JSON or JSONL format'
    )
    
    parser.add_argument(
        'output',
        nargs='?',
        help='Output file path (required for single input file)'
    )
    
    parser.add_argument(
        '--output-dir',
        help='Output directory for batch processing'
    )
    
    # Get available models dynamically from TemplateAdapterFactory
    available_models = TemplateAdapterFactory.get_available_templates()
    
    parser.add_argument(
        '--model',
        required=True,
        choices=available_models,
        help=f'Target model type. Available: {", ".join(available_models)}'
    )
    
    parser.add_argument(
        '--format',
        choices=['json', 'jsonl', 'text'],
        default='jsonl',
        help='Output format (default: jsonl)'
    )
    
    parser.add_argument(
        '--include-metadata',
        action='store_true',
        help='Include original data as metadata in output'
    )
    
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    if len(args.input) == 1 and not args.output and not args.output_dir:
        parser.error("Either output file or --output-dir must be specified")
    
    if len(args.input) > 1 and args.output:
        parser.error("Cannot specify output file with multiple inputs. Use --output-dir instead")
    
    if len(args.input) > 1 and not args.output_dir:
        parser.error("--output-dir is required for batch processing")
    
    total_converted = 0
    
    if len(args.input) == 1:
        input_file = args.input[0]
        output_file = args.output or os.path.join(
            args.output_dir,
            f"{Path(input_file).stem}_{args.model}.{args.format}"
        )
        
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
        
        try:
            count = convert_dataset(
                input_file,
                output_file,
                args.model,
                args.format,
                args.include_metadata,
                logger
            )
            total_converted += count
        except Exception as e:
            logger.error(f"Failed to process {input_file}: {e}")
            return 1
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        
        for input_file in args.input:
            output_file = os.path.join(
                args.output_dir,
                f"{Path(input_file).stem}_{args.model}.{args.format}"
            )
            
            try:
                count = convert_dataset(
                    input_file,
                    output_file,
                    args.model,
                    args.format,
                    args.include_metadata,
                    logger
                )
                total_converted += count
            except Exception as e:
                logger.error(f"Failed to process {input_file}: {e}")
                continue
    
    logger.info(f"Total entries converted: {total_converted}")
    return 0


if __name__ == '__main__':
    sys.exit(main())