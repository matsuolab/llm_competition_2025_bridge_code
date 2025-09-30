"""
Data cleansing module for SFT data generation.
Handles cleaning and validation of generated data to ensure proper formatting.
"""

import re
import json
from typing import Optional, Dict, Any, List
from pathlib import Path


class DataCleaner:
    """Main data cleaner class for processing generated data."""
    
    def __init__(self):
        """Initialize the data cleaner with common patterns."""
        # Patterns for cleaning
        self.markdown_bold_pattern = re.compile(r'\*\*([^*]+)\*\*')
        self.markdown_italic_pattern = re.compile(r'\*([^*]+)\*')
        self.answer_prefix_patterns = [
            re.compile(r'^answer:\s*', re.IGNORECASE),
            re.compile(r'^the answer is\s*', re.IGNORECASE),
            re.compile(r'^answer is\s*', re.IGNORECASE),
            re.compile(r'^the correct answer is\s*', re.IGNORECASE),
        ]
        self.think_prefix_patterns = [
            re.compile(r'^### Thinking Process:\s*\n?', re.MULTILINE),
            re.compile(r'^\*\*Thinking Process:\*\*\s*\n?', re.MULTILINE),
            re.compile(r'^Thinking Process:\s*\n?', re.MULTILINE),
            re.compile(r'^<think>\s*\n?', re.MULTILINE),
            re.compile(r'^</think>\s*$', re.MULTILINE),
        ]
    
    def clean_text(self, text: str) -> str:
        """Remove unwanted characters and formatting from text.
        
        Args:
            text: The text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove any control characters except newlines and tabs
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Remove weird characters that might appear at the beginning
        # like [O, [P, etc.
        text = re.sub(r'^[\[\]]+[A-Z](?=[A-Z\s])', '', text)
        
        # Fix multiple consecutive spaces (but preserve single newlines)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Fix multiple consecutive newlines (max 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text
    
    def clean_answer(self, answer: str) -> Optional[str]:
        """Clean answer field by removing unwanted formatting and validating content.
        
        Args:
            answer: The answer to clean
            
        Returns:
            Cleaned answer or None if invalid
        """
        if not answer or not isinstance(answer, str):
            return None
        
        # Basic cleaning
        answer = self.clean_text(answer)
        
        # Remove answer prefixes first (before removing markdown)
        for pattern in self.answer_prefix_patterns:
            answer = pattern.sub('', answer).strip()
        
        # Remove ** from beginning and end (may appear after removing prefix)
        if answer.startswith("** "):
            answer = answer[3:]
        elif answer.startswith("**"):
            answer = answer[2:].strip()
        
        if answer.endswith("**"):
            answer = answer[:-2].strip()
        
        # Remove markdown formatting
        answer = self.markdown_bold_pattern.sub(r'\1', answer)
        answer = self.markdown_italic_pattern.sub(r'\1', answer)
        
        # Remove unnecessary quotes around single letters/choices
        if re.match(r'^["\']([A-Z])["\']$', answer):
            answer = answer[1:-1]
        
        # Clean up choice answers with periods or parentheses
        if re.match(r'^[A-Z][\)\.]\s*', answer):
            answer = answer[0]
        
        # Clean LaTeX expressions
        answer = self.clean_latex(answer)
        
        # Validate the answer
        if not self.validate_answer(answer):
            return None
        
        return answer
    
    def clean_latex(self, text: str) -> str:
        """Clean and fix common LaTeX formatting issues.
        
        Args:
            text: Text containing LaTeX
            
        Returns:
            Cleaned LaTeX text
        """
        # Fix double backslashes that should be single
        text = re.sub(r'\\\\([a-zA-Z]+)', r'\\\1', text)
        
        # Fix LaTeX commands without proper spacing
        text = re.sub(r'(\\[a-zA-Z]+)([0-9])', r'\1 \2', text)
        
        # Fix missing spaces in LaTeX expressions
        # But be careful not to add spaces in subscripts/superscripts
        text = re.sub(r'(\\log|\\sin|\\cos|\\tan|\\ln|\\exp)([a-zA-Z0-9])', r'\1 \2', text)
        
        # Fix common LaTeX bracket issues
        text = text.replace('\\\\(', '\\(')
        text = text.replace('\\\\)', '\\)')
        text = text.replace('\\\\[', '\\[')
        text = text.replace('\\\\]', '\\]')
        
        # Ensure dollar signs are properly used (not escaped unless needed)
        # This is tricky - only fix obvious cases
        text = re.sub(r'(?<!\\)\$\$', r'$$', text)  # Ensure $$ is not escaped
        
        return text
    
    def validate_answer(self, answer: str) -> bool:
        """Validate that the answer is in an acceptable format.
        
        Args:
            answer: The answer to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not answer:
            return False
        
        # Check if it's a single letter choice
        if re.match(r'^[A-Z]$', answer):
            return True
        
        # Check if it's a number (integer or decimal, possibly negative)
        if re.match(r'^-?\d+(\.\d+)?$', answer):
            return True
        
        # Check if it's a fraction
        if re.match(r'^-?\d+/\d+$', answer):
            return True
        
        # Check if it's a percentage
        if re.match(r'^-?\d+(\.\d+)?%$', answer):
            return True
        
        # Check if it contains LaTeX (has backslash or dollar signs)
        if '\\' in answer or '$' in answer:
            # Basic LaTeX validation - should have matching delimiters
            dollar_count = answer.count('$') - answer.count('\\$')
            if dollar_count % 2 != 0:
                return False
            # Allow LaTeX expressions up to 500 characters
            if len(answer) <= 500:
                return True
        
        # Check for invalid markdown formatting
        if '**' in answer or '##' in answer or '###' in answer:
            return False
        
        # Allow short text answers up to 100 characters
        if len(answer) <= 100:
            return True
        
        return False
    
    def clean_think(self, think: str) -> Optional[str]:
        """Clean the thinking process field.
        
        Args:
            think: The thinking process text
            
        Returns:
            Cleaned thinking text or None if invalid
        """
        if not think or not isinstance(think, str):
            return None
        
        # Basic cleaning
        think = self.clean_text(think)
        
        # Remove thinking process prefixes
        for pattern in self.think_prefix_patterns:
            think = pattern.sub('', think)
        
        # Remove trailing markers like "###"
        if think.endswith("###"):
            think = think[:-3].rstrip()
        
        # Remove <think> tags if present
        think = re.sub(r'</?think>', '', think, flags=re.IGNORECASE)
        
        # Remove any exam-related text if it appears at the beginning
        # (Removed specific exam name reference for competition compliance)
        think = re.sub(r'^[^\n]*(?:exam|assessment|test)[^\n]*\n*', '', think, flags=re.IGNORECASE)
        
        # Clean LaTeX in thinking
        think = self.clean_latex(think)
        
        # Skip if think content is empty after cleaning
        if not think.strip():
            return None
        
        return think
    
    def clean_question(self, question: str) -> Optional[str]:
        """Clean the question field.
        
        Args:
            question: The question text
            
        Returns:
            Cleaned question or None if invalid
        """
        if not question or not isinstance(question, str):
            return None
        
        # Basic cleaning
        question = self.clean_text(question)
        
        # Clean LaTeX in question
        question = self.clean_latex(question)
        
        # Skip if question is empty after cleaning
        if not question.strip():
            return None
        
        return question
    
    def clean_data_entry(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Clean a single data entry.
        
        Args:
            data: Dictionary containing the data entry
            
        Returns:
            Cleaned data entry or None if invalid
        """
        cleaned = {}
        
        # Clean question
        if 'question' in data:
            cleaned_question = self.clean_question(data['question'])
            if not cleaned_question:
                return None
            cleaned['question'] = cleaned_question
        else:
            return None  # Question is required
        
        # Clean think
        if 'think' in data:
            cleaned_think = self.clean_think(data['think'])
            if cleaned_think:
                cleaned['think'] = cleaned_think
            else:
                # Think field is optional but if present should be valid
                return None
        
        # Clean answer
        if 'answer' in data:
            cleaned_answer = self.clean_answer(data['answer'])
            if not cleaned_answer:
                return None
            cleaned['answer'] = cleaned_answer
        else:
            return None  # Answer is required
        
        # Copy other fields as-is
        for key, value in data.items():
            if key not in ['question', 'think', 'answer']:
                cleaned[key] = value
        
        return cleaned
    
    def clean_jsonl_file(self, input_path: str, output_path: str = None) -> int:
        """Clean a JSONL file and write the cleaned version.
        
        Args:
            input_path: Path to input JSONL file
            output_path: Path to output JSONL file (if None, overwrites input)
            
        Returns:
            Number of successfully cleaned entries
        """
        if output_path is None:
            output_path = input_path
        
        cleaned_entries = []
        skipped_count = 0
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        cleaned = self.clean_data_entry(data)
                        if cleaned:
                            cleaned_entries.append(cleaned)
                        else:
                            skipped_count += 1
                            print(f"  Skipped entry {line_num}: Invalid after cleaning")
                    except json.JSONDecodeError as e:
                        print(f"  Error parsing line {line_num}: {e}")
                        skipped_count += 1
                    except Exception as e:
                        print(f"  Unexpected error on line {line_num}: {e}")
                        skipped_count += 1
            
            # Write cleaned entries
            with open(output_path, 'w', encoding='utf-8') as f:
                for entry in cleaned_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            print(f"  Cleaned {len(cleaned_entries)} entries, skipped {skipped_count}")
            return len(cleaned_entries)
            
        except Exception as e:
            print(f"Error processing file {input_path}: {e}")
            return 0
    
    def clean_batch(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean a batch of data entries.
        
        Args:
            data_list: List of data dictionaries
            
        Returns:
            List of cleaned data dictionaries
        """
        cleaned_list = []
        for data in data_list:
            cleaned = self.clean_data_entry(data)
            if cleaned:
                cleaned_list.append(cleaned)
        return cleaned_list


def clean_generated_output(output_text: str) -> Dict[str, str]:
    """Clean the raw output from LLM generation.
    
    This function handles the specific format issues seen in LLM outputs,
    such as extra characters at the beginning, malformed JSON, etc.
    
    Args:
        output_text: Raw output text from LLM
        
    Returns:
        Dictionary with cleaned 'question', 'think', and 'answer' fields
    """
    cleaner = DataCleaner()
    
    # Remove any leading garbage characters
    output_text = re.sub(r'^[\[\]]+[A-Z](?=[A-Za-z\s{])', '', output_text)
    
    # Try to parse as JSON first
    try:
        # Find JSON boundaries
        json_start = output_text.find('{')
        json_end = output_text.rfind('}')
        if json_start >= 0 and json_end > json_start:
            json_str = output_text[json_start:json_end+1]
            data = json.loads(json_str)
            
            # Clean the parsed data
            if isinstance(data, dict):
                result = {}
                if 'question' in data:
                    result['question'] = cleaner.clean_question(data['question'])
                if 'think' in data:
                    result['think'] = cleaner.clean_think(data['think'])
                if 'answer' in data:
                    result['answer'] = cleaner.clean_answer(data['answer'])
                return result
    except:
        pass
    
    # If JSON parsing fails, try to extract fields manually
    result = {}
    
    # Extract question
    question_match = re.search(r'"question"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', output_text)
    if question_match:
        result['question'] = cleaner.clean_question(question_match.group(1))
    
    # Extract think
    think_match = re.search(r'"think"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', output_text)
    if think_match:
        result['think'] = cleaner.clean_think(think_match.group(1))
    
    # Extract answer
    answer_match = re.search(r'"answer"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', output_text)
    if answer_match:
        result['answer'] = cleaner.clean_answer(answer_match.group(1))
    
    return result


# Utility functions for integration with existing pipeline
def integrate_cleaner_into_pipeline(data_item: Dict[str, Any]) -> Dict[str, Any]:
    """Integrate the cleaner into the existing data generation pipeline.
    
    This function can be called after generating each data item to ensure
    it's properly cleaned before saving.
    
    Args:
        data_item: Generated data item dictionary
        
    Returns:
        Cleaned data item dictionary
    """
    cleaner = DataCleaner()
    cleaned = cleaner.clean_data_entry(data_item)
    return cleaned if cleaned else data_item


if __name__ == "__main__":
    # Test the cleaner with sample data
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        
        cleaner = DataCleaner()
        cleaned_count = cleaner.clean_jsonl_file(input_file, output_file)
        print(f"Cleaning complete: {cleaned_count} entries processed")
    else:
        # Run some basic tests
        cleaner = DataCleaner()
        
        # Test answer cleaning
        test_answers = [
            "** 60",
            "**12**",
            "Answer: 42",
            "The answer is A",
            "A)",
            "3.14159",
            "-1/2",
            "\\frac{1}{2}",
        ]
        
        print("Testing answer cleaning:")
        for answer in test_answers:
            cleaned = cleaner.clean_answer(answer)
            print(f"  '{answer}' -> '{cleaned}'")
        
        print("\nCleaner module loaded successfully!")