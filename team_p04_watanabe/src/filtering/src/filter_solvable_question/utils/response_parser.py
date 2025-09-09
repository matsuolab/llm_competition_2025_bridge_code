import re
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ResponseParser:
    """Parse LLM response into structured components."""
    
    def __init__(self):
        self.reasoning_pattern = r"<think>(.*?)</think>"
        self.answer_pattern = r"####\s*(.*?)(?:\n|$)"
    
    def parse_response(self, raw_response: str) -> Dict[str, str]:
        """
        Parse raw LLM response into structured components.
        
        Args:
            raw_response: Raw output from LLM
            
        Returns:
            Dictionary with 'raw_output', 'reasoning', and 'answer' sections
        """
        result = {
            'raw_output': raw_response,
            'reasoning': self._extract_reasoning(raw_response),
            'answer': self._extract_answer(raw_response)
        }
        
        return result
    
    def _extract_reasoning(self, response: str) -> Optional[str]:
        """Extract reasoning section between <think> and </think> tags."""
        match = re.search(self.reasoning_pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            reasoning = match.group(1).strip()
            return reasoning if reasoning else None
        return None
    
    def _extract_answer(self, response: str) -> Optional[str]:
        """Extract answer section after #### marker."""
        match = re.search(self.answer_pattern, response, re.MULTILINE | re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            return answer if answer else None
        return None
    
    def parse_batch(self, raw_responses: list) -> list:
        """
        Parse multiple raw responses.
        
        Args:
            raw_responses: List of raw LLM outputs
            
        Returns:
            List of parsed response dictionaries
        """
        return [self.parse_response(response) for response in raw_responses]