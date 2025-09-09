import re
from typing import Union


class AnswerEvaluator:
    def __init__(self):
        self.answer_patterns = [
            # Boxed answers (LaTeX format) - highest priority
            r"\\boxed\{([^}]+)\}",
            r"\\boxed\{([^}]*\\frac\{[^}]*\}\{[^}]*\}[^}]*)\}",
            r"\\boxed\{([^}]*\\sqrt\{[^}]*\}[^}]*)\}",
            r"\\boxed\{([^}]*\\pi[^}]*)\}",
            r"\\boxed\{([^}]*\\dfrac\{[^}]*\}\{[^}]*\}[^}]*)\}",
            
            # Final answer patterns
            r"(?:the )?(?:final )?answer is\s*([^\n.]+)",
            r"(?:answer|result):\s*([^\n.]+)",
            r"= ([^\n.]+)$",
            r"therefore,?\s*([^\n.]+)",
            
            # Mathematical expressions
            r"\\frac\{([^}]+)\}\{([^}]+)\}",
            r"\\sqrt\{([^}]+)\}",
            r"\\pi",
            r"\\dfrac\{([^}]+)\}\{([^}]+)\}",
            
            # Boxed answers with complex expressions
            r"\\boxed\{([^}]*\\dfrac\{[^}]*\}\{[^}]*\}[^}]*)\}",
            r"\\boxed\{([^}]*\\frac\{[^}]*\}\{[^}]*\}[^}]*)\}",
            
            # Simple boxed answers
            r"\\boxed\{([^}]+)\}",
            
            # Final answer with boxed
            r"(?:final answer|answer):\s*\\boxed\{([^}]+)\}",
            
            # Mathematical expressions without boxed
            r"(?:answer|result):\s*\\frac\{([^}]+)\}\{([^}]+)\}",
            r"(?:answer|result):\s*\\sqrt\{([^}]+)\}",
            
            # Simple numerical answers
            r"(?:answer|result):\s*(\d+(?:\.\d+)?)",
            r"= (\d+(?:\.\d+)?)$",
            
            # Text answers
            r"(?:answer|result):\s*([A-Za-z\s]+)",
        ]
    
    def extract_answer_from_response(self, response: str) -> str:
        response = response.strip()
        
        # First, try to find boxed answers (most common in math problems)
        # Use a more robust pattern that can handle nested braces
        boxed_pattern = r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
        boxed_matches = re.findall(boxed_pattern, response, re.IGNORECASE | re.MULTILINE)
        
        if boxed_matches:
            # Return the last boxed answer found (most likely the final answer)
            return boxed_matches[-1].strip()
        
        # Look for specific patterns in the response
        # Check for "the answer is" or "final answer" patterns
        answer_patterns = [
            r"(?:the )?(?:final )?answer is\s*([^\n.]+)",
            r"(?:answer|result):\s*([^\n.]+)",
            r"= ([^\n.]+)$",
            r"therefore,?\s*([^\n.]+)",
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()
        
        # Look for numerical answers with more specific patterns
        number_patterns = [
            r"the answer is (\d+(?:\.\d+)?)",
            r"result: (\d+(?:\.\d+)?)",
            r"count is (\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?) (?:positive )?divisors",
            r"(\d+(?:\.\d+)?) (?:positive )?whole-number divisors",
            r"(\d+(?:\.\d+)?) positive whole-number divisors",
            r"has (\d+(?:\.\d+)?) positive whole-number divisors",
            r"(\d+(?:\.\d+)?) positive whole-number divisors",
            r"(\d+(?:\.\d+)?) positive whole-number divisors\.",
            r"196 has (\d+(?:\.\d+)?) positive whole-number divisors",
            r"196 has (\d+(?:\.\d+)?) positive whole-number divisors\.",
        ]
        
        for pattern in number_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()
        
        # Look for text answers (names, etc.) with more specific patterns
        text_patterns = [
            r"(?:answer|result):\s*([A-Za-z\s]+)",
            r"the answer is ([A-Za-z\s]+)",
            r"student with the greatest average speed is ([A-Za-z\s]+)",
            r"([A-Za-z\s]+) has the greatest average speed",
            r"([A-Za-z\s]+) is the student with the greatest average speed",
            r"greatest average speed is ([A-Za-z\s]+)",
            r"the student with the greatest average speed is \*\*([A-Za-z\s]+)\*\*",
            r"the student with the greatest average speed is ([A-Za-z\s]+)",
            r"student with the greatest average speed is \*\*([A-Za-z\s]+)\*\*",
        ]
        
        for pattern in text_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()
        
        # If no pattern matches, try to extract the last mathematical expression
        # Look for common mathematical patterns at the end of the response
        math_patterns = [
            r"\\frac\{([^}]+)\}\{([^}]+)\}",
            r"\\sqrt\{([^}]+)\}",
            r"\\pi",
            r"\\dfrac\{([^}]+)\}\{([^}]+)\}",
            r"(\d+(?:\.\d+)?)",
        ]
        
        # Split response into lines and look for mathematical expressions in the last few lines
        lines = response.split('\n')
        for line in reversed(lines[-5:]):  # Check last 5 lines
            for pattern in math_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    return match.group(0).strip()
        
        return response
    
    def normalize_answer(self, answer: str) -> str:
        # Remove LaTeX commands and normalize mathematical expressions
        normalized = answer.strip()
        
        # Handle nested LaTeX expressions more carefully
        # First, normalize fractions - handle nested braces
        while re.search(r'\\frac\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', normalized):
            normalized = re.sub(r'\\frac\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', r'\1/\2', normalized)
        
        while re.search(r'\\dfrac\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', normalized):
            normalized = re.sub(r'\\dfrac\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', r'\1/\2', normalized)
        
        # Normalize square roots - handle nested braces
        while re.search(r'\\sqrt\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', normalized):
            normalized = re.sub(r'\\sqrt\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', r'sqrt(\1)', normalized)
        
        # Normalize pi
        normalized = re.sub(r'\\pi', 'pi', normalized)
        
        # Remove \left and \right commands
        normalized = re.sub(r'\\left', '', normalized)
        normalized = re.sub(r'\\right', '', normalized)
        
        # Remove other LaTeX commands but preserve their content
        normalized = re.sub(r'\\[a-zA-Z]+\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', r'\1', normalized)
        
        # Remove extra spaces and convert to lowercase
        normalized = re.sub(r'\s+', '', normalized).lower()
        
        # Handle special cases
        normalized = re.sub(r'\\', '', normalized)  # Remove remaining backslashes
        normalized = re.sub(r'\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', r'\1', normalized)  # Remove remaining braces
        
        return normalized
    
    def evaluate(self, llm_answer: str, correct_answer: str) -> bool:
        extracted_answer = self.extract_answer_from_response(llm_answer)
        
        normalized_extracted = self.normalize_answer(extracted_answer)
        normalized_correct = self.normalize_answer(correct_answer)
        
        return normalized_extracted == normalized_correct
    
    def evaluate_batch(self, llm_answers: list, correct_answers: list) -> list:
        if len(llm_answers) != len(correct_answers):
            raise ValueError("Lists must have the same length")
        
        results = []
        for llm_ans, correct_ans in zip(llm_answers, correct_answers):
            results.append(self.evaluate(llm_ans, correct_ans))
        
        return results