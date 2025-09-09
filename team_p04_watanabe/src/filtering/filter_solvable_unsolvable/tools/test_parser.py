#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from filter_solvable_question.core.answer_evaluator import AnswerEvaluator

def test_answer_extraction():
    evaluator = AnswerEvaluator()
    
    # Test cases
    test_cases = [
        {
            'response': 'The answer is \\boxed{(3, \\frac{\\pi}{2})}',
            'expected': '(3, \\frac{\\pi}{2})'
        },
        {
            'response': '\\boxed{p - q}',
            'expected': 'p - q'
        },
        {
            'response': '\\boxed{\\dfrac{14}{3}}',
            'expected': '\\dfrac{14}{3}'
        },
        {
            'response': 'The final answer is \\boxed{9}',
            'expected': '9'
        }
    ]
    
    print("Testing answer extraction:")
    for i, test_case in enumerate(test_cases):
        extracted = evaluator.extract_answer_from_response(test_case['response'])
        print(f"Test {i+1}:")
        print(f"  Response: {test_case['response']}")
        print(f"  Extracted: {extracted}")
        print(f"  Expected: {test_case['expected']}")
        print(f"  Match: {extracted == test_case['expected']}")
        print()

def test_normalization():
    evaluator = AnswerEvaluator()
    
    # Test cases
    test_cases = [
        {
            'input': '\\frac{14}{3}',
            'expected': '14/3'
        },
        {
            'input': '\\dfrac{14}{3}',
            'expected': '14/3'
        },
        {
            'input': '\\sqrt{9}',
            'expected': 'sqrt(9)'
        },
        {
            'input': '\\pi',
            'expected': 'pi'
        },
        {
            'input': 'p - q',
            'expected': 'p-q'
        }
    ]
    
    print("Testing answer normalization:")
    for i, test_case in enumerate(test_cases):
        normalized = evaluator.normalize_answer(test_case['input'])
        print(f"Test {i+1}:")
        print(f"  Input: {test_case['input']}")
        print(f"  Normalized: {normalized}")
        print(f"  Expected: {test_case['expected']}")
        print(f"  Match: {normalized == test_case['expected']}")
        print()

def test_evaluation():
    evaluator = AnswerEvaluator()
    
    # Test cases
    test_cases = [
        {
            'llm_answer': 'The answer is \\boxed{\\dfrac{14}{3}}',
            'correct_answer': '\\frac{14}{3}',
            'expected': True
        },
        {
            'llm_answer': '\\boxed{p - q}',
            'correct_answer': 'p - q',
            'expected': True
        },
        {
            'llm_answer': 'The answer is \\boxed{(3, \\frac{\\pi}{2})}',
            'correct_answer': '(3, \\frac{\\pi}{2})',
            'expected': True
        }
    ]
    
    print("Testing answer evaluation:")
    for i, test_case in enumerate(test_cases):
        result = evaluator.evaluate(test_case['llm_answer'], test_case['correct_answer'])
        print(f"Test {i+1}:")
        print(f"  LLM Answer: {test_case['llm_answer']}")
        print(f"  Correct Answer: {test_case['correct_answer']}")
        print(f"  Result: {result}")
        print(f"  Expected: {test_case['expected']}")
        print(f"  Match: {result == test_case['expected']}")
        print()

if __name__ == "__main__":
    test_answer_extraction()
    test_normalization()
    test_evaluation() 