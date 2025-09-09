#!/usr/bin/env python3

import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from filter_solvable_question.core.answer_evaluator import AnswerEvaluator

def test_actual_results():
    evaluator = AnswerEvaluator()
    
    # Load actual results
    with open('results/math/MATH-500/Qwen_Qwen3-0.6B/results.json', 'r') as f:
        data = json.load(f)
    
    print("Testing with actual results:")
    print("=" * 50)
    
    # Access the question_statistics array
    questions = data['results']['question_statistics']
    
    total = 0
    correct = 0
    
    for i, item in enumerate(questions):
        raw_output = item['model_responses']['Qwen/Qwen3-0.6B']['raw_output']
        correct_answer = item['correct_answer']
        
        # Extract answer from raw_output
        extracted = evaluator.extract_answer_from_response(raw_output)
        
        # Evaluate
        result = evaluator.evaluate(raw_output, correct_answer)
        
        total += 1
        if result:
            correct += 1
        
        if i < 5:
            print(f"Question {i+1}:")
            print(f"  Correct Answer: {correct_answer}")
            print(f"  Extracted Answer: {extracted}")
            print(f"  Evaluation Result: {result}")
            print(f"  Raw Output Preview: {raw_output[:100]}...")
            print("-" * 30)
    
    print("\n==============================")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {correct/total:.4f}")
    print("==============================")

if __name__ == "__main__":
    test_actual_results() 