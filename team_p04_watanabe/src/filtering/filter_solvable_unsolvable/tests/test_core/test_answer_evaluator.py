import pytest
from filter_solvable_question.core.answer_evaluator import AnswerEvaluator


class TestAnswerEvaluator:
    def test_init(self):
        evaluator = AnswerEvaluator()
        assert evaluator is not None
    
    def test_evaluate_exact_match_returns_true(self):
        evaluator = AnswerEvaluator()
        llm_answer = "4"
        correct_answer = "4"
        
        result = evaluator.evaluate(llm_answer, correct_answer)
        
        assert result is True
    
    def test_evaluate_different_answers_returns_false(self):
        evaluator = AnswerEvaluator()
        llm_answer = "5"
        correct_answer = "4"
        
        result = evaluator.evaluate(llm_answer, correct_answer)
        
        assert result is False
    
    def test_evaluate_with_whitespace_normalization(self):
        evaluator = AnswerEvaluator()
        llm_answer = "  4  "
        correct_answer = "4"
        
        result = evaluator.evaluate(llm_answer, correct_answer)
        
        assert result is True
    
    def test_evaluate_complex_mathematical_expression(self):
        evaluator = AnswerEvaluator()
        llm_answer = "\\(\\frac{C_{n_1}^{a_1} \\cdot C_{n_2}^{a_2}}{C_N^A}\\)"
        correct_answer = "\\(\\frac{C_{n_1}^{a_1} \\cdot C_{n_2}^{a_2}}{C_N^A}\\)"
        
        result = evaluator.evaluate(llm_answer, correct_answer)
        
        assert result is True
    
    def test_extract_answer_from_llm_response_with_direct_answer(self):
        evaluator = AnswerEvaluator()
        response = "The answer is 4"
        
        result = evaluator.extract_answer_from_response(response)
        
        assert result == "4"
    
    def test_extract_answer_from_llm_response_with_complex_text(self):
        evaluator = AnswerEvaluator()
        response = "Let me solve this step by step. First, I calculate... The final answer is 42."
        
        result = evaluator.extract_answer_from_response(response)
        
        assert result == "42"
    
    def test_extract_answer_from_llm_response_no_answer_pattern(self):
        evaluator = AnswerEvaluator()
        response = "This is just some text without a clear answer pattern."
        
        result = evaluator.extract_answer_from_response(response)
        
        assert result == response.strip()