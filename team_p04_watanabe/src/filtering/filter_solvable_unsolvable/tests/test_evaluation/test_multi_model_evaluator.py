import pytest
from unittest.mock import patch, MagicMock
from filter_solvable_question.evaluation.multi_model_evaluator import MultiModelEvaluator


class TestMultiModelEvaluator:
    def test_init_with_model_names(self):
        models = ["model1", "model2"]
        evaluator = MultiModelEvaluator(models)
        assert evaluator.model_names == models
    
    def test_evaluate_with_multiple_models_returns_aggregated_results(self):
        models = ["model1", "model2"]
        evaluator = MultiModelEvaluator(models)
        
        questions = ["Q1", "Q2"]
        correct_answers = ["A1", "A2"]
        
        with patch.object(evaluator, '_evaluate_single_model') as mock_eval:
            mock_eval.side_effect = [
                [True, False],  # model1 results
                [False, True]   # model2 results
            ]
            
            results = evaluator.evaluate_questions(questions, correct_answers)
            
            assert len(results['question_results']) == 2
            assert results['question_results'][0]['question'] == "Q1"
            assert results['question_results'][0]['success_rate'] == 0.5  # 1/2 models correct
            assert results['question_results'][1]['success_rate'] == 0.5  # 1/2 models correct
    
    def test_calculate_question_success_rates(self):
        evaluator = MultiModelEvaluator(["model1", "model2", "model3"])
        
        question_evaluations = {
            0: [True, False, True],   # Q0: 2/3 success
            1: [False, False, False], # Q1: 0/3 success
            2: [True, True, True]     # Q2: 3/3 success
        }
        
        questions = ["Q0", "Q1", "Q2"]
        correct_answers = ["A0", "A1", "A2"]
        
        results = evaluator._calculate_question_success_rates(
            questions, correct_answers, question_evaluations
        )
        
        assert len(results['question_results']) == 3
        assert results['question_results'][0]['success_rate'] == 2/3
        assert results['question_results'][1]['success_rate'] == 0.0
        assert results['question_results'][2]['success_rate'] == 1.0
    
    def test_get_overall_statistics(self):
        evaluator = MultiModelEvaluator(["model1", "model2"])
        
        question_results = [
            {'success_rate': 1.0},
            {'success_rate': 0.5},
            {'success_rate': 0.0}
        ]
        
        stats = evaluator._get_overall_statistics(question_results)
        
        assert stats['total_questions'] == 3
        assert stats['average_success_rate'] == 0.5
        assert stats['fully_solved_questions'] == 1
        assert stats['unsolved_questions'] == 1