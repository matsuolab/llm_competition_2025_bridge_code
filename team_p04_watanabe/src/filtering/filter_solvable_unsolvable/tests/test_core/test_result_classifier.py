import pytest
from filter_solvable_question.core.result_classifier import ResultClassifier


class TestResultClassifier:
    def test_init(self):
        classifier = ResultClassifier()
        assert classifier is not None
    
    def test_classify_results_separates_correct_and_incorrect(self):
        classifier = ResultClassifier()
        questions = ["Q1", "Q2", "Q3"]
        answers = ["A1", "A2", "A3"]
        evaluations = [True, False, True]
        
        result = classifier.classify_results(questions, answers, evaluations)
        
        assert len(result['solved']) == 2
        assert len(result['unsolved']) == 1
        assert result['solved'][0]['question'] == "Q1"
        assert result['solved'][1]['question'] == "Q3"
        assert result['unsolved'][0]['question'] == "Q2"
    
    def test_classify_results_includes_all_fields(self):
        classifier = ResultClassifier()
        questions = ["What is 2+2?"]
        answers = ["4"]
        evaluations = [True]
        
        result = classifier.classify_results(questions, answers, evaluations)
        
        solved_item = result['solved'][0]
        assert 'question' in solved_item
        assert 'answer' in solved_item
        assert 'llm_response' in solved_item
        assert 'correct' in solved_item
    
    def test_get_statistics_calculates_accuracy(self):
        classifier = ResultClassifier()
        evaluations = [True, False, True, True, False]
        
        stats = classifier.get_statistics(evaluations)
        
        assert stats['total'] == 5
        assert stats['correct'] == 3
        assert stats['incorrect'] == 2
        assert stats['accuracy'] == 0.6
    
    def test_get_statistics_empty_list(self):
        classifier = ResultClassifier()
        evaluations = []
        
        stats = classifier.get_statistics(evaluations)
        
        assert stats['total'] == 0
        assert stats['correct'] == 0
        assert stats['incorrect'] == 0
        assert stats['accuracy'] == 0.0
    
    def test_save_results_creates_proper_structure(self):
        classifier = ResultClassifier()
        questions = ["Q1", "Q2"]
        answers = ["A1", "A2"]
        evaluations = [True, False]
        llm_responses = ["Response1", "Response2"]
        
        results = classifier.classify_results(questions, answers, evaluations, llm_responses)
        
        assert 'solved' in results
        assert 'unsolved' in results
        assert 'statistics' in results