from typing import List, Dict, Any
import json

from ..core.llm_solver import LLMSolver
from ..core.answer_evaluator import AnswerEvaluator


class MultiModelEvaluator:
    def __init__(self, model_names: List[str]):
        self.model_names = model_names
        self.answer_evaluator = AnswerEvaluator()
    
    def evaluate_questions(self, questions: List[str], correct_answers: List[str]) -> Dict[str, Any]:
        question_evaluations = {}
        model_results = {}
        
        for i in range(len(questions)):
            question_evaluations[i] = []
        
        for model_name in self.model_names:
            print(f"Evaluating with model: {model_name}")
            model_evals = self._evaluate_single_model(model_name, questions, correct_answers)
            model_results[model_name] = model_evals
            
            for i, evaluation in enumerate(model_evals):
                question_evaluations[i].append(evaluation)
        
        results = self._calculate_question_success_rates(questions, correct_answers, question_evaluations)
        results['model_results'] = model_results
        results['overall_statistics'] = self._get_overall_statistics(results['question_results'])
        
        return results
    
    def _evaluate_single_model(self, model_name: str, questions: List[str], correct_answers: List[str]) -> List[bool]:
        solver = LLMSolver(model_name)
        solver.initialize_model()
        
        llm_responses = solver.solve_questions_batch(questions)
        evaluations = self.answer_evaluator.evaluate_batch(llm_responses, correct_answers)
        
        return evaluations
    
    def _calculate_question_success_rates(
        self, 
        questions: List[str], 
        correct_answers: List[str], 
        question_evaluations: Dict[int, List[bool]]
    ) -> Dict[str, Any]:
        question_results = []
        
        for i, (question, correct_answer) in enumerate(zip(questions, correct_answers)):
            evaluations = question_evaluations[i]
            success_count = sum(evaluations)
            total_models = len(evaluations)
            success_rate = success_count / total_models if total_models > 0 else 0.0
            
            question_results.append({
                'question_index': i,
                'question': question,
                'correct_answer': correct_answer,
                'success_count': success_count,
                'total_models': total_models,
                'success_rate': success_rate,
                'model_evaluations': evaluations
            })
        
        return {'question_results': question_results}
    
    def _get_overall_statistics(self, question_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_questions = len(question_results)
        if total_questions == 0:
            return {
                'total_questions': 0,
                'average_success_rate': 0.0,
                'fully_solved_questions': 0,
                'unsolved_questions': 0
            }
        
        success_rates = [q['success_rate'] for q in question_results]
        average_success_rate = sum(success_rates) / total_questions
        fully_solved_questions = sum(1 for rate in success_rates if rate == 1.0)
        unsolved_questions = sum(1 for rate in success_rates if rate == 0.0)
        
        return {
            'total_questions': total_questions,
            'average_success_rate': average_success_rate,
            'fully_solved_questions': fully_solved_questions,
            'unsolved_questions': unsolved_questions
        }
    
    def save_results(self, results: Dict[str, Any], filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)