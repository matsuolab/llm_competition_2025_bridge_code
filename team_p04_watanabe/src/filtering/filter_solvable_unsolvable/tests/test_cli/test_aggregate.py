import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from filter_solvable_question.cli.aggregate import ResultAggregator


class TestResultAggregator:
    def test_init_with_results_dir(self):
        results_dir = Path("./test_results")
        aggregator = ResultAggregator(results_dir)
        
        assert aggregator.results_dir == results_dir
    
    def test_load_all_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            
            # Create test result files
            dataset1_dir = base_dir / "dataset1"
            dataset1_dir.mkdir()
            result1 = {
                "results": {"overall_statistics": {"accuracy": 0.8}},
                "metadata": {"dataset_name": "dataset1"}
            }
            (dataset1_dir / "results.json").write_text(json.dumps(result1))
            
            dataset2_dir = base_dir / "dataset2"
            dataset2_dir.mkdir()
            result2 = {
                "results": {"overall_statistics": {"accuracy": 0.6}},
                "metadata": {"dataset_name": "dataset2"}
            }
            (dataset2_dir / "results.json").write_text(json.dumps(result2))
            
            aggregator = ResultAggregator(base_dir)
            results = aggregator.load_all_results()
            
            assert len(results) == 2
            assert any(r['metadata']['dataset_name'] == 'dataset1' for r in results)
            assert any(r['metadata']['dataset_name'] == 'dataset2' for r in results)
    
    def test_aggregate_statistics(self):
        results = [
            {
                "results": {
                    "overall_statistics": {
                        "total_questions": 100,
                        "average_success_rate": 0.8,
                        "fully_solved_questions": 60,
                        "unsolved_questions": 10
                    }
                },
                "metadata": {"dataset_name": "dataset1", "models": ["model1"]}
            },
            {
                "results": {
                    "overall_statistics": {
                        "total_questions": 200,
                        "average_success_rate": 0.6,
                        "fully_solved_questions": 80,
                        "unsolved_questions": 40
                    }
                },
                "metadata": {"dataset_name": "dataset2", "models": ["model1"]}
            }
        ]
        
        aggregator = ResultAggregator(Path("./test"))
        aggregated = aggregator.aggregate_statistics(results)
        
        assert aggregated['total_datasets'] == 2
        assert aggregated['total_questions'] == 300
        assert aggregated['overall_success_rate'] == pytest.approx(0.467, rel=1e-2)
        assert len(aggregated['dataset_summaries']) == 2
    
    def test_generate_summary_report(self):
        aggregated_stats = {
            'total_datasets': 2,
            'total_questions': 300,
            'total_models': 2,
            'overall_success_rate': 0.7,
            'dataset_summaries': [
                {'dataset_name': 'dataset1', 'success_rate': 0.8, 'total_questions': 100, "fully_solved": 60, "unsolved": 10},
                {'dataset_name': 'dataset2', 'success_rate': 0.6, 'total_questions': 200, "fully_solved": 80, "unsolved": 40}
            ]
        }
        
        aggregator = ResultAggregator(Path("./test"))
        report = aggregator.generate_summary_report(aggregated_stats)
        
        assert "Total Datasets: 2" in report
        assert "Total Questions: 300" in report
        assert "Overall Success Rate: 70.0%" in report
        assert "dataset1" in report
        assert "dataset2" in report