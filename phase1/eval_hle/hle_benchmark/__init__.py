import argparse
import hydra

def predict():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cais/hle", help="HLE HF Dataset")
    parser.add_argument("--provider", type=str, default="vllm", choices=["openai", "ollama", "vllm"])
    parser.add_argument("--model", type=str, help="Model Name", required=True)
    parser.add_argument("--max_completion_tokens", type=int, default=None, help="Limit completion tokens. Recommended to avoid model collapse. Not aviable with ollama.")
    parser.add_argument("--reasoning", )
    parser.add_argument("--num_workers", type=int, default=10, help="Async semaphore size. This depends on your rate limit.")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit evaluation to first N samples")
    args = parser.parse_args()

    if args.provider == "openai":
        from hle_benchmark import openai_predictions
        openai_predictions.main(args)
    elif args.provider == "ollama":
        from hle_benchmark import ollama_predictions
        ollama_predictions.main(args)
    elif args.provider == "vllm":
        from hle_benchmark import vllm_predictions
        vllm_predictions.main()

def judge():
    from hle_benchmark.run_judge_results import main
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cais/hle", help="HLE HF Dataset")
    parser.add_argument("--model", type=str, help="Model Name")
    parser.add_argument("--predictions", type=str, help="Model Predictions Path")
    parser.add_argument("--num_workers", type=int, default=100, help="Async semaphore size. This depends on your rate limit.")
    parser.add_argument("--judge", type=str, default="gpt-4.1-mini-2025-04-14", help="Judge model") # prev: "gpt-4o-2024-08-06", "o3-mini-2025-01-31"
    args = parser.parse_args()
    
    if not args.predictions:
        if not args.model:
            print("Arguments either model or predictions is required.")
        args.predictions = f"predictions/hle_{args.model}.json"

    main(args)