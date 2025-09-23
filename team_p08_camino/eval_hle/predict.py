import hydra

@hydra.main(config_name="config", version_base=None, config_path="conf")
def main(cfg):
    if cfg.provider == "openai":
        from hle_benchmark import openai_predictions
        openai_predictions.main(cfg)
    elif cfg.provider == "ollama":
        from hle_benchmark import ollama_predictions
        ollama_predictions.main(cfg)
    elif cfg.provider == "vllm":
        from hle_benchmark import vllm_predictions
        vllm_predictions.main(cfg)

if __name__ == "__main__":
    main()